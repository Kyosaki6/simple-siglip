import argparse
import json
import logging
import os
import sys
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.dataset import VideoSample, load_surveillance_manifest, stratified_group_kfold
from src.features import extract_or_load_features
from src.heads import (
    DEFAULT_TEMPLATES,
    GENERIC_TEMPLATES,
    balanced_class_weights,
    encode_prompt_ensemble,
    macro_f1,
    per_class_f1,
    pool_video_embedding,
    predict,
    save_head,
    train_head,
)
from src.metrics import (
    STEC_TO_MSAD,
    evaluate_stec_probe,
    paired_bootstrap_macro_f1_diff,
    paired_bootstrap_test,
    wilson_score_interval,
)
from src.siglip_engine import SigLIPEngine
from src.video_processor import VideoProcessor

logger = logging.getLogger(__name__)

# Pre-declared (not chosen after seeing MSAD results) adapter recipe.
TUNED_WD_GRID: Tuple[float, ...] = (1e-4, 1e-3, 1e-2)
TUNED_DEFAULT_WD: float = 1e-3
TEXT_INIT_SCALE: float = 10.0
INNER_SPLITS: int = 3
MIN_EFFECT_FOR_GAIN: float = 0.05


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="5-fold cross-validation runner on MSAD with STEC exploratory probe and bootstrap CI."
    )
    parser.add_argument(
        "--msad-root",
        type=str,
        default=os.environ.get("MSAD_ROOT"),
        help="Path to MSAD dataset root folder (default: $MSAD_ROOT).",
    )
    parser.add_argument(
        "--stec-root",
        type=str,
        default=os.environ.get("STEC_ROOT"),
        help="Path to STEC dataset root folder (default: $STEC_ROOT).",
    )
    parser.add_argument("--cache-dir", type=str, default="cache/features", help="Embedding cache directory.")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Model checkpoints directory.")
    parser.add_argument("--out", type=str, default="eval_real_results.json", help="Output path for JSON report.")
    parser.add_argument("--model", type=str, default="google/siglip-base-patch16-224", help="SigLIP model ID.")
    parser.add_argument("--fps", type=float, default=1.0, help="Frame extraction FPS rate.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--group-threshold",
        type=float,
        default=0.90,
        help="Background-correlation threshold for camera grouping (lower = larger groups, stricter CV).",
    )
    parser.add_argument("--n-splits", type=int, default=5, help="Number of cross-validation folds.")
    return parser.parse_args(argv)


def _write_json_atomic(data: Dict[str, Any], path: str) -> None:
    target_dir = os.path.dirname(os.path.abspath(path))
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)
    tmp_path = os.path.join(target_dir, f".tmp_{uuid.uuid4().hex}.json")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _extract_and_pool_all(
    samples: List[VideoSample], engine: Any, processor: Any, cache_dir: str
) -> Tuple[torch.Tensor, int]:
    pooled_vectors: List[torch.Tensor] = []
    total_frames = 0
    for s in samples:
        feat = extract_or_load_features(s.path, engine, processor, cache_dir)
        embeds = torch.from_numpy(feat["embeddings"]).float()
        total_frames += embeds.shape[0]
        pooled_vectors.append(pool_video_embedding(embeds, mode="mean"))
    if not pooled_vectors:
        dim = getattr(getattr(engine, "model", None), "config", None)
        feat_dim = getattr(getattr(dim, "vision_config", None), "hidden_size", 768)
        return torch.empty((0, feat_dim), dtype=torch.float32), 0
    return torch.stack(pooled_vectors, dim=0), total_frames


def _train_text_init_head(
    x: torch.Tensor,
    y: np.ndarray,
    num_classes: int,
    text_init: torch.Tensor,
    weight_decay: float,
    seed: int,
) -> Any:
    head, _ = train_head(
        features=x,
        labels=torch.from_numpy(y),
        num_classes=num_classes,
        weight_decay=weight_decay,
        seed=seed,
        class_weights=balanced_class_weights(y, num_classes),
        init_weight=text_init * TEXT_INIT_SCALE,
    )
    return head


def _tune_weight_decay(
    x: torch.Tensor,
    y: np.ndarray,
    groups: List[int],
    num_classes: int,
    text_init: torch.Tensor,
    seed: int,
) -> float:
    """Pick weight decay by grouped inner CV on the training data only."""
    labels = [str(v) for v in y.tolist()]
    try:
        inner = stratified_group_kfold(labels, groups, n_splits=INNER_SPLITS, seed=seed)
    except ValueError as exc:
        logger.warning("Inner CV unavailable (%s); using default weight_decay=%g", exc, TUNED_DEFAULT_WD)
        return TUNED_DEFAULT_WD

    best_wd, best_score = TUNED_DEFAULT_WD, -1.0
    for wd in TUNED_WD_GRID:
        preds = np.zeros(len(y), dtype=np.int64)
        for tr, va in inner:
            head = _train_text_init_head(x[tr], y[tr], num_classes, text_init, wd, seed)
            p, _ = predict(head, x[va])
            preds[va] = p.cpu().numpy()
        score = macro_f1(y, preds, num_classes)
        if score > best_score:
            best_wd, best_score = wd, score
    return best_wd


def _strongest_baseline(scores: Dict[str, float]) -> str:
    """Name of the best-scoring training-free method (deterministic tie-break by name)."""
    return max(sorted(scores), key=lambda k: scores[k])


def evaluate_surveillance(
    args: argparse.Namespace,
    engine: Optional[Any] = None,
    processor: Optional[Any] = None,
) -> Dict[str, Any]:
    if not args.msad_root or not os.path.isdir(args.msad_root):
        raise FileNotFoundError(f"MSAD root directory does not exist: {args.msad_root}")
    if not args.stec_root or not os.path.isdir(args.stec_root):
        raise FileNotFoundError(f"STEC root directory does not exist: {args.stec_root}")

    manifest = load_surveillance_manifest(
        msad_root=args.msad_root, stec_root=args.stec_root, group_threshold=args.group_threshold
    )
    msad_samples, stec_samples = manifest.get("msad", []), manifest.get("stec", [])
    if not msad_samples:
        raise ValueError(f"No MSAD samples found under {args.msad_root}")

    active_engine = engine if engine is not None else SigLIPEngine(model_name=args.model)
    active_proc = processor if processor is not None else VideoProcessor(target_fps=args.fps)

    msad_embeds, msad_frames = _extract_and_pool_all(msad_samples, active_engine, active_proc, args.cache_dir)
    stec_embeds, stec_frames = _extract_and_pool_all(stec_samples, active_engine, active_proc, args.cache_dir)
    total_frames = msad_frames + stec_frames

    msad_labels = [s.label for s in msad_samples]
    msad_groups = [s.group_id for s in msad_samples]
    if len(set(msad_groups)) > 0.9 * len(msad_samples):
        logger.warning(
            "Camera grouping is nearly a no-op (%d groups for %d videos at threshold %.2f): "
            "CV folds are effectively ungrouped and may leak camera/background. "
            "Lower --group-threshold to test sensitivity.",
            len(set(msad_groups)), len(msad_samples), args.group_threshold,
        )
    class_names = sorted(list(set(msad_labels)))
    label_to_idx = {name: i for i, name in enumerate(class_names)}
    num_classes = len(class_names)
    y_true = np.array([label_to_idx[lbl] for lbl in msad_labels], dtype=np.int64)

    # 1. Zero-shot baseline ("This is a photo of {}.")
    single_prompts = [f"This is a photo of {cls.replace('_', ' ').strip().lower()}." for cls in class_names]
    zs_text_embeds = F.normalize(active_engine.encode_texts(single_prompts).float(), p=2, dim=-1)
    sims_zs, _ = active_engine.score_logits(msad_embeds, zs_text_embeds)
    y_pred_zs = sims_zs.argmax(dim=-1).cpu().numpy()

    # 2. Prompt ensembles: surveillance-worded vs generic control (isolates wording from ensembling)
    ensemble_embeds = encode_prompt_ensemble(class_names, active_engine, templates=DEFAULT_TEMPLATES)
    y_pred_ens = (msad_embeds @ ensemble_embeds.T).argmax(dim=-1).cpu().numpy()
    generic_embeds = encode_prompt_ensemble(class_names, active_engine, templates=GENERIC_TEMPLATES)
    y_pred_gen = (msad_embeds @ generic_embeds.T).argmax(dim=-1).cpu().numpy()

    # 3. 5-Fold adapters (out-of-fold): baseline recipe and pre-declared tuned recipe
    splits = stratified_group_kfold(msad_labels, msad_groups, n_splits=args.n_splits, seed=args.seed)
    y_pred_adapter = np.zeros(len(msad_samples), dtype=np.int64)
    y_pred_tuned = np.zeros(len(msad_samples), dtype=np.int64)
    fold_wds: List[float] = []
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        fold_head, _ = train_head(
            features=msad_embeds[train_idx],
            labels=torch.from_numpy(y_true[train_idx]),
            num_classes=num_classes,
            seed=args.seed + fold_idx,
        )
        preds, _ = predict(fold_head, msad_embeds[test_idx])
        y_pred_adapter[test_idx] = preds.cpu().numpy()

        # Text-init, class-balanced, weight decay tuned on the training fold only.
        wd = _tune_weight_decay(
            msad_embeds[train_idx], y_true[train_idx], [msad_groups[i] for i in train_idx],
            num_classes, ensemble_embeds, args.seed + fold_idx,
        )
        fold_wds.append(wd)
        tuned_head = _train_text_init_head(
            msad_embeds[train_idx], y_true[train_idx], num_classes, ensemble_embeds, wd, args.seed + fold_idx
        )
        preds_t, _ = predict(tuned_head, msad_embeds[test_idx])
        y_pred_tuned[test_idx] = preds_t.cpu().numpy()

    # Metrics. All bootstraps are cluster bootstraps over camera groups.
    f1 = {
        "zero_shot": macro_f1(y_true, y_pred_zs, num_classes),
        "prompt_ensemble": macro_f1(y_true, y_pred_ens, num_classes),
        "generic_ensemble": macro_f1(y_true, y_pred_gen, num_classes),
    }
    f1_ad = macro_f1(y_true, y_pred_adapter, num_classes)
    f1_tuned = macro_f1(y_true, y_pred_tuned, num_classes)
    preds_by_name = {
        "zero_shot": y_pred_zs, "prompt_ensemble": y_pred_ens, "generic_ensemble": y_pred_gen,
        "adapter": y_pred_adapter, "adapter_tuned": y_pred_tuned,
    }

    def boot(a: str, b: str) -> Tuple[float, float, float]:
        return paired_bootstrap_macro_f1_diff(
            y_true=y_true, y_a=preds_by_name[a], y_b=preds_by_name[b],
            num_classes=num_classes, seed=args.seed, groups=msad_groups,
        )

    diff, ci_lower, ci_upper = paired_bootstrap_test(
        y_true=y_true, y_adapter=y_pred_adapter, y_zero_shot=y_pred_zs,
        num_classes=num_classes, seed=args.seed, groups=msad_groups,
    )

    pairwise: Dict[str, Dict[str, Any]] = {}
    for a, b in (
        ("prompt_ensemble", "zero_shot"),
        ("generic_ensemble", "zero_shot"),
        ("prompt_ensemble", "generic_ensemble"),
        ("adapter", "prompt_ensemble"),
        ("adapter_tuned", "zero_shot"),
        ("adapter_tuned", "prompt_ensemble"),
        ("adapter_tuned", "adapter"),
    ):
        d, lo, hi = boot(a, b)
        pairwise[f"{a}_minus_{b}"] = {"diff": round(d, 4), "ci_95": [lo, hi], "significant": bool(lo > 0 or hi < 0)}

    # Verdict: the adapter must beat the STRONGEST training-free baseline, not just zero-shot.
    baseline_name = _strongest_baseline(f1)
    verdicts: Dict[str, str] = {}
    for name in ("adapter", "adapter_tuned"):
        d, lo, _ = boot(name, baseline_name)
        verdicts[name] = "gain" if (d >= MIN_EFFECT_FOR_GAIN and lo > 0) else "no_gain"
    verdict = verdicts["adapter_tuned"]
    verdict_vs_zero_shot = "gain" if (diff >= MIN_EFFECT_FOR_GAIN and ci_lower > 0) else "no_gain"

    # Final heads trained on all MSAD samples. Tuned head is the checkpoint (pre-declared primary).
    final_wd = _tune_weight_decay(msad_embeds, y_true, msad_groups, num_classes, ensemble_embeds, args.seed)
    final_head = _train_text_init_head(msad_embeds, y_true, num_classes, ensemble_embeds, final_wd, args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, "adapter_msad_base224.npz")
    save_head(
        path=ckpt_path,
        head=final_head,
        class_names=class_names,
        hyperparams={
            "model": args.model, "fps": args.fps, "seed": args.seed, "n_splits": args.n_splits,
            "recipe": "text_init_balanced_tuned_wd", "weight_decay": final_wd,
            "text_init_scale": TEXT_INIT_SCALE,
        },
        model_name=args.model,
        label_map=STEC_TO_MSAD,
        metrics={
            "macro_f1": f1_tuned, "baseline": baseline_name,
            "verdict": verdict, "pairwise": pairwise,
        },
    )

    stec_probe = evaluate_stec_probe(
        stec_samples, stec_embeds, final_head, active_engine, zs_text_embeds, label_to_idx, STEC_TO_MSAD
    )

    per_class_metrics = {
        cls: {
            "zero_shot_f1": round(float(a), 4),
            "prompt_ensemble_f1": round(float(b), 4),
            "generic_ensemble_f1": round(float(c), 4),
            "adapter_f1": round(float(d_), 4),
            "adapter_tuned_f1": round(float(e), 4),
        }
        for cls, a, b, c, d_, e in zip(
            class_names,
            per_class_f1(y_true, y_pred_zs, num_classes),
            per_class_f1(y_true, y_pred_ens, num_classes),
            per_class_f1(y_true, y_pred_gen, num_classes),
            per_class_f1(y_true, y_pred_adapter, num_classes),
            per_class_f1(y_true, y_pred_tuned, num_classes),
        )
    }

    report = {
        "msad_5fold_pooled_macro_f1": {
            "zero_shot": round(f1["zero_shot"], 4),
            "prompt_ensemble": round(f1["prompt_ensemble"], 4),
            "generic_ensemble": round(f1["generic_ensemble"], 4),
            "adapter": round(f1_ad, 4),
            "adapter_tuned": round(f1_tuned, 4),
        },
        "bootstrap_method": "cluster bootstrap over camera groups (group_id), 1000 resamples",
        "bootstrap_ci_adapter_minus_zero_shot": [ci_lower, ci_upper],
        "strongest_baseline": baseline_name,
        "verdict": verdict,
        "verdict_adapter_baseline_recipe": verdicts["adapter"],
        "verdict_vs_zero_shot": verdict_vs_zero_shot,
        "tuned_adapter_fold_weight_decay": fold_wds,
        "pairwise_bootstrap": pairwise,
        "stec_cross_probe": stec_probe,
        "per_class_metrics": per_class_metrics,
        "n_videos": len(msad_samples),
        "n_groups": len(set(msad_groups)),
        "group_threshold": args.group_threshold,
        "largest_group_size": int(np.bincount(np.unique(msad_groups, return_inverse=True)[1]).max()),
        "n_frames": int(total_frames),
        "model_id": args.model,
        "seed": args.seed,
    }
    _write_json_atomic(report, args.out)
    logger.info("Evaluation complete. Report written to %s (verdict=%s vs %s)", args.out, verdict, baseline_name)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = parse_args(argv)
    if not args.msad_root or not os.path.isdir(args.msad_root):
        logger.error("MSAD root directory missing or does not exist (set --msad-root or $MSAD_ROOT): %s", args.msad_root)
        return 1
    if not args.stec_root or not os.path.isdir(args.stec_root):
        logger.error("STEC root directory missing or does not exist (set --stec-root or $STEC_ROOT): %s", args.stec_root)
        return 1
    try:
        evaluate_surveillance(args)
        return 0
    except Exception as exc:
        logger.error("Evaluation pipeline failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
