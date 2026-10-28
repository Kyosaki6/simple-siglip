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


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="5-fold cross-validation runner on MSAD with STEC exploratory probe and bootstrap CI."
    )
    parser.add_argument(
        "--msad-root",
        type=str,
        default="/home/thinh/D/Project/NCKH/MSAD_anomaly-20261022T151528Z-1-001/MSAD_anomaly",
        help="Path to MSAD dataset root folder.",
    )
    parser.add_argument(
        "--stec-root",
        type=str,
        default="/home/thinh/D/Project/NCKH/Data_stec-20261022T151522Z-1-006/Data_stec",
        help="Path to STEC dataset root folder.",
    )
    parser.add_argument("--cache-dir", type=str, default="cache/features", help="Embedding cache directory.")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Model checkpoints directory.")
    parser.add_argument("--out", type=str, default="eval_real_results.json", help="Output path for JSON report.")
    parser.add_argument("--model", type=str, default="google/siglip-base-patch16-224", help="SigLIP model ID.")
    parser.add_argument("--fps", type=float, default=1.0, help="Frame extraction FPS rate.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
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


def evaluate_surveillance(
    args: argparse.Namespace,
    engine: Optional[Any] = None,
    processor: Optional[Any] = None,
) -> Dict[str, Any]:
    if not os.path.isdir(args.msad_root):
        raise FileNotFoundError(f"MSAD root directory does not exist: {args.msad_root}")
    if not os.path.isdir(args.stec_root):
        raise FileNotFoundError(f"STEC root directory does not exist: {args.stec_root}")

    manifest = load_surveillance_manifest(msad_root=args.msad_root, stec_root=args.stec_root)
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
    class_names = sorted(list(set(msad_labels)))
    label_to_idx = {name: i for i, name in enumerate(class_names)}
    num_classes = len(class_names)
    y_true = np.array([label_to_idx[lbl] for lbl in msad_labels], dtype=np.int64)

    # 1. Zero-shot baseline ("This is a photo of {}.")
    single_prompts = [f"This is a photo of {cls.replace('_', ' ').strip().lower()}." for cls in class_names]
    zs_text_embeds = F.normalize(active_engine.encode_texts(single_prompts).float(), p=2, dim=-1)
    sims_zs, _ = active_engine.score_logits(msad_embeds, zs_text_embeds)
    y_pred_zs = sims_zs.argmax(dim=-1).cpu().numpy()

    # 2. Prompt ensemble zero-shot
    ensemble_embeds = encode_prompt_ensemble(class_names, active_engine, templates=DEFAULT_TEMPLATES)
    sims_ens = msad_embeds @ ensemble_embeds.T
    y_pred_ens = sims_ens.argmax(dim=-1).cpu().numpy()

    # 3. 5-Fold Linear Adapter Head (out-of-fold evaluation)
    splits = stratified_group_kfold(msad_labels, msad_groups, n_splits=args.n_splits, seed=args.seed)
    y_pred_adapter = np.zeros(len(msad_samples), dtype=np.int64)
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        fold_head, _ = train_head(
            features=msad_embeds[train_idx],
            labels=torch.from_numpy(y_true[train_idx]),
            num_classes=num_classes,
            seed=args.seed + fold_idx,
        )
        preds, _ = predict(fold_head, msad_embeds[test_idx])
        y_pred_adapter[test_idx] = preds.cpu().numpy()

    # Metrics & bootstrap test
    f1_zs = macro_f1(y_true, y_pred_zs, num_classes)
    f1_ens = macro_f1(y_true, y_pred_ens, num_classes)
    f1_ad = macro_f1(y_true, y_pred_adapter, num_classes)
    diff, ci_lower, ci_upper = paired_bootstrap_test(
        y_true=y_true, y_adapter=y_pred_adapter, y_zero_shot=y_pred_zs, num_classes=num_classes, seed=args.seed
    )
    verdict = "gain" if (diff >= 0.05 and ci_lower > 0) else "no_gain"
    pairwise: Dict[str, Dict[str, Any]] = {}
    for name, (y_a, y_b) in {
        "prompt_ensemble_minus_zero_shot": (y_pred_ens, y_pred_zs),
        "adapter_minus_prompt_ensemble": (y_pred_adapter, y_pred_ens),
    }.items():
        d, lo, hi = paired_bootstrap_macro_f1_diff(
            y_true=y_true, y_a=y_a, y_b=y_b, num_classes=num_classes, seed=args.seed
        )
        pairwise[name] = {"diff": round(d, 4), "ci_95": [lo, hi], "significant": bool(lo > 0 or hi < 0)}

    # Train final adapter head on all MSAD samples & save checkpoint
    final_head, _ = train_head(
        features=msad_embeds, labels=torch.from_numpy(y_true), num_classes=num_classes, seed=args.seed
    )
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, "adapter_msad_base224.pt")
    save_head(
        path=ckpt_path,
        head=final_head,
        class_names=class_names,
        hyperparams={"model": args.model, "fps": args.fps, "seed": args.seed, "n_splits": args.n_splits},
        model_name=args.model,
        label_map=STEC_TO_MSAD,
        metrics={"macro_f1": f1_ad, "diff_vs_zs": diff, "ci": [ci_lower, ci_upper]},
    )

    stec_probe = evaluate_stec_probe(
        stec_samples, stec_embeds, final_head, active_engine, zs_text_embeds, label_to_idx, STEC_TO_MSAD
    )

    per_class_metrics = {
        cls: {
            "zero_shot_f1": round(float(f1_zs_c), 4),
            "prompt_ensemble_f1": round(float(f1_ens_c), 4),
            "adapter_f1": round(float(f1_ad_c), 4),
        }
        for cls, f1_zs_c, f1_ens_c, f1_ad_c in zip(
            class_names,
            per_class_f1(y_true, y_pred_zs, num_classes),
            per_class_f1(y_true, y_pred_ens, num_classes),
            per_class_f1(y_true, y_pred_adapter, num_classes),
        )
    }

    report = {
        "msad_5fold_pooled_macro_f1": {
            "zero_shot": round(f1_zs, 4),
            "prompt_ensemble": round(f1_ens, 4),
            "adapter": round(f1_ad, 4),
        },
        "bootstrap_ci_adapter_minus_zero_shot": [ci_lower, ci_upper],
        "verdict": verdict,
        "pairwise_bootstrap": pairwise,
        "stec_cross_probe": stec_probe,
        "per_class_metrics": per_class_metrics,
        "n_videos": len(msad_samples),
        "n_frames": int(total_frames),
        "model_id": args.model,
        "seed": args.seed,
    }
    _write_json_atomic(report, args.out)
    logger.info("Evaluation complete. Report written to %s (verdict=%s)", args.out, verdict)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = parse_args(argv)
    if not os.path.isdir(args.msad_root):
        logger.error("MSAD root directory does not exist: %s", args.msad_root)
        return 1
    if not os.path.isdir(args.stec_root):
        logger.error("STEC root directory does not exist: %s", args.stec_root)
        return 1
    try:
        evaluate_surveillance(args)
        return 0
    except Exception as exc:
        logger.error("Evaluation pipeline failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
