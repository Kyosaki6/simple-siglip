import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.dataset import VideoSample
from src.heads import macro_f1, predict

STEC_TO_MSAD: Dict[str, str] = {
    "fighting": "Fighting",
    "people_falling": "People_falling",
    "theft": "Robbery",
}


def wilson_score_interval(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate Wilson binomial score confidence interval (no scipy/sklearn)."""
    if n <= 0:
        return 0.0, 0.0
    z = 1.959963984540054 if abs(confidence - 0.95) < 1e-4 else 1.96
    p_hat = float(k) / float(n)
    denom = 1.0 + (z**2) / n
    center = (p_hat + (z**2) / (2.0 * n)) / denom
    half_width = (z * math.sqrt(max(0.0, (p_hat * (1.0 - p_hat) / n) + (z**2) / (4.0 * (n**2))))) / denom
    return round(max(0.0, center - half_width), 4), round(min(1.0, center + half_width), 4)


def paired_bootstrap_test(
    y_true: np.ndarray,
    y_adapter: np.ndarray,
    y_zero_shot: np.ndarray,
    num_classes: int,
    n_resamples: int = 1000,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Compute paired bootstrap difference and 95% CI for macro_f1(adapter) - macro_f1(zero_shot)."""
    n_samples = len(y_true)
    diff = macro_f1(y_true, y_adapter, num_classes) - macro_f1(y_true, y_zero_shot, num_classes)
    rng = np.random.default_rng(seed)
    deltas = np.zeros(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        deltas[b] = macro_f1(y_true[idx], y_adapter[idx], num_classes) - macro_f1(y_true[idx], y_zero_shot[idx], num_classes)
    return float(diff), round(float(np.percentile(deltas, 2.5)), 4), round(float(np.percentile(deltas, 97.5)), 4)


def evaluate_stec_probe(
    stec_samples: List[VideoSample],
    stec_embeds: torch.Tensor,
    final_head: Any,
    engine: Any,
    zs_text_embeds: torch.Tensor,
    label_to_idx: Dict[str, int],
    mapping: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Evaluate cross-domain probe on STEC videos with Wilson 95% score CIs."""
    active_mapping = mapping if mapping is not None else STEC_TO_MSAD
    stec_per_class: Dict[str, Any] = {}
    mapped_correct_zs, mapped_correct_ad, total_mapped = 0, 0, 0

    if len(stec_samples) > 0 and stec_embeds.shape[0] > 0:
        stec_preds_ad, _ = predict(final_head, stec_embeds)
        stec_sims_zs, _ = engine.score_logits(stec_embeds, zs_text_embeds)
        stec_preds_zs = stec_sims_zs.argmax(dim=-1).cpu().numpy()

        for stec_cls, msad_target in active_mapping.items():
            if msad_target not in label_to_idx:
                continue
            t_idx = label_to_idx[msad_target]
            cls_indices = [i for i, s in enumerate(stec_samples) if s.label == stec_cls]
            n_c = len(cls_indices)
            k_zs = sum(1 for i in cls_indices if stec_preds_zs[i] == t_idx)
            k_ad = sum(1 for i in cls_indices if stec_preds_ad[i] == t_idx)
            mapped_correct_zs += k_zs
            mapped_correct_ad += k_ad
            total_mapped += n_c
            stec_per_class[stec_cls] = {
                "msad_class": msad_target,
                "n": n_c,
                "zero_shot": {
                    "correct": k_zs,
                    "accuracy": round(k_zs / n_c, 4) if n_c else 0.0,
                    "wilson_95_ci": list(wilson_score_interval(k_zs, n_c)),
                },
                "adapter": {
                    "correct": k_ad,
                    "accuracy": round(k_ad / n_c, 4) if n_c else 0.0,
                    "wilson_95_ci": list(wilson_score_interval(k_ad, n_c)),
                },
            }

    return {
        "disclaimer": "Sample size is small (n < 10 per class, total n=14); results lack statistical significance and are exploratory.",
        "mapped_classes": active_mapping,
        "per_class": stec_per_class,
        "overall_mapped": {
            "n_samples": total_mapped,
            "zero_shot_accuracy": round(mapped_correct_zs / total_mapped, 4) if total_mapped else 0.0,
            "adapter_accuracy": round(mapped_correct_ad / total_mapped, 4) if total_mapped else 0.0,
        },
        "total_stec_videos": len(stec_samples),
    }
