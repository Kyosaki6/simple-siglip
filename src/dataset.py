import hashlib
import logging
import os
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

LABEL_MAP: Dict[str, str] = {
    # MSAD 11 canonical classes
    "Assault": "Assault",
    "Explosion": "Explosion",
    "Fighting": "Fighting",
    "Fire": "Fire",
    "Object_falling": "Object_falling",
    "People_falling": "People_falling",
    "Robbery": "Robbery",
    "Shooting": "Shooting",
    "Traffic_accident": "Traffic_accident",
    "Vandalism": "Vandalism",
    "Water_incident": "Water_incident",
    # STEC Vietnamese folder mappings (NFC normalized)
    unicodedata.normalize("NFC", "Ăn trộm"): "theft",
    unicodedata.normalize("NFC", "TRỘM CẮP"): "theft",
    unicodedata.normalize("NFC", "Đánh nhau"): "fighting",
    unicodedata.normalize("NFC", "NGÃ"): "people_falling",
    unicodedata.normalize("NFC", "Fallen_person"): "people_falling",
    "Security_guard_sitting_in_motorbike": "Security_guard_sitting_in_motorbike",
    "Security_guard_smoking": "Security_guard_smoking",
    "Security_guard_unprofessional_posture": "Security_guard_unprofessional_posture",
}

MSAD_CLASSES: Tuple[str, ...] = (
    "Assault", "Explosion", "Fighting", "Fire", "Object_falling", "People_falling",
    "Robbery", "Shooting", "Traffic_accident", "Vandalism", "Water_incident",
)


@dataclass(frozen=True)
class VideoSample:
    path: str
    label: str
    source: str
    sha256: str
    group_id: int
    duration_sec: float
    fps: float


def compute_sha256(file_path: str, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as fp:
        while chunk := fp.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def extract_video_thumbnail_and_meta(
    file_path: str,
    target_size: Tuple[int, int] = (32, 18),
) -> Optional[Tuple[float, float, np.ndarray]]:
    cap = cv2.VideoCapture(file_path)
    try:
        if not cap.isOpened():
            logger.warning("Cannot open video: %s", file_path)
            return None
        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fc <= 0 or fps <= 0:
            logger.warning("Zero-frame or invalid FPS video: %s (fc=%d, fps=%.2f)", file_path, fc, fps)
            return None

        mid = max(0, fc // 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()

        if not ret or frame is None:
            logger.warning("Failed to decode frame from video: %s", file_path)
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thumb = cv2.resize(gray, target_size).astype(np.float32).flatten()
        duration_sec = round(fc / fps, 2)
        fps_clean = round(fps, 2)
        return duration_sec, fps_clean, thumb
    finally:
        cap.release()


def group_by_background_correlation(features: np.ndarray, threshold: float = 0.90) -> List[int]:
    n = len(features)
    if n == 0:
        return []
    if n == 1:
        return [0]

    x = np.asarray(features, dtype=np.float64)
    x_centered = x - np.mean(x, axis=1, keepdims=True)
    norms = np.linalg.norm(x_centered, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1e-12, norms)
    x_norm = x_centered / norms
    corr = x_norm @ x_norm.T

    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    rows, cols = np.nonzero(np.triu(corr > threshold, k=1))
    for i, j in zip(rows.tolist(), cols.tolist()):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    return [find(i) for i in range(n)]


def stratified_group_kfold(
    labels: List[str],
    groups: List[int],
    n_splits: int = 5,
    seed: int = 0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    n_samples = len(labels)
    if n_samples != len(groups):
        raise ValueError(f"Length mismatch: {n_samples} labels vs {len(groups)} groups")
    if n_samples == 0:
        raise ValueError("Cannot fold empty dataset")
    if n_splits < 2:
        raise ValueError(f"n_splits must be at least 2, got {n_splits}")

    unique_labels = sorted(set(labels))
    label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
    n_classes = len(unique_labels)

    group_map: Dict[int, Dict[str, Any]] = {}
    for idx, (lbl, grp) in enumerate(zip(labels, groups)):
        entry = group_map.setdefault(grp, {"indices": [], "counts": np.zeros(n_classes, dtype=np.int64)})
        entry["indices"].append(idx)
        entry["counts"][label_to_idx[lbl]] += 1

    n_groups = len(group_map)
    largest = max(len(e["indices"]) for e in group_map.values())
    if largest > n_samples / n_splits:
        logger.warning(
            "Largest group holds %d of %d samples (> 1/%d): folds will be unbalanced. "
            "Background grouping may be chaining unrelated cameras; consider a stricter threshold.",
            largest, n_samples, n_splits,
        )
    if n_splits > n_groups:
        raise ValueError(f"Cannot have n_splits={n_splits} greater than groups={n_groups}")

    unique_groups = sorted(group_map.keys())
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(unique_groups))
    shuffled_groups = [unique_groups[i] for i in perm]
    shuffled_groups.sort(key=lambda g: len(group_map[g]["indices"]), reverse=True)

    total_label_counts = np.zeros(n_classes, dtype=np.float64)
    for g in unique_groups:
        total_label_counts += group_map[g]["counts"]
    target_counts = total_label_counts / n_splits

    fold_counts = np.zeros((n_splits, n_classes), dtype=np.float64)
    fold_samples: List[List[int]] = [[] for _ in range(n_splits)]

    for g in shuffled_groups:
        g_counts = group_map[g]["counts"]
        best_fold = -1
        best_metric = (float("inf"), float("inf"), -1)

        for f in range(n_splits):
            curr_diff = fold_counts[f] - target_counts
            new_diff = curr_diff + g_counts
            delta_sq_err = float(np.sum(new_diff ** 2) - np.sum(curr_diff ** 2))
            fold_size = float(len(fold_samples[f]))
            metric = (delta_sq_err, fold_size, f)
            if metric < best_metric:
                best_metric = metric
                best_fold = f

        fold_counts[best_fold] += g_counts
        fold_samples[best_fold].extend(group_map[g]["indices"])

    all_indices = np.arange(n_samples, dtype=np.int64)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for f in range(n_splits):
        test_idx = np.array(sorted(fold_samples[f]), dtype=np.int64)
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[test_idx] = False
        train_idx = all_indices[train_mask]
        splits.append((train_idx, test_idx))

    return splits


def _scan_mp4_files(root_dir: str) -> List[str]:
    files = [os.path.join(r, f) for r, _, fs in os.walk(root_dir) for f in fs if f.lower().endswith(".mp4")]
    return sorted(files)


def _process_video_item(path: str) -> Optional[Tuple[str, str, float, float, np.ndarray]]:
    try:
        sha256 = compute_sha256(path)
        meta = extract_video_thumbnail_and_meta(path)
        if meta is None:
            return None
        dur, fps, thumb = meta
        return path, sha256, dur, fps, thumb
    except Exception as exc:
        logger.warning("Error reading video %s: %s", path, exc)
        return None


def _resolve_stec_label(rel_path: str) -> Optional[str]:
    parts = Path(rel_path).parts
    if parts[-1].lower().startswith("smartbench_"):
        return None
    for part in reversed(parts[:-1]):
        nfc_part = unicodedata.normalize("NFC", part).strip()
        if nfc_part in LABEL_MAP:
            return LABEL_MAP[nfc_part]
    return None


def _load_partition(
    root_dir: Optional[str],
    source: str,
    workers: int,
    include_unlabeled: bool = False,
    group_threshold: float = 0.90,
) -> Tuple[List[VideoSample], List[VideoSample]]:
    if root_dir is None:
        logger.warning("%s root is None; returning empty list", source.upper())
        return [], []
    if not os.path.isdir(root_dir):
        logger.warning("%s root directory does not exist: %s", source.upper(), root_dir)
        return [], []

    paths = _scan_mp4_files(root_dir)
    logger.info("Discovered %d %s video files", len(paths), source.upper())
    with ThreadPoolExecutor(max_workers=workers) as executor:
        raw_results = list(executor.map(_process_video_item, paths))

    seen_hashes: Dict[str, str] = {}
    valid_items: List[Tuple[str, str, str, float, float, np.ndarray]] = []
    unlabeled: List[VideoSample] = []

    for item in raw_results:
        if item is None:
            continue
        path, sha, dur, fps, thumb = item
        if sha in seen_hashes:
            logger.info("%s duplicate skipped: %s (matches %s)", source.upper(), path, seen_hashes[sha])
            continue
        seen_hashes[sha] = path
        rel = os.path.relpath(path, root_dir)

        if source == "msad":
            parts = Path(rel).parts
            top_folder = unicodedata.normalize("NFC", parts[0]) if len(parts) > 1 else ""
            if top_folder not in MSAD_CLASSES:
                logger.warning("MSAD video outside a known class folder skipped: %s", path)
                continue
            label = LABEL_MAP[top_folder]
        else:
            label = _resolve_stec_label(rel)
            if label is None:
                if include_unlabeled:
                    unlabeled.append(VideoSample(path, "unlabeled", source, sha, len(unlabeled), dur, fps))
                continue

        valid_items.append((path, label, sha, dur, fps, thumb))

    samples: List[VideoSample] = []
    if valid_items:
        thumbs = np.stack([v[5] for v in valid_items])
        group_ids = group_by_background_correlation(thumbs, threshold=group_threshold)
        for (p, lbl, sha, dur, fps, _), gid in zip(valid_items, group_ids):
            samples.append(VideoSample(p, lbl, source, sha, gid, dur, fps))

    return samples, unlabeled


def load_surveillance_manifest(
    msad_root: Optional[str] = None,
    stec_root: Optional[str] = None,
    include_unlabeled: bool = False,
    max_workers: Optional[int] = None,
    group_threshold: float = 0.90,
) -> Dict[str, List[VideoSample]]:
    t_start = time.time()
    workers = max_workers or min(12, (os.cpu_count() or 4) * 2)

    msad_samples, _ = _load_partition(msad_root, "msad", workers, False, group_threshold)
    stec_samples, unlabeled = _load_partition(stec_root, "stec", workers, include_unlabeled, group_threshold)

    logger.info("Manifest loaded in %.2fs: %d MSAD, %d STEC", time.time() - t_start, len(msad_samples), len(stec_samples))
    manifest: Dict[str, List[VideoSample]] = {"msad": msad_samples, "stec": stec_samples}
    if include_unlabeled:
        manifest["unlabeled"] = unlabeled
    return manifest
