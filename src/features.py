import logging
import math
import os
import re
import unicodedata
import uuid
import zipfile
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

from src.dataset import compute_sha256
from src.siglip_engine import SigLIPEngine
from src.video_processor import VideoProcessor

logger = logging.getLogger(__name__)


def sanitize_model_name(model_name: str) -> str:
    """Sanitize model identifier for filesystem paths."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", model_name)


def get_feature_cache_path(
    cache_dir: str,
    model_name: str,
    video_sha256: str,
    target_fps: float,
    max_frames: Optional[int],
) -> str:
    """Construct deterministic cache file path for video features."""
    sanitized = sanitize_model_name(model_name)
    return os.path.join(cache_dir, sanitized, f"{video_sha256}_fps{target_fps}_max{max_frames}.npz")


def _read_video_duration(video_path: str) -> float:
    """Read duration from video metadata in seconds."""
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            return 0.0
        fc, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
        return round(float(fc / fps), 2) if fc > 0 and fps > 0 else 0.0
    finally:
        cap.release()


def _validate_and_load_cache(
    cache_path: str,
    engine: SigLIPEngine,
    processor: VideoProcessor,
    video_path: str,
) -> Optional[Dict[str, Any]]:
    """Validate cached .npz features file and return payload if valid, else None."""
    if not os.path.exists(cache_path):
        return None

    try:
        with np.load(cache_path, allow_pickle=False) as data:
            keys = ("embeddings", "timestamps", "video_path", "duration", "model_name", "target_fps", "max_frames")
            if any(k not in data for k in keys):
                raise KeyError("Missing required keys in cache")

            stored_model = str(data["model_name"])
            stored_fps = float(data["target_fps"])
            stored_max = int(data["max_frames"])
            expected_max = -1 if processor.max_frames is None else int(processor.max_frames)
            if (
                stored_model != engine.model_name
                or not math.isclose(stored_fps, float(processor.target_fps), rel_tol=1e-5, abs_tol=1e-5)
                or stored_max != expected_max
            ):
                raise ValueError("Cached model_name or sampling config mismatch")

            embeddings, timestamps = data["embeddings"], data["timestamps"]
            expected_dim = int(engine.model.config.vision_config.hidden_size)
            if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2 or embeddings.shape[1] != expected_dim or embeddings.shape[0] <= 0:
                raise ValueError("Invalid embeddings shape in cache")
            if not np.issubdtype(embeddings.dtype, np.floating) or not np.all(np.isfinite(embeddings)):
                raise ValueError("Invalid embeddings values in cache")
            if not isinstance(timestamps, np.ndarray) or timestamps.ndim != 1 or len(timestamps) != len(embeddings) or not np.all(np.isfinite(timestamps)):
                raise ValueError("Invalid timestamps in cache")
            if np.any(np.abs(np.linalg.norm(embeddings, axis=-1) - 1.0) >= 1e-3):
                raise ValueError("Embeddings rows are not unit norm")

            return {
                "embeddings": embeddings.astype(np.float32),
                "timestamps": timestamps.astype(np.float32),
                "video_path": video_path,
                "duration": float(data["duration"]),
                "model_name": stored_model,
                "cache_hit": True,
            }
    except (zipfile.BadZipFile, ValueError, KeyError, EOFError, OSError, Exception) as exc:
        logger.warning("Invalid or corrupt cache file %s: %s. Removing and re-extracting.", cache_path, exc)
        if os.path.exists(cache_path):
            try:
                os.remove(cache_path)
            except OSError:
                pass
        return None


def _write_cache_atomic(
    cache_path: str,
    embeddings: np.ndarray,
    timestamps: np.ndarray,
    video_path: str,
    duration: float,
    model_name: str,
    target_fps: float,
    max_frames: Optional[int],
) -> None:
    """Write embeddings and metadata atomically via temporary file and os.replace."""
    target_dir = os.path.dirname(cache_path)
    os.makedirs(target_dir, exist_ok=True)
    tmp_path = os.path.join(target_dir, f".tmp_{uuid.uuid4().hex}.npz")

    try:
        np.savez(
            tmp_path,
            embeddings=embeddings.astype(np.float32),
            timestamps=timestamps.astype(np.float32),
            video_path=np.array(video_path),
            duration=np.float32(duration),
            model_name=np.array(model_name),
            target_fps=np.float32(target_fps),
            max_frames=np.int64(-1 if max_frames is None else max_frames),
            format_version=np.int32(1),
        )
        os.replace(tmp_path, cache_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def extract_or_load_features(
    video_path: str,
    engine: SigLIPEngine,
    processor: VideoProcessor,
    cache_dir: str,
    batch_size: int = 16,
) -> Dict[str, Any]:
    """
    Extract or load cached normalized embeddings and timestamps for a video.

    Returns:
        Dict containing embeddings (float32 N,D), timestamps (float32 N,),
        video_path, duration, model_name, and cache_hit flag.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    norm_path = unicodedata.normalize("NFC", video_path)
    target_video_path = norm_path if os.path.exists(norm_path) else video_path
    if not os.path.exists(target_video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    video_sha256 = compute_sha256(target_video_path)
    cache_path = get_feature_cache_path(
        cache_dir=cache_dir,
        model_name=engine.model_name,
        video_sha256=video_sha256,
        target_fps=processor.target_fps,
        max_frames=processor.max_frames,
    )

    cached_result = _validate_and_load_cache(cache_path, engine, processor, video_path)
    if cached_result is not None:
        return cached_result

    batch_images: List[Image.Image] = []
    batch_timestamps: List[float] = []
    all_embeds: List[np.ndarray] = []
    all_timestamps: List[float] = []

    def _flush_batch() -> None:
        if batch_images:
            tensor_embeds = engine.encode_images(batch_images)
            all_embeds.append(tensor_embeds.cpu().numpy().astype(np.float32))
            all_timestamps.extend(batch_timestamps)
            batch_images.clear()
            batch_timestamps.clear()

    for item in processor.iter_frames(target_video_path):
        batch_images.append(item["image"])
        batch_timestamps.append(float(item["timestamp"]))
        if len(batch_images) >= batch_size:
            _flush_batch()
    _flush_batch()

    if not all_embeds:
        raise ValueError(f"Video contains zero extractable frames: {video_path}")

    embeddings = np.concatenate(all_embeds, axis=0).astype(np.float32)
    timestamps = np.array(all_timestamps, dtype=np.float32)

    duration = _read_video_duration(target_video_path)
    if duration <= 0.0 and len(all_timestamps) > 0:
        duration = round(float(all_timestamps[-1]), 2)

    _write_cache_atomic(
        cache_path=cache_path,
        embeddings=embeddings,
        timestamps=timestamps,
        video_path=video_path,
        duration=duration,
        model_name=engine.model_name,
        target_fps=processor.target_fps,
        max_frames=processor.max_frames,
    )

    return {
        "embeddings": embeddings,
        "timestamps": timestamps,
        "video_path": video_path,
        "duration": duration,
        "model_name": engine.model_name,
        "cache_hit": False,
    }


def extract_many(
    video_paths: List[str],
    engine: SigLIPEngine,
    processor: VideoProcessor,
    cache_dir: str,
    on_error: Optional[Callable[..., Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Extract features for multiple videos, skipping already cached entries.

    Continues past single video errors, recording {'error': str(exc)}
    for failed videos in the returned dictionary.
    """
    results: Dict[str, Dict[str, Any]] = {}
    total = len(video_paths)

    for i, path in enumerate(video_paths, start=1):
        try:
            res = extract_or_load_features(path, engine, processor, cache_dir)
            logger.info("%d/%d %s cache_hit=%s", i, total, path, res.get("cache_hit", False))
            results[path] = res
        except Exception as exc:
            logger.warning("%d/%d %s error: %s", i, total, path, exc)
            if on_error is not None:
                try:
                    on_error(path, exc)
                except TypeError:
                    on_error(exc)
            results[path] = {"error": str(exc)}

    return results
