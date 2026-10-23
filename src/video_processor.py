import math
import os
import shutil
import subprocess
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image

DEFAULT_FALLBACK_FPS: float = 25.0


class VideoProcessor:
    """Extract and sample frames from video files with timestamp tracking."""

    def __init__(self, target_fps: float = 1.0, max_frames: Optional[int] = None) -> None:
        """
        Args:
            target_fps: Number of frames to extract per second of video.
            max_frames: Maximum total frames to extract (optional cap).
        """
        if not math.isfinite(target_fps) or target_fps <= 0:
            raise ValueError(f"target_fps must be a finite positive number, got {target_fps}")
        if max_frames is not None and max_frames <= 0:
            raise ValueError(f"max_frames must be positive or None, got {max_frames}")
        self.target_fps = target_fps
        self.max_frames = max_frames

    def iter_frames(
        self, video_path: str
    ) -> Iterator[Dict[str, Any]]:
        """
        Yield frames one-by-one at target FPS from video.

        Yields:
            Dict:
                - "image": PIL.Image (RGB)
                - "timestamp": float (in seconds)
                - "frame_idx": int (original frame index)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        try:
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = DEFAULT_FALLBACK_FPS

            frame_interval = 1.0 / self.target_fps
            next_timestamp = 0.0
            yielded_count = 0
            frame_idx = 0

            while True:
                current_timestamp = frame_idx / fps
                if current_timestamp >= next_timestamp - 1e-6:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    timestamp = round(current_timestamp, 2)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb_frame)

                    yield {
                        "image": pil_img,
                        "timestamp": timestamp,
                        "frame_idx": frame_idx,
                    }

                    yielded_count += 1
                    next_timestamp += frame_interval
                    if self.max_frames and yielded_count >= self.max_frames:
                        break
                else:
                    ret = cap.grab()
                    if not ret:
                        break

                frame_idx += 1
        finally:
            cap.release()

    def extract_frames(
        self, video_path: str
    ) -> List[Dict[str, Any]]:
        """
        Extract frames at target FPS from video.

        Returns:
            List of dicts:
                - "image": PIL.Image (RGB)
                - "timestamp": float (in seconds)
                - "frame_idx": int (original frame index)
        """
        return list(self.iter_frames(video_path))

    @staticmethod
    def extract_frame_at_timestamp(video_path: str, timestamp_sec: float) -> Optional[Image.Image]:
        """Extract a single frame image closest to timestamp_sec."""
        if not os.path.exists(video_path):
            return None
        cap = cv2.VideoCapture(video_path)
        try:
            if not cap.isOpened():
                return None
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = DEFAULT_FALLBACK_FPS
            frame_idx = int(round(timestamp_sec * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                return None
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb)
        finally:
            cap.release()

    @staticmethod
    def clip_video(
        video_path: str,
        start_sec: float,
        end_sec: float,
        output_path: str,
    ) -> str:
        """Cut a subclip from start_sec to end_sec and save to output_path."""
        if not os.path.exists(video_path):
            raise ValueError(f"Cannot open video: {video_path}")

        if not math.isfinite(start_sec) or not math.isfinite(end_sec):
            raise ValueError(f"start_sec and end_sec must be finite numbers, got {start_sec}, {end_sec}")

        start_sec = max(0.0, float(start_sec))
        if start_sec >= end_sec:
            raise ValueError(f"start_sec ({start_sec}) must be strictly less than end_sec ({end_sec})")

        cap = cv2.VideoCapture(video_path)
        try:
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = DEFAULT_FALLBACK_FPS
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            start_frame = max(0, int(round(start_sec * fps)))
            end_frame = max(start_frame, int(round(end_sec * fps)))
            if total_frames > 0:
                end_frame = min(total_frames, end_frame)

            if end_frame <= start_frame:
                raise ValueError(
                    f"Clamped frame range [{start_frame}, {end_frame}) yields zero frames (start beyond video end)"
                )

            abs_output_path = os.path.abspath(output_path)
            output_dir = os.path.dirname(abs_output_path)
            os.makedirs(output_dir, exist_ok=True)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            frames_written = 0
            try:
                if not out.isOpened():
                    raise RuntimeError(f"Failed to open VideoWriter for {output_path}")

                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                current = start_frame
                while current < end_frame:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    out.write(frame)
                    frames_written += 1
                    current += 1
            finally:
                out.release()

            if frames_written == 0:
                if os.path.exists(output_path):
                    os.remove(output_path)
                raise ValueError(
                    f"Clamped frame range [{start_frame}, {end_frame}) yields zero frames (start beyond video end)"
                )
        except Exception:
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except OSError:
                    pass
            raise
        finally:
            cap.release()

        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin:
            temp_output = f"{output_path}.tmp_{os.getpid()}.mp4"
            try:
                cmd = [
                    ffmpeg_bin,
                    "-y",
                    "-i",
                    output_path,
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    temp_output,
                ]
                proc = subprocess.run(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                if proc.returncode == 0 and os.path.exists(temp_output):
                    os.replace(temp_output, output_path)
                else:
                    if os.path.exists(temp_output):
                        os.remove(temp_output)
            except Exception:
                if os.path.exists(temp_output):
                    os.remove(temp_output)

        return output_path
