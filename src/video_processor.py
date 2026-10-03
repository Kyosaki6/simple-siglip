import os
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image


class VideoProcessor:
    """Extract and sample frames from video files with timestamp tracking."""

    def __init__(self, target_fps: float = 1.0, max_frames: Optional[int] = None):
        """
        Args:
            target_fps: Number of frames to extract per second of video.
            max_frames: Maximum total frames to extract (optional cap).
        """
        self.target_fps = target_fps
        self.max_frames = max_frames

    def extract_frames(
        self, video_path: str
    ) -> List[Dict[str, Union[Image.Image, float, int]]]:
        """
        Extract frames at target FPS from video.

        Returns:
            List of dicts:
                - "image": PIL.Image (RGB)
                - "timestamp": float (in seconds)
                - "frame_idx": int (original frame index)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0:
            fps = 25.0  # Fallback standard fps

        step = max(1, int(round(fps / self.target_fps)))

        extracted = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step == 0:
                timestamp = frame_idx / fps
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)

                extracted.append(
                    {
                        "image": pil_img,
                        "timestamp": round(timestamp, 2),
                        "frame_idx": frame_idx,
                    }
                )

                if self.max_frames and len(extracted) >= self.max_frames:
                    break

            frame_idx += 1

        cap.release()
        return extracted

    @staticmethod
    def extract_frame_at_timestamp(video_path: str, timestamp_sec: float) -> Optional[Image.Image]:
        """Extract a single frame image closest to timestamp_sec."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_idx = int(round(timestamp_sec * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    @staticmethod
    def clip_video(
        video_path: str,
        start_sec: float,
        end_sec: float,
        output_path: str,
    ) -> str:
        """Cut a subclip from start_sec to end_sec and save to output_path."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        start_frame = max(0, int(start_sec * fps))
        end_frame = min(total_frames, int(end_sec * fps))

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current = start_frame
        while current <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            current += 1

        cap.release()
        out.release()
        return output_path
