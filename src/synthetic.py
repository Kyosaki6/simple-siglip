import os
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

DUMMY_SCENES: List[Tuple[str, Tuple[int, int, int]]] = [
    ("red scene with an apple", (0, 0, 200)),
    ("green field with trees", (0, 200, 0)),
    ("blue ocean with water", (200, 100, 0)),
]

BENCHMARK_SCENES: List[Dict[str, Any]] = [
    {
        "category": "red apple in fruit basket",
        "queries": ["a bright red apple", "fresh red fruit", "crimson red object"],
        "base_color": (30, 30, 210),  # BGR
        "pattern": "circle",
        "pattern_color": (0, 0, 160),
    },
    {
        "category": "lush green forest trees",
        "queries": ["green forest trees", "vibrant green foliage", "nature woodland greenery"],
        "base_color": (30, 180, 40),
        "pattern": "rectangles",
        "pattern_color": (20, 120, 25),
    },
    {
        "category": "deep blue ocean water",
        "queries": ["blue ocean water", "deep sea waves", "clear blue aquatic surface"],
        "base_color": (210, 110, 20),
        "pattern": "waves",
        "pattern_color": (240, 160, 50),
    },
    {
        "category": "golden yellow desert sand dunes",
        "queries": ["golden yellow sand dunes", "bright desert sand", "warm yellow arid landscape"],
        "base_color": (50, 200, 230),
        "pattern": "lines",
        "pattern_color": (20, 160, 190),
    },
    {
        "category": "dark night city neon lights",
        "queries": ["dark night city with neon lights", "illuminated night city skyline", "dark night darkness with glow"],
        "base_color": (25, 20, 30),
        "pattern": "dots",
        "pattern_color": (220, 100, 250),
    },
    {
        "category": "white snowy winter mountain",
        "queries": ["white snowy winter mountain", "pure white snow landscape", "frosty blizzard winter"],
        "base_color": (235, 235, 240),
        "pattern": "peaks",
        "pattern_color": (190, 190, 200),
    },
    {
        "category": "fiery orange sunset skyline",
        "queries": ["fiery orange sunset sky", "evening sunset glow", "warm orange horizon dusk"],
        "base_color": (30, 100, 235),
        "pattern": "gradient",
        "pattern_color": (20, 50, 180),
    },
    {
        "category": "modern gray highway asphalt road",
        "queries": ["gray asphalt highway road", "concrete road lane", "industrial gray roadway"],
        "base_color": (110, 110, 110),
        "pattern": "stripes",
        "pattern_color": (230, 230, 230),
    },
]


def generate_dummy_video(
    output_path: str = "sample_data/demo.mp4",
    duration_sec: int = 6,
) -> str:
    """Generate a synthetic video with different colored scenes and text."""
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fps = 20
    total_frames = duration_sec * fps
    width, height = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")

    try:
        frames_per_scene = max(1, total_frames // len(DUMMY_SCENES))

        for i in range(total_frames):
            scene_idx = min(i // frames_per_scene, len(DUMMY_SCENES) - 1)
            desc, color = DUMMY_SCENES[scene_idx]

            frame = np.full((height, width, 3), color, dtype=np.uint8)
            cv2.putText(
                frame,
                desc,
                (10, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            out.write(frame)
    finally:
        out.release()

    print(f"Generated synthetic demo video at: {output_path}")
    return output_path


def generate_benchmark_video(
    video_path: str,
    duration_per_scene: float = 3.0,
    fps: int = 20,
    with_text_overlay: bool = False,
) -> List[Dict[str, Any]]:
    """
    Generate synthetic multi-scene benchmark video with diverse visual scenes,
    textures, colors, patterns, and geometric features.
    Returns ground-truth segment annotations.
    """
    output_dir = os.path.dirname(os.path.abspath(video_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    width, height = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    if not out.isOpened():
        raise RuntimeError(f"Failed to open video writer for {video_path}")

    frames_per_scene = int(duration_per_scene * fps)
    ground_truth: List[Dict[str, Any]] = []

    try:
        for idx, sc in enumerate(BENCHMARK_SCENES):
            start_sec = idx * duration_per_scene
            end_sec = start_sec + duration_per_scene
            center_sec = (start_sec + end_sec) / 2.0

            ground_truth.append({
                "scene_idx": idx,
                "category": sc["category"],
                "queries": sc["queries"],
                "start_sec": round(start_sec, 2),
                "end_sec": round(end_sec, 2),
                "center_sec": round(center_sec, 2),
            })

            for _ in range(frames_per_scene):
                frame = np.full((height, width, 3), sc["base_color"], dtype=np.uint8)

                # Draw visual patterns
                pattern = sc["pattern"]
                if pattern == "circle":
                    cv2.circle(frame, (width // 2, height // 2), 65, sc["pattern_color"], -1)
                    cv2.circle(frame, (width // 2, height // 2), 20, (20, 20, 120), -1)
                elif pattern == "rectangles":
                    cv2.rectangle(frame, (40, 60), (120, 200), sc["pattern_color"], -1)
                    cv2.rectangle(frame, (180, 40), (280, 220), sc["pattern_color"], -1)
                elif pattern == "waves":
                    for w in range(0, width, 40):
                        cv2.ellipse(frame, (w + 20, height // 2), (25, 15), 0, 0, 180, sc["pattern_color"], 3)
                elif pattern == "lines":
                    for y in range(40, height, 40):
                        cv2.line(frame, (0, y), (width, y + 15), sc["pattern_color"], 4)
                elif pattern == "dots":
                    for cx, cy in [(60, 60), (200, 80), (120, 160), (260, 180)]:
                        cv2.circle(frame, (cx, cy), 15, sc["pattern_color"], -1)
                elif pattern == "peaks":
                    pts = np.array([[30, 200], [100, 70], [170, 200], [240, 90], [300, 200]], np.int32)
                    cv2.polylines(frame, [pts], False, sc["pattern_color"], 5)
                elif pattern == "gradient":
                    for y in range(height):
                        alpha = y / float(height)
                        frame[y, :] = (np.array(sc["base_color"]) * (1 - alpha) + np.array(sc["pattern_color"]) * alpha).astype(np.uint8)
                elif pattern == "stripes":
                    cv2.line(frame, (width // 2, 0), (width // 2, height), sc["pattern_color"], 8)

                if with_text_overlay:
                    cv2.putText(
                        frame,
                        sc["category"],
                        (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                out.write(frame)
    finally:
        out.release()

    return ground_truth
