import argparse
import os
import sys
import numpy as np
import cv2

from src.siglip_engine import SigLIPEngine
from src.video_processor import VideoProcessor
from src.video_search import VideoSearchEngine


def generate_dummy_video(output_path: str = "sample_data/demo.mp4", duration_sec: int = 6):
    """Generate a synthetic video with different colored scenes and text."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fps = 20
    total_frames = duration_sec * fps
    width, height = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    scenes = [
        ("red scene with an apple", (0, 0, 200)),
        ("green field with trees", (0, 200, 0)),
        ("blue ocean with water", (200, 100, 0)),
    ]

    frames_per_scene = total_frames // len(scenes)

    for i in range(total_frames):
        scene_idx = min(i // frames_per_scene, len(scenes) - 1)
        desc, color = scenes[scene_idx]

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

    out.release()
    print(f"Generated synthetic demo video at: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="SigLIP Video Search & Zero-shot Classifier")
    parser.add_argument("--video", type=str, default=None, help="Path to video file")
    parser.add_argument("--query", type=str, default="blue ocean with water", help="Search query")
    parser.add_argument(
        "--labels",
        type=str,
        default="red scene with an apple,green field with trees,blue ocean with water,car on road",
        help="Comma-separated labels for zero-shot classification",
    )
    parser.add_argument("--fps", type=float, default=1.0, help="Frames sampled per second")
    parser.add_argument("--model", type=str, default="google/siglip-base-patch16-224", help="SigLIP HF model")
    parser.add_argument("--generate-dummy", action="store_true", help="Generate synthetic video to test")

    args = parser.parse_args()

    video_path = args.video
    if args.generate_dummy or not video_path:
        video_path = generate_dummy_video()

    print(f"Initializing SigLIP model: {args.model}...")
    engine = SigLIPEngine(model_name=args.model)
    processor = VideoProcessor(target_fps=args.fps)
    searcher = VideoSearchEngine(engine=engine, processor=processor)

    print(f"Indexing video: {video_path}...")
    num_frames = searcher.index_video(video_path)
    print(f"Indexed {num_frames} frames from video.")

    # 1. Zero-shot classification
    if args.labels:
        labels = [l.strip() for l in args.labels.split(",") if l.strip()]
        print("\n--- Zero-Shot Video Classification ---")
        rankings = searcher.classify_video_zero_shot(labels)
        for r in rankings:
            print(f"  [{r['mean_score']:.4f}] {r['label']} (peak: {r['max_score']:.4f})")

    # 2. Moment search
    if args.query:
        print(f"\n--- Text-to-Moment Search for: '{args.query}' ---")
        results = searcher.search_moments(args.query, top_k=3)
        for rank, res in enumerate(results, 1):
            print(f"  #{rank}: Timestamp {res.timestamp:.2f}s | Score: {res.score:.4f} | Frame: {res.frame_idx}")


if __name__ == "__main__":
    main()
