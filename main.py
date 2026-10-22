import argparse
import sys

from src.siglip_engine import SigLIPEngine
from src.synthetic import generate_dummy_video
from src.video_processor import VideoProcessor
from src.video_search import VideoSearchEngine

__all__ = ["generate_dummy_video", "main"]


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
