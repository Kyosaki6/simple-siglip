import argparse
import os
import sys
from typing import Dict, List, Optional

from src.heads import load_head, pool_video_embedding, predict
from src.siglip_engine import SigLIPEngine
from src.synthetic import generate_dummy_video
from src.video_processor import VideoProcessor
from src.video_search import VideoSearchEngine

__all__ = ["generate_dummy_video", "main"]


def _resolve_adapter_classes(
    user_labels: List[str],
    class_names: List[str],
    label_map: Optional[Dict[str, str]] = None,
) -> List[str]:
    mapping = label_map or {}
    matched: List[str] = []
    for raw_label in user_labels:
        cleaned = raw_label.strip()
        if not cleaned:
            continue
        target: Optional[str] = None
        if cleaned in class_names:
            target = cleaned
        elif cleaned.lower() in mapping and mapping[cleaned.lower()] in class_names:
            target = mapping[cleaned.lower()]
        elif cleaned in mapping and mapping[cleaned] in class_names:
            target = mapping[cleaned]
        else:
            norm_label = cleaned.lower().replace(" ", "_")
            for c in class_names:
                if c.lower() == norm_label or c.lower().replace("_", " ") == cleaned.lower():
                    target = c
                    break
        if target is not None and target not in matched:
            matched.append(target)
    return matched if matched else list(class_names)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="SigLIP Video Search & Zero-shot Classifier")
    parser.add_argument("--video", type=str, default=None, help="Path to video file")
    parser.add_argument("--query", type=str, default="blue ocean with water", help="Search query")
    parser.add_argument(
        "--labels",
        type=str,
        default="red scene with an apple,green field with trees,blue ocean with water,car on road",
        help="Comma-separated labels for zero-shot classification",
    )
    parser.add_argument(
        "--adapter-checkpoint",
        type=str,
        default=None,
        help="Path to trained linear adapter checkpoint (.npz)",
    )
    parser.add_argument("--fps", type=float, default=1.0, help="Frames sampled per second")
    parser.add_argument("--model", type=str, default="google/siglip-base-patch16-224", help="SigLIP HF model")
    parser.add_argument("--generate-dummy", action="store_true", help="Generate synthetic video to test")
    parser.add_argument(
        "--dummy-out",
        type=str,
        default=None,
        help="Output path for generated synthetic dummy video (default: sample_data/demo.mp4)",
    )
    parser.add_argument(
        "--min-gap",
        type=float,
        default=2.0,
        help="Minimum temporal gap in seconds between retrieved moments (NMS)",
    )

    args = parser.parse_args(argv)

    if args.dummy_out and not args.generate_dummy:
        sys.stderr.write("Error: --dummy-out requires --generate-dummy.\n")
        return 2

    if args.video is not None:
        if not os.path.exists(args.video):
            sys.stderr.write(f"Error: Video file not found: {args.video}\n")
            return 2
        if args.generate_dummy:
            sys.stderr.write("Error: Cannot specify both --video and --generate-dummy.\n")
            return 2
        video_path = args.video
    elif args.generate_dummy:
        project_dir = os.path.dirname(os.path.abspath(__file__))
        default_dummy_path = os.path.join(project_dir, "sample_data", "demo.mp4")
        video_path = os.path.abspath(args.dummy_out) if args.dummy_out else default_dummy_path
        generate_dummy_video(output_path=video_path)
    else:
        sys.stderr.write("Error: Please provide --video <path> or --generate-dummy.\n")
        parser.print_usage(sys.stderr)
        return 2

    if args.adapter_checkpoint is not None and not os.path.exists(args.adapter_checkpoint):
        sys.stderr.write(f"Error: Adapter checkpoint not found: {args.adapter_checkpoint}\n")
        return 2

    try:
        print(f"Initializing SigLIP model: {args.model}...")
        engine = SigLIPEngine(model_name=args.model)
        processor = VideoProcessor(target_fps=args.fps)
        searcher = VideoSearchEngine(engine=engine, processor=processor)

        print(f"Indexing video: {video_path}...")
        num_frames = searcher.index_video(video_path)
        print(f"Indexed {num_frames} frames from video.")

        if args.adapter_checkpoint:
            head, meta = load_head(args.adapter_checkpoint, expected_model_name=args.model)
            class_names: List[str] = meta["class_names"]
            label_map: Dict[str, str] = meta.get("label_map") or {}

            raw_labels = [l.strip() for l in args.labels.split(",") if l.strip()] if args.labels else []
            eval_classes = _resolve_adapter_classes(raw_labels, class_names, label_map)

            pooled_emb = pool_video_embedding(searcher.frame_embeddings, mode="mean")
            preds, probs = predict(head, pooled_emb.unsqueeze(0))

            print("\n--- Trained Linear Adapter Video Classification ---")
            print(f"Checkpoint: {args.adapter_checkpoint} | Total classes: {len(class_names)}")
            print("Note: Scores are probabilities derived from trained linear adapter head.")

            rankings = []
            for c in eval_classes:
                idx = class_names.index(c)
                p = float(probs[0, idx].item())
                rankings.append({"class": c, "prob": p})
            rankings.sort(key=lambda x: x["prob"], reverse=True)

            max_class_len = max([len("Class / Label")] + [len(r["class"]) for r in rankings])
            header = f"{'Rank':<5} {'Class / Label':<{max_class_len}} {'Adapter Prob':>14}"
            print(header)
            print("-" * len(header))
            for rank, r in enumerate(rankings, 1):
                p_pct = f"{r['prob'] * 100:.2f}%"
                print(f"{rank:<5} {r['class']:<{max_class_len}} {p_pct:>14}")

            top_idx = int(preds[0].item())
            top_class = class_names[top_idx]
            top_prob = float(probs[0, top_idx].item())
            print(f"\nOverall Top Prediction (Checkpoint): {top_class} ({top_prob * 100:.2f}%)")
        elif args.labels:
            labels = [l.strip() for l in args.labels.split(",") if l.strip()]
            if labels:
                rankings = searcher.classify_video_zero_shot(labels)
                print("\n--- Zero-Shot Video Classification ---")
                max_label_len = max([len("Label")] + [len(r["label"]) for r in rankings])
                header = (
                    f"{'Rank':<5} {'Label':<{max_label_len}} "
                    f"{'Mean Cos':>10} {'Max Cos':>10} {'Mean Prob':>11} {'Max Prob':>11}"
                )
                print(header)
                print("-" * len(header))
                for rank, r in enumerate(rankings, 1):
                    mean_p_pct = f"{r['mean_prob'] * 100:.2f}%"
                    max_p_pct = f"{r['max_prob'] * 100:.2f}%"
                    print(
                        f"{rank:<5} {r['label']:<{max_label_len}} "
                        f"{r['mean_score']:>10.4f} {r['max_score']:>10.4f} "
                        f"{mean_p_pct:>11} {max_p_pct:>11}"
                    )

        # 2. Moment search
        if args.query and args.query.strip():
            print(f"\n--- Text-to-Moment Search for: '{args.query.strip()}' ---")
            results = searcher.search_moments(
                args.query.strip(), top_k=3, min_gap_sec=args.min_gap
            )
            for rank, res in enumerate(results, 1):
                print(
                    f"  #{rank}: Timestamp {res.timestamp:.2f}s | "
                    f"Score: {res.score:.4f} | Frame: {res.frame_idx}"
                )

        return 0
    except (ValueError, FileNotFoundError, RuntimeError, KeyError) as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
