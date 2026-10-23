# allow: SIZE_OK — comprehensive benchmark suite covering retrieval, zero-shot classification, and FPS ablation
import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.siglip_engine import SigLIPEngine
from src.synthetic import generate_benchmark_video
from src.video_processor import VideoProcessor
from src.video_search import VideoSearchEngine


def calc_interval_distance(timestamp: float, start_sec: float, end_sec: float) -> float:
    """Compute distance in seconds from timestamp to half-open interval [start_sec, end_sec)."""
    if timestamp < start_sec:
        return start_sec - timestamp
    if timestamp < end_sec:
        return 0.0
    return timestamp - end_sec


def _atomic_save_json(data: Dict[str, Any], out_path: str) -> None:
    """Save JSON data atomically via a temporary file and os.replace."""
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    tmp_path = f"{out_path}.tmp.{os.getpid()}"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, out_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _index_video_honest(
    engine: SigLIPEngine,
    processor: VideoProcessor,
    video_path: str,
    batch_size: int = 16,
) -> Tuple[VideoSearchEngine, int, float, float]:
    """Extract frames once and encode with separate, honest timings."""
    t0 = time.perf_counter()
    extracted_frames = processor.extract_frames(video_path)
    extract_time = time.perf_counter() - t0

    images = [f["image"] for f in extracted_frames]
    t1 = time.perf_counter()
    chunks: List[torch.Tensor] = []
    for i in range(0, len(images), batch_size):
        chunk = images[i : i + batch_size]
        chunks.append(engine.encode_images(chunk))
    if chunks:
        embeds = torch.cat(chunks, dim=0)
    else:
        hidden_dim = getattr(engine.model.config.vision_config, "hidden_size", 768)
        embeds = torch.empty((0, hidden_dim))
    encode_time = time.perf_counter() - t1

    searcher = VideoSearchEngine(engine=engine, processor=processor)
    searcher.frame_embeddings = embeds
    searcher.frame_metadata = [
        {
            "timestamp": f["timestamp"],
            "frame_idx": f["frame_idx"],
            "video_path": video_path,
        }
        for f in extracted_frames
    ]
    return searcher, len(extracted_frames), extract_time, encode_time


def _evaluate_retrieval(
    searcher: VideoSearchEngine,
    gt_scenes: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[float]]:
    """Evaluate text-to-moment retrieval on half-open scene intervals."""
    retrieval_records: List[Dict[str, Any]] = []
    top1_hits = 0
    top3_hits = 0
    top5_hits = 0
    reciprocal_ranks: List[float] = []
    interval_mae_errors: List[float] = []
    center_mae_errors: List[float] = []
    hit_interval_0s = 0
    hit_interval_1s = 0
    hit_interval_2s = 0
    hit_tol_1s = 0
    hit_tol_2s = 0
    total_queries = 0
    latencies: List[float] = []

    for sc in gt_scenes:
        s_start, s_end, center = sc["start_sec"], sc["end_sec"], sc["center_sec"]
        for q in sc["queries"]:
            total_queries += 1
            t_q0 = time.perf_counter()
            # min_gap_sec=0.0: ranking metrics (R@K, MRR) evaluate raw frame ordering without NMS
            results = searcher.search_moments(q, top_k=5, min_gap_sec=0.0)
            latencies.append((time.perf_counter() - t_q0) * 1000)

            hit_rank = 0
            best_ts = results[0].timestamp if results else 0.0

            # Half-open scene membership: s_start <= timestamp < s_end
            for rank_idx, res in enumerate(results, 1):
                if s_start <= res.timestamp < s_end:
                    if hit_rank == 0:
                        hit_rank = rank_idx

            if hit_rank == 1:
                top1_hits += 1
            if 1 <= hit_rank <= 3:
                top3_hits += 1
            if 1 <= hit_rank <= 5:
                top5_hits += 1
            reciprocal_ranks.append(1.0 / hit_rank if hit_rank > 0 else 0.0)

            err_interval = calc_interval_distance(best_ts, s_start, s_end)
            err_center = abs(best_ts - center)
            interval_mae_errors.append(err_interval)
            center_mae_errors.append(err_center)

            if err_interval == 0.0:
                hit_interval_0s += 1
            if err_interval <= 1.0:
                hit_interval_1s += 1
            if err_interval <= 2.0:
                hit_interval_2s += 1
            if err_center <= 1.0:
                hit_tol_1s += 1
            if err_center <= 2.0:
                hit_tol_2s += 1

            retrieval_records.append({
                "scene": sc["category"],
                "query": q,
                "hit_rank": hit_rank,
                "top_timestamp": best_ts,
                "target_start": s_start,
                "target_end": s_end,
                "target_center": center,
                "time_error": round(err_interval, 2),
                "time_error_interval": round(err_interval, 2),
                "time_error_center": round(err_center, 2),
                "top_score": results[0].score if results else 0.0,
            })

    r1 = (top1_hits / total_queries) * 100 if total_queries else 0.0
    r3 = (top3_hits / total_queries) * 100 if total_queries else 0.0
    r5 = (top5_hits / total_queries) * 100 if total_queries else 0.0
    mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
    mean_mae_interval = float(np.mean(interval_mae_errors)) if interval_mae_errors else 0.0
    mean_mae_center = float(np.mean(center_mae_errors)) if center_mae_errors else 0.0
    hit_rate_interval_0s = (hit_interval_0s / total_queries) * 100 if total_queries else 0.0
    hit_rate_interval_1s = (hit_interval_1s / total_queries) * 100 if total_queries else 0.0
    hit_rate_interval_2s = (hit_interval_2s / total_queries) * 100 if total_queries else 0.0
    hit_rate_1s = (hit_tol_1s / total_queries) * 100 if total_queries else 0.0
    hit_rate_2s = (hit_tol_2s / total_queries) * 100 if total_queries else 0.0

    retrieval_dict = {
        "total_queries": total_queries,
        "r1": round(r1, 2),
        "r3": round(r3, 2),
        "r5": round(r5, 2),
        "mrr": round(mrr, 4),
        "mae_sec": round(mean_mae_interval, 2),
        "mae_to_interval_sec": round(mean_mae_interval, 2),
        "mae_to_center_sec": round(mean_mae_center, 2),
        "hit_rate_interval_0s": round(hit_rate_interval_0s, 2),
        "hit_rate_interval_1s": round(hit_rate_interval_1s, 2),
        "hit_rate_interval_2s": round(hit_rate_interval_2s, 2),
        "hit_rate_1s": round(hit_rate_interval_1s, 2),
        "hit_rate_2s": round(hit_rate_interval_2s, 2),
        "hit_rate_center_1s": round(hit_rate_1s, 2),
        "hit_rate_center_2s": round(hit_rate_2s, 2),
        "sample_queries": retrieval_records[:8],
    }
    return retrieval_dict, latencies


def _evaluate_classification(
    engine: SigLIPEngine,
    searcher: VideoSearchEngine,
    gt_scenes: List[Dict[str, Any]],
    distractors: List[str],
) -> Tuple[Dict[str, Any], float]:
    """Evaluate zero-shot segment and video classification using calibrated probabilities."""
    all_candidate_labels = [sc["category"] for sc in gt_scenes] + distractors
    scene_count = len(gt_scenes)

    t_text_0 = time.perf_counter()
    text_embeds = engine.encode_texts(all_candidate_labels)
    text_encode_time = time.perf_counter() - t_text_0
    ms_per_text_query = (text_encode_time / len(all_candidate_labels)) * 1000 if all_candidate_labels else 0.0

    scene_cls_correct_top1 = 0
    scene_cls_correct_top3 = 0
    scene_details: List[Dict[str, Any]] = []
    calibrated_true_probs: List[float] = []
    calibrated_false_probs: List[float] = []

    for sc in gt_scenes:
        true_label = sc["category"]
        # Half-open scene membership: start_sec <= timestamp < end_sec
        scene_frame_indices = [
            i for i, meta in enumerate(searcher.frame_metadata)
            if sc["start_sec"] <= meta["timestamp"] < sc["end_sec"]
        ]
        if not scene_frame_indices:
            continue
        assert searcher.frame_embeddings is not None
        sub_embeds = searcher.frame_embeddings[scene_frame_indices]
        sim_matrix, prob_matrix = engine.score_logits(sub_embeds, text_embeds)
        sim_scores = sim_matrix.mean(dim=0)
        prob_scores = prob_matrix.mean(dim=0)

        ranked_indices = torch.argsort(prob_scores, descending=True).tolist()
        pred_top1 = all_candidate_labels[ranked_indices[0]]
        pred_top3 = [all_candidate_labels[idx] for idx in ranked_indices[:3]]

        top1_match = (pred_top1 == true_label)
        top3_match = (true_label in pred_top3)
        if top1_match:
            scene_cls_correct_top1 += 1
        if top3_match:
            scene_cls_correct_top3 += 1

        true_label_idx = all_candidate_labels.index(true_label)
        true_score = float(sim_scores[true_label_idx].item())
        top_false_score = float(max([sim_scores[idx].item() for idx in ranked_indices if idx != true_label_idx]))
        true_prob = float(prob_scores[true_label_idx].item())
        top_false_prob = float(max([prob_scores[idx].item() for idx in ranked_indices if idx != true_label_idx]))

        calibrated_true_probs.append(true_prob)
        calibrated_false_probs.append(top_false_prob)

        scene_details.append({
            "category": true_label,
            "pred_top1": pred_top1,
            "top1_match": top1_match,
            "similarity_score": round(true_score, 4),
            "sigmoid_prob": round(true_prob, 4),
            "mean_prob": round(true_prob, 4),
            "margin": round(true_score - top_false_score, 4),
            "prob_margin": round(true_prob - top_false_prob, 4),
        })

    cls_acc_top1 = (scene_cls_correct_top1 / scene_count) * 100 if scene_count else 0.0
    cls_acc_top3 = (scene_cls_correct_top3 / scene_count) * 100 if scene_count else 0.0
    avg_true_prob = float(np.mean(calibrated_true_probs)) if calibrated_true_probs else 0.0
    avg_false_prob = float(np.mean(calibrated_false_probs)) if calibrated_false_probs else 0.0

    global_rankings = searcher.classify_video_zero_shot(all_candidate_labels)
    true_labels_set = set(sc["category"] for sc in gt_scenes)
    distractor_scores = [r["mean_score"] for r in global_rankings if r["label"] in distractors]
    true_scores = [r["mean_score"] for r in global_rankings if r["label"] in true_labels_set]
    distractor_probs = [r["mean_prob"] for r in global_rankings if r["label"] in distractors]
    true_probs_global = [r["mean_prob"] for r in global_rankings if r["label"] in true_labels_set]

    avg_true_score = float(np.mean(true_scores)) if true_scores else 0.0
    avg_distractor_score = float(np.mean(distractor_scores)) if distractor_scores else 0.0
    avg_true_prob_global = float(np.mean(true_probs_global)) if true_probs_global else 0.0
    avg_distractor_prob_global = float(np.mean(distractor_probs)) if distractor_probs else 0.0

    classification_dict = {
        "num_classes": len(all_candidate_labels),
        "scene_count": scene_count,
        "top1_acc": round(cls_acc_top1, 2),
        "top3_acc": round(cls_acc_top3, 2),
        "avg_true_score": round(avg_true_score, 4),
        "avg_distractor_score": round(avg_distractor_score, 4),
        "separation_margin": round(avg_true_score - avg_distractor_score, 4),
        "avg_true_sigmoid_prob": round(avg_true_prob, 4),
        "avg_false_sigmoid_prob": round(avg_false_prob, 4),
        "avg_true_prob_global": round(avg_true_prob_global, 4),
        "avg_distractor_prob_global": round(avg_distractor_prob_global, 4),
        "separation_margin_prob": round(avg_true_prob_global - avg_distractor_prob_global, 4),
        "scene_details": scene_details,
    }
    return classification_dict, ms_per_text_query


def _evaluate_ablation(
    engine: SigLIPEngine,
    video_path: str,
    gt_scenes: List[Dict[str, Any]],
    fps_list: List[float],
) -> List[Dict[str, Any]]:
    """Evaluate retrieval sensitivity across frame sampling rates with honest timings."""
    ablation_results: List[Dict[str, Any]] = []
    total_queries = sum(len(sc["queries"]) for sc in gt_scenes)

    for test_fps in fps_list:
        ab_proc = VideoProcessor(target_fps=test_fps)
        ab_searcher, num_ab_frames, ab_ext_time, ab_enc_time = _index_video_honest(engine, ab_proc, video_path)
        ab_indexing_time = ab_ext_time + ab_enc_time

        ab_top1 = 0
        ab_interval_maes: List[float] = []
        ab_center_maes: List[float] = []
        for sc in gt_scenes:
            for q in sc["queries"]:
                res = ab_searcher.search_moments(q, top_k=1, min_gap_sec=0.0)
                if res:
                    top_ts = res[0].timestamp
                    # Half-open scene membership: start_sec <= timestamp < end_sec
                    if sc["start_sec"] <= top_ts < sc["end_sec"]:
                        ab_top1 += 1
                    ab_interval_maes.append(calc_interval_distance(top_ts, sc["start_sec"], sc["end_sec"]))
                    ab_center_maes.append(abs(top_ts - sc["center_sec"]))

        ab_r1 = (ab_top1 / total_queries) * 100 if total_queries else 0.0
        ab_mae_interval = float(np.mean(ab_interval_maes)) if ab_interval_maes else 0.0
        ab_mae_center = float(np.mean(ab_center_maes)) if ab_center_maes else 0.0

        ablation_results.append({
            "fps": test_fps,
            "frames_indexed": num_ab_frames,
            "extraction_time_sec": round(ab_ext_time, 2),
            "encode_time_sec": round(ab_enc_time, 2),
            "indexing_time_sec": round(ab_indexing_time, 2),
            "throughput_fps": round(num_ab_frames / ab_indexing_time, 2) if ab_indexing_time > 0 else 0.0,
            "encode_fps": round(num_ab_frames / ab_enc_time, 2) if ab_enc_time > 0 else 0.0,
            "recall_at_1": round(ab_r1, 2),
            "mae_sec": round(ab_mae_interval, 2),
            "mae_to_interval_sec": round(ab_mae_interval, 2),
            "mae_to_center_sec": round(ab_mae_center, 2),
        })
    return ablation_results


def evaluate_model(
    model_name: str = "google/siglip-base-patch16-224",
    video_path: str = "sample_data/eval_benchmark.mp4",
    fps: float = 1.0,
    out_path: str = "eval_results.json",
    with_text_overlay: bool = False,
    fps_list: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Execute complete benchmark evaluation and save JSON atomically."""
    dataset_tag = "ocr_control_with_text_overlay" if with_text_overlay else "clean_no_overlay"
    print("=" * 65)
    print(f"SigLIP Full Model Evaluation: {model_name}")
    print(f"Dataset mode: {dataset_tag} (text_overlay={with_text_overlay})")
    print("=" * 65)

    # Fast model load validation before generating video or writing files
    engine = SigLIPEngine(model_name=model_name)
    num_params = sum(p.numel() for p in engine.model.parameters())
    logit_scale = float(engine.model.logit_scale.exp().item())
    logit_bias = float(engine.model.logit_bias.item())

    gt_scenes = generate_benchmark_video(
        video_path=video_path,
        duration_per_scene=3.0,
        fps=20,
        with_text_overlay=with_text_overlay,
    )
    total_duration = gt_scenes[-1]["end_sec"]

    processor = VideoProcessor(target_fps=fps)
    searcher, num_frames, extract_time, encode_time = _index_video_honest(engine, processor, video_path)
    end_to_end_time = extract_time + encode_time

    extraction_fps = num_frames / extract_time if extract_time > 0 else 0.0
    encode_fps = num_frames / encode_time if encode_time > 0 else 0.0
    end_to_end_fps = num_frames / end_to_end_time if end_to_end_time > 0 else 0.0

    ms_per_frame_encode = (encode_time / num_frames * 1000) if num_frames > 0 else 0.0
    ms_per_frame_extract = (extract_time / num_frames * 1000) if num_frames > 0 else 0.0
    ms_per_frame_e2e = (end_to_end_time / num_frames * 1000) if num_frames > 0 else 0.0

    retrieval_data, latencies = _evaluate_retrieval(searcher, gt_scenes)
    avg_query_latency_ms = float(np.mean(latencies)) if latencies else 0.0

    distractors = [
        "busy airport runway with airplanes",
        "supermarket grocery store aisles",
        "underground subway train station",
        "basketball court sporting arena",
    ]
    classification_data, ms_per_text_query = _evaluate_classification(engine, searcher, gt_scenes, distractors)

    effective_fps_list = fps_list if fps_list is not None else [0.5, 1.0, 2.0]
    ablation_data = _evaluate_ablation(engine, video_path, gt_scenes, effective_fps_list)

    eval_data = {
        "model_name": model_name,
        "dataset": dataset_tag,
        "text_overlay": with_text_overlay,
        "num_params": num_params,
        "device": str(engine.device),
        "dtype": str(engine.dtype),
        "logit_scale": round(logit_scale, 2),
        "logit_bias": round(logit_bias, 2),
        "video_duration_sec": total_duration,
        "fps": fps,
        "num_frames": num_frames,
        "extraction_fps": round(extraction_fps, 2),
        "encode_fps": round(encode_fps, 2),
        "end_to_end_index_fps": round(end_to_end_fps, 2),
        "extraction_throughput_fps": round(end_to_end_fps, 2),
        "extraction_time_sec": round(extract_time, 4),
        "encode_time_sec": round(encode_time, 4),
        "index_time_sec": round(end_to_end_time, 4),
        "ms_per_frame": round(ms_per_frame_encode, 2),
        "ms_per_frame_encode": round(ms_per_frame_encode, 2),
        "ms_per_frame_extract": round(ms_per_frame_extract, 2),
        "ms_per_frame_end_to_end": round(ms_per_frame_e2e, 2),
        "ms_per_text_query": round(ms_per_text_query, 2),
        "avg_query_latency_ms": round(avg_query_latency_ms, 2),
        "retrieval": retrieval_data,
        "classification": classification_data,
        "ablation_fps": ablation_data,
    }

    _atomic_save_json(eval_data, out_path)
    print(f"\nSummary results saved atomically to: {out_path}")
    return eval_data


def parse_fps_list(value: Optional[str]) -> Optional[List[float]]:
    """Parse comma-separated FPS list string."""
    if not value or not value.strip():
        return None
    try:
        return [float(x.strip()) for x in value.split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid fps list format: {value}") from exc


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint for benchmark evaluation."""
    parser = argparse.ArgumentParser(
        description="Run benchmark evaluation for SigLIP video retrieval and zero-shot classification."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/siglip-base-patch16-224",
        help="HuggingFace model ID or local path.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="eval_results.json",
        help="Output path for evaluation JSON results.",
    )
    parser.add_argument(
        "--video",
        type=str,
        default="sample_data/eval_benchmark.mp4",
        help="Path where benchmark video will be generated.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Frame extraction rate for primary benchmark.",
    )
    parser.add_argument(
        "--with-text-overlay",
        action="store_true",
        default=False,
        help="Enable OCR label text overlay on video frames (leakage baseline control).",
    )
    parser.add_argument(
        "--fps-list",
        type=str,
        default="0.5,1.0,2.0",
        help="Comma-separated target FPS list for ablation study (e.g. '0.5,1.0,2.0' or '' to skip).",
    )

    args = parser.parse_args(argv)

    try:
        fps_list = parse_fps_list(args.fps_list)
        evaluate_model(
            model_name=args.model,
            video_path=args.video,
            fps=args.fps,
            out_path=args.out,
            with_text_overlay=args.with_text_overlay,
            fps_list=fps_list,
        )
        return 0
    except Exception as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
