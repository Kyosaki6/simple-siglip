import os
import sys
import time
import json
import cv2
import numpy as np
import torch
from typing import List, Dict, Any

from src.siglip_engine import SigLIPEngine
from src.video_processor import VideoProcessor
from src.video_search import VideoSearchEngine


def generate_benchmark_video(video_path: str, duration_per_scene: float = 3.0, fps: int = 20) -> List[Dict[str, Any]]:
    """
    Generate synthetic multi-scene benchmark video with diverse visual scenes,
    textures, colors, patterns, and geometric features.
    Returns ground-truth segment annotations.
    """
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    width, height = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    scenes = [
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

    frames_per_scene = int(duration_per_scene * fps)
    ground_truth = []

    for idx, sc in enumerate(scenes):
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

        for f in range(frames_per_scene):
            frame = np.full((height, width, 3), sc["base_color"], dtype=np.uint8)

            # Draw visual patterns
            if sc["pattern"] == "circle":
                cv2.circle(frame, (width // 2, height // 2), 65, sc["pattern_color"], -1)
                cv2.circle(frame, (width // 2, height // 2), 20, (20, 20, 120), -1)
            elif sc["pattern"] == "rectangles":
                cv2.rectangle(frame, (40, 60), (120, 200), sc["pattern_color"], -1)
                cv2.rectangle(frame, (180, 40), (280, 220), sc["pattern_color"], -1)
            elif sc["pattern"] == "waves":
                for w in range(0, width, 40):
                    cv2.ellipse(frame, (w + 20, height // 2), (25, 15), 0, 0, 180, sc["pattern_color"], 3)
            elif sc["pattern"] == "lines":
                for y in range(40, height, 40):
                    cv2.line(frame, (0, y), (width, y + 15), sc["pattern_color"], 4)
            elif sc["pattern"] == "dots":
                for cx, cy in [(60, 60), (200, 80), (120, 160), (260, 180)]:
                    cv2.circle(frame, (cx, cy), 15, sc["pattern_color"], -1)
            elif sc["pattern"] == "peaks":
                pts = np.array([[30, 200], [100, 70], [170, 200], [240, 90], [300, 200]], np.int32)
                cv2.polylines(frame, [pts], False, sc["pattern_color"], 5)
            elif sc["pattern"] == "gradient":
                for y in range(height):
                    alpha = y / float(height)
                    frame[y, :] = (np.array(sc["base_color"]) * (1 - alpha) + np.array(sc["pattern_color"]) * alpha).astype(np.uint8)
            elif sc["pattern"] == "stripes":
                cv2.line(frame, (width // 2, 0), (width // 2, height), sc["pattern_color"], 8)

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

    out.release()
    return ground_truth


def evaluate_model(
    model_name: str = "google/siglip-base-patch16-224",
    video_path: str = "sample_data/eval_benchmark.mp4",
    fps: float = 1.0,
):
    print("=" * 65)
    print(f"SigLIP Full Model Evaluation: {model_name}")
    print("=" * 65)

    gt_scenes = generate_benchmark_video(video_path, duration_per_scene=3.0, fps=20)
    total_duration = gt_scenes[-1]["end_sec"]

    # Model initialization
    t0 = time.perf_counter()
    engine = SigLIPEngine(model_name=model_name)
    init_latency_sec = time.perf_counter() - t0
    num_params = sum(p.numel() for p in engine.model.parameters())

    logit_scale = float(engine.model.logit_scale.exp().item())
    logit_bias = float(engine.model.logit_bias.item())

    processor = VideoProcessor(target_fps=fps)
    searcher = VideoSearchEngine(engine=engine, processor=processor)

    # Frame extraction & vision encoding
    t_start_extract = time.perf_counter()
    extracted_frames = processor.extract_frames(video_path)
    extract_time = time.perf_counter() - t_start_extract

    t_start_encode = time.perf_counter()
    num_indexed = searcher.index_video(video_path)
    encode_time = time.perf_counter() - t_start_encode

    num_frames = len(extracted_frames)
    fps_throughput = num_frames / encode_time if encode_time > 0 else 0
    ms_per_frame = (encode_time / num_frames * 1000) if num_frames > 0 else 0

    # 1. Text-to-Moment Retrieval Evaluation
    retrieval_records = []
    top1_hits = 0
    top3_hits = 0
    top5_hits = 0
    reciprocal_ranks = []
    mae_errors = []
    hit_tol_1s = 0
    hit_tol_2s = 0
    total_queries = 0
    retrieval_latencies = []

    for sc in gt_scenes:
        s_start = sc["start_sec"]
        s_end = sc["end_sec"]
        center = sc["center_sec"]

        for q in sc["queries"]:
            total_queries += 1
            t_q0 = time.perf_counter()
            results = searcher.search_moments(q, top_k=5)
            q_lat = (time.perf_counter() - t_q0) * 1000
            retrieval_latencies.append(q_lat)

            hit_rank = 0
            best_ts = results[0].timestamp if results else 0.0

            for rank_idx, res in enumerate(results, 1):
                if s_start <= res.timestamp <= s_end:
                    if hit_rank == 0:
                        hit_rank = rank_idx

            if hit_rank == 1:
                top1_hits += 1
            if 1 <= hit_rank <= 3:
                top3_hits += 1
            if 1 <= hit_rank <= 5:
                top5_hits += 1

            if hit_rank > 0:
                reciprocal_ranks.append(1.0 / hit_rank)
            else:
                reciprocal_ranks.append(0.0)

            time_err = abs(best_ts - center)
            mae_errors.append(time_err)
            if time_err <= 1.0:
                hit_tol_1s += 1
            if time_err <= 2.0:
                hit_tol_2s += 1

            retrieval_records.append({
                "scene": sc["category"],
                "query": q,
                "hit_rank": hit_rank,
                "top_timestamp": best_ts,
                "target_center": center,
                "time_error": round(time_err, 2),
                "top_score": results[0].score if results else 0.0,
            })

    r1 = (top1_hits / total_queries) * 100
    r3 = (top3_hits / total_queries) * 100
    r5 = (top5_hits / total_queries) * 100
    mrr = float(np.mean(reciprocal_ranks))
    mean_mae = float(np.mean(mae_errors))
    hit_rate_1s = (hit_tol_1s / total_queries) * 100
    hit_rate_2s = (hit_tol_2s / total_queries) * 100
    avg_query_latency_ms = float(np.mean(retrieval_latencies))

    # 2. Zero-Shot Video & Segment Classification Evaluation
    distractors = [
        "busy airport runway with airplanes",
        "supermarket grocery store aisles",
        "underground subway train station",
        "basketball court sporting arena",
    ]
    all_candidate_labels = [sc["category"] for sc in gt_scenes] + distractors
    scene_count = len(gt_scenes)

    # Encode texts
    t_text_0 = time.perf_counter()
    text_embeds = engine.encode_texts(all_candidate_labels)  # (K, D)
    text_encode_time = time.perf_counter() - t_text_0
    ms_per_text_query = (text_encode_time / len(all_candidate_labels)) * 1000

    scene_cls_correct_top1 = 0
    scene_cls_correct_top3 = 0
    scene_details = []
    calibrated_true_probs = []
    calibrated_false_probs = []

    for sc in gt_scenes:
        true_label = sc["category"]
        scene_frame_indices = [
            i for i, meta in enumerate(searcher.frame_metadata)
            if sc["start_sec"] <= meta["timestamp"] <= sc["end_sec"]
        ]
        if not scene_frame_indices:
            continue
        sub_embeds = searcher.frame_embeddings[scene_frame_indices]  # (N_s, D)
        sim_scores = (sub_embeds @ text_embeds.T).mean(dim=0)  # (K,)

        # Native SigLIP Sigmoid Probability: sigma(scale * score + bias)
        siglip_logits = logit_scale * sim_scores + logit_bias
        siglip_probs = torch.sigmoid(siglip_logits)

        ranked_indices = torch.argsort(sim_scores, descending=True).tolist()
        pred_top1 = all_candidate_labels[ranked_indices[0]]
        pred_top3 = [all_candidate_labels[idx] for idx in ranked_indices[:3]]

        top1_match = (pred_top1 == true_label)
        top3_match = (true_label in pred_top3)

        if top1_match:
            scene_cls_correct_top1 += 1
        if top3_match:
            scene_cls_correct_top3 += 1

        true_label_idx = all_candidate_labels.index(true_label)
        true_score = float(sim_scores[true_label_idx])
        top_false_score = float(max([sim_scores[idx] for idx in ranked_indices if idx != true_label_idx]))
        margin = true_score - top_false_score

        true_prob = float(siglip_probs[true_label_idx])
        false_prob = float(max([siglip_probs[idx] for idx in ranked_indices if idx != true_label_idx]))
        calibrated_true_probs.append(true_prob)
        calibrated_false_probs.append(false_prob)

        scene_details.append({
            "category": true_label,
            "pred_top1": pred_top1,
            "top1_match": top1_match,
            "similarity_score": round(true_score, 4),
            "sigmoid_prob": round(true_prob, 4),
            "margin": round(margin, 4),
        })

    cls_acc_top1 = (scene_cls_correct_top1 / scene_count) * 100
    cls_acc_top3 = (scene_cls_correct_top3 / scene_count) * 100
    avg_true_prob = float(np.mean(calibrated_true_probs))
    avg_false_prob = float(np.mean(calibrated_false_probs))

    # Whole-video zero-shot classification ranking
    global_rankings = searcher.classify_video_zero_shot(all_candidate_labels)
    true_labels_set = set(sc["category"] for sc in gt_scenes)
    distractor_scores = [r["mean_score"] for r in global_rankings if r["label"] in distractors]
    true_scores = [r["mean_score"] for r in global_rankings if r["label"] in true_labels_set]
    avg_true_score = float(np.mean(true_scores))
    avg_distractor_score = float(np.mean(distractor_scores))
    distractor_margin = avg_true_score - avg_distractor_score

    # 3. FPS Sampling Rate Sensitivity (Ablation Study)
    ablation_results = []
    for test_fps in [0.5, 1.0, 2.0]:
        t_ab_0 = time.perf_counter()
        ab_proc = VideoProcessor(target_fps=test_fps)
        ab_searcher = VideoSearchEngine(engine=engine, processor=ab_proc)
        num_ab_frames = ab_searcher.index_video(video_path)
        ab_indexing_time = time.perf_counter() - t_ab_0

        ab_top1 = 0
        ab_maes = []
        for sc in gt_scenes:
            for q in sc["queries"]:
                res = ab_searcher.search_moments(q, top_k=1)
                if res:
                    if sc["start_sec"] <= res[0].timestamp <= sc["end_sec"]:
                        ab_top1 += 1
                    ab_maes.append(abs(res[0].timestamp - sc["center_sec"]))

        ab_r1 = (ab_top1 / total_queries) * 100
        ab_mae = float(np.mean(ab_maes)) if ab_maes else 0.0

        ablation_results.append({
            "fps": test_fps,
            "frames_indexed": num_ab_frames,
            "indexing_time_sec": round(ab_indexing_time, 2),
            "throughput_fps": round(num_ab_frames / ab_indexing_time, 2),
            "recall_at_1": round(ab_r1, 2),
            "mae_sec": round(ab_mae, 2),
        })

    eval_data = {
        "model_name": model_name,
        "num_params": num_params,
        "device": str(engine.device),
        "dtype": str(engine.dtype),
        "logit_scale": round(logit_scale, 2),
        "logit_bias": round(logit_bias, 2),
        "video_duration_sec": total_duration,
        "fps": fps,
        "num_frames": num_frames,
        "extraction_throughput_fps": round(fps_throughput, 2),
        "ms_per_frame": round(ms_per_frame, 2),
        "ms_per_text_query": round(ms_per_text_query, 2),
        "avg_query_latency_ms": round(avg_query_latency_ms, 2),
        "retrieval": {
            "total_queries": total_queries,
            "r1": round(r1, 2),
            "r3": round(r3, 2),
            "r5": round(r5, 2),
            "mrr": round(mrr, 4),
            "mae_sec": round(mean_mae, 2),
            "hit_rate_1s": round(hit_rate_1s, 2),
            "hit_rate_2s": round(hit_rate_2s, 2),
            "sample_queries": retrieval_records[:8],
        },
        "classification": {
            "num_classes": len(all_candidate_labels),
            "scene_count": scene_count,
            "top1_acc": round(cls_acc_top1, 2),
            "top3_acc": round(cls_acc_top3, 2),
            "avg_true_score": round(avg_true_score, 4),
            "avg_distractor_score": round(avg_distractor_score, 4),
            "separation_margin": round(distractor_margin, 4),
            "avg_true_sigmoid_prob": round(avg_true_prob, 4),
            "avg_false_sigmoid_prob": round(avg_false_prob, 4),
            "scene_details": scene_details,
        },
        "ablation_fps": ablation_results,
    }

    out_json = "eval_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)

    print("\nSummary Results Saved to eval_results.json")
    return eval_data


if __name__ == "__main__":
    evaluate_model()
