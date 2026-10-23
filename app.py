import math
import os
import shutil
import tempfile
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import cv2
import gradio as gr
import pandas as pd
from PIL import Image

from src.siglip_engine import SigLIPEngine
from src.synthetic import generate_dummy_video
from src.video_processor import VideoProcessor
from src.video_search import VideoSearchEngine

DEFAULT_DEMO_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "sample_data", "demo.mp4"
)
MAX_INDEX_CACHE_SIZE: int = 4
MAX_UI_FRAMES: int = 3600
MAX_CLIP_KEEP: int = 8
MAX_CLIP_AGE_SEC: float = 600.0

_engine_cache: Dict[str, SigLIPEngine] = {}
_engine_lock = threading.Lock()
_index_cache: OrderedDict[Tuple[str, int, int, float, str], VideoSearchEngine] = OrderedDict()
_index_lock = threading.Lock()


def get_engine(model_name: str) -> SigLIPEngine:
    with _engine_lock:
        if model_name not in _engine_cache:
            _engine_cache[model_name] = SigLIPEngine(model_name=model_name)
        return _engine_cache[model_name]


def get_index(video_path: str, fps: float, model_name: str) -> VideoSearchEngine:
    abs_path = os.path.abspath(video_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    st = os.stat(abs_path)
    key = (abs_path, st.st_mtime_ns, st.st_size, float(fps), model_name)

    with _index_lock:
        if key in _index_cache:
            searcher = _index_cache[key]
            _index_cache.move_to_end(key)
            return searcher

    engine = get_engine(model_name)
    processor = VideoProcessor(target_fps=float(fps))
    searcher = VideoSearchEngine(engine=engine, processor=processor)
    if searcher.index_video(abs_path) == 0:
        raise ValueError(f"Could not extract any frames from {video_path}")

    with _index_lock:
        _index_cache[key] = searcher
        _index_cache.move_to_end(key)
        while len(_index_cache) > MAX_INDEX_CACHE_SIZE:
            _index_cache.popitem(last=False)

    return searcher


def _prune_clip_directories(max_age_sec: float = MAX_CLIP_AGE_SEC, max_keep: int = MAX_CLIP_KEEP) -> None:
    temp_dir = tempfile.gettempdir()
    uid = getattr(os, "getuid", lambda: None)()
    now = time.time()
    candidates: List[Tuple[float, str]] = []
    try:
        entries = os.listdir(temp_dir)
    except OSError:
        return
    for name in entries:
        if not name.startswith("siglip_clip_"):
            continue
        path = os.path.join(temp_dir, name)
        if not os.path.isdir(path):
            continue
        try:
            st = os.stat(path)
            if uid is not None and st.st_uid != uid:
                continue
            if (now - st.st_mtime) > max_age_sec:
                shutil.rmtree(path, ignore_errors=True)
            else:
                candidates.append((st.st_mtime, path))
        except OSError:
            continue
    candidates.sort(key=lambda x: x[0], reverse=True)
    for _, path in candidates[max_keep:]:
        shutil.rmtree(path, ignore_errors=True)


def _create_clip_dir() -> str:
    _prune_clip_directories()
    return tempfile.mkdtemp(prefix="siglip_clip_")


def _resolve_video_path(video_file: Any) -> str:
    if not video_file:
        if not os.path.exists(DEFAULT_DEMO_PATH):
            generate_dummy_video(DEFAULT_DEMO_PATH)
        return DEFAULT_DEMO_PATH
    return video_file if isinstance(video_file, str) else getattr(video_file, "name", str(video_file))


def _validate_video(video_path: str, fps: float) -> Optional[str]:
    if not os.path.exists(video_path):
        return f"❌ Không tìm thấy file video (Video file not found): {video_path}"
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            return f"❌ Không thể đọc file video (Cannot read video file): {video_path}"
        fps_native = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    finally:
        cap.release()
    if fps_native > 0 and total_frames > 0:
        est = int(math.ceil((total_frames / fps_native) * float(fps)))
        if est > MAX_UI_FRAMES:
            return (
                f"❌ Video quá dài: ước tính {est} frames vượt quá giới hạn giao diện "
                f"({MAX_UI_FRAMES} frames). Vui lòng giảm FPS hoặc chọn video ngắn hơn "
                f"(Video exceeds UI frame limit of {MAX_UI_FRAMES})."
            )
    return None


def process_video_and_search(
    video_file: Any, query_text: str, fps: float, model_name: str, top_k: int, min_gap_sec: float = 2.0
):
    if not query_text or not query_text.strip():
        return "❌ Vui lòng nhập câu truy vấn tìm kiếm (Search query cannot be empty).", [], None, None
    clean_query = query_text.strip()
    video_path = _resolve_video_path(video_file)
    val_err = _validate_video(video_path, fps)
    if val_err:
        return val_err, [], None, None

    try:
        searcher = get_index(video_path, fps=fps, model_name=model_name)
        results = searcher.search_moments(clean_query, top_k=int(top_k), min_gap_sec=float(min_gap_sec))
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        return f"❌ Lỗi xử lý video (Video processing error): {e}", [], None, None

    gallery, summary, top_ts = [], [], (results[0].timestamp if results else None)
    for rank, res in enumerate(results, 1):
        img = VideoProcessor.extract_frame_at_timestamp(video_path, res.timestamp)
        if img is not None:
            gallery.append((img, f"#{rank} | {res.timestamp:.2f}s | Score: {res.score:.4f}"))
        summary.append(
            {"Rank": rank, "Timestamp (s)": res.timestamp, "Frame Index": res.frame_idx, "Similarity Score": res.score}
        )

    clip_path, clip_note = None, ""
    if top_ts is not None:
        try:
            c_dir = _create_clip_dir()
            bname = os.path.splitext(os.path.basename(video_path))[0]
            out = os.path.join(c_dir, f"clip_{bname}_{int(round(top_ts))}.mp4")
            VideoProcessor.clip_video(video_path, max(0.0, top_ts - 1.5), top_ts + 1.5, out)
            clip_path = out
        except ValueError as e:
            clip_note = f" (Lưu ý: Không thể cắt clip: {e})"

    n_frames = len(searcher.frame_metadata)
    msg = f"Indexed {n_frames} frames ({float(fps):.1f} FPS). Found {len(results)} matches for '{clean_query}'.{clip_note}"
    return msg, gallery, pd.DataFrame(summary), clip_path


def classify_video(video_file: Any, labels_str: str, fps: float, model_name: str):
    labels = [l.strip() for l in (labels_str or "").split(",") if l.strip()]
    if not labels:
        return "❌ Vui lòng cung cấp ít nhất một nhãn phân loại hợp lệ (Please provide at least one valid label).", None
    video_path = _resolve_video_path(video_file)
    val_err = _validate_video(video_path, fps)
    if val_err:
        return val_err, None

    try:
        searcher = get_index(video_path, fps=fps, model_name=model_name)
        rankings = searcher.classify_video_zero_shot(labels)
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        return f"❌ Lỗi xử lý video (Video processing error): {e}", None

    df = pd.DataFrame([
        {
            "Label": r["label"],
            "Mean cosine": r["mean_score"],
            "Max cosine": r["max_score"],
            "Mean probability %": round(r["mean_prob"] * 100, 2),
            "Max probability %": round(r["max_prob"] * 100, 2),
        }
        for r in rankings
    ])
    n_frames = len(searcher.frame_metadata)
    return f"Indexed {n_frames} frames ({float(fps):.1f} FPS). Classified against {len(labels)} categories.", df


def build_app() -> gr.Blocks:
    with gr.Blocks(title="SigLIP Video Intelligence Lab") as demo:
        gr.Markdown(
            "# 🎥 SigLIP Video Intelligence Lab\n"
            "**Text-to-Moment Retrieval & Zero-shot Classification powered by SigLIP (Sigmoid Loss for Language Image Pre-training)**"
        )
        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(label="Input Video (Leave empty to use sample demo)")
                model_dropdown = gr.Dropdown(
                    choices=["google/siglip-base-patch16-224", "google/siglip-so400m-patch14-384"],
                    value="google/siglip-base-patch16-224",
                    label="SigLIP Backbone",
                )
                fps_slider = gr.Slider(minimum=0.5, maximum=5.0, value=1.0, step=0.5, label="Sampling Rate (FPS)")
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("🔍 Text-to-Moment Search"):
                        query_input = gr.Textbox(
                            value="blue ocean with water",
                            label="Natural Language Query",
                            placeholder="e.g., 'a red apple', 'goal celebration', 'car speeding'",
                        )
                        top_k_slider = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Top K Moments")
                        min_gap_slider = gr.Slider(
                            minimum=0.0, maximum=10.0, value=2.0, step=0.5, label="Min gap between moments (s)"
                        )
                        search_btn = gr.Button("Search Video Moments", variant="primary")
                        search_status = gr.Markdown()
                        results_gallery = gr.Gallery(label="Matching Frame Moments", columns=3, height="auto")
                        results_df = gr.Dataframe(label="Detailed Rankings")
                        clipped_output = gr.Video(label="Auto-Extracted Top Clip (±1.5s)")
                        search_btn.click(
                            fn=process_video_and_search,
                            inputs=[video_input, query_input, fps_slider, model_dropdown, top_k_slider, min_gap_slider],
                            outputs=[search_status, results_gallery, results_df, clipped_output],
                        )
                    with gr.TabItem("🏷️ Zero-Shot Classification"):
                        labels_input = gr.Textbox(
                            value="red scene with an apple, green field with trees, blue ocean with water, car on road",
                            label="Candidate Labels (Comma-separated)",
                        )
                        classify_btn = gr.Button("Classify Video", variant="primary")
                        classify_status = gr.Markdown()
                        classify_df = gr.Dataframe(label="Label Probabilities / Scores")
                        classify_btn.click(
                            fn=classify_video,
                            inputs=[video_input, labels_input, fps_slider, model_dropdown],
                            outputs=[classify_status, classify_df],
                        )
    return demo


def main() -> None:
    host = os.getenv("SIGLIP_APP_HOST", "127.0.0.1")
    port = int(os.getenv("SIGLIP_APP_PORT", "7860"))
    build_app().launch(theme=gr.themes.Soft(), server_name=host, server_port=port, share=False)


if __name__ == "__main__":
    main()
