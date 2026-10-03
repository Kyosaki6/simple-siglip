import os
import tempfile
from typing import List, Tuple
import gradio as gr
import pandas as pd
from PIL import Image

from src.siglip_engine import SigLIPEngine
from src.video_processor import VideoProcessor
from src.video_search import VideoSearchEngine
from main import generate_dummy_video


# Global state
_engine_cache = {}


def get_engine(model_name: str) -> SigLIPEngine:
    if model_name not in _engine_cache:
        _engine_cache[model_name] = SigLIPEngine(model_name=model_name)
    return _engine_cache[model_name]


def process_video_and_search(
    video_file,
    query_text: str,
    fps: float,
    model_name: str,
    top_k: int,
):
    if not video_file:
        dummy_path = "sample_data/demo.mp4"
        if not os.path.exists(dummy_path):
            generate_dummy_video(dummy_path)
        video_path = dummy_path
    else:
        video_path = video_file if isinstance(video_file, str) else video_file.name

    engine = get_engine(model_name)
    processor = VideoProcessor(target_fps=float(fps))
    searcher = VideoSearchEngine(engine=engine, processor=processor)

    num_frames = searcher.index_video(video_path)
    if num_frames == 0:
        return f"Could not extract frames from {video_path}", [], None, None

    results = searcher.search_moments(query_text, top_k=int(top_k))

    gallery_items = []
    summary_rows = []
    top_timestamp = None

    for rank, res in enumerate(results, 1):
        frame_img = VideoProcessor.extract_frame_at_timestamp(video_path, res.timestamp)
        caption = f"#{rank} | {res.timestamp:.2f}s | Score: {res.score:.4f}"
        if frame_img is not None:
            gallery_items.append((frame_img, caption))
        summary_rows.append(
            {
                "Rank": rank,
                "Timestamp (s)": res.timestamp,
                "Frame Index": res.frame_idx,
                "Similarity Score": res.score,
            }
        )
        if top_timestamp is None:
            top_timestamp = res.timestamp

    # Generate snippet around top result
    clipped_video_path = None
    if top_timestamp is not None:
        temp_dir = tempfile.gettempdir()
        out_clip = os.path.join(temp_dir, f"clip_{os.path.basename(video_path)}_{int(top_timestamp)}.mp4")
        start_t = max(0.0, top_timestamp - 1.5)
        end_t = top_timestamp + 1.5
        VideoProcessor.clip_video(video_path, start_t, end_t, out_clip)
        clipped_video_path = out_clip

    df = pd.DataFrame(summary_rows)
    msg = f"Indexed {num_frames} frames ({fps} FPS). Found {len(results)} matches for '{query_text}'."
    return msg, gallery_items, df, clipped_video_path


def classify_video(
    video_file,
    labels_str: str,
    fps: float,
    model_name: str,
):
    if not video_file:
        dummy_path = "sample_data/demo.mp4"
        if not os.path.exists(dummy_path):
            generate_dummy_video(dummy_path)
        video_path = dummy_path
    else:
        video_path = video_file if isinstance(video_file, str) else video_file.name

    labels = [l.strip() for l in labels_str.split(",") if l.strip()]
    if not labels:
        return "Please provide at least one label.", None

    engine = get_engine(model_name)
    processor = VideoProcessor(target_fps=float(fps))
    searcher = VideoSearchEngine(engine=engine, processor=processor)

    num_frames = searcher.index_video(video_path)
    rankings = searcher.classify_video_zero_shot(labels)

    df = pd.DataFrame(
        [
            {
                "Candidate Label": r["label"],
                "Mean Score": r["mean_score"],
                "Peak Score": r["max_score"],
            }
            for r in rankings
        ]
    )

    msg = f"Indexed {num_frames} frames. Classified against {len(labels)} categories."
    return msg, df


def build_app():
    with gr.Blocks(title="SigLIP Video Intelligence Lab", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🎥 SigLIP Video Intelligence Lab
            **Text-to-Moment Retrieval & Zero-shot Classification powered by SigLIP (Sigmoid Loss for Language Image Pre-training)**
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(label="Input Video (Leave empty to use sample demo)")
                model_dropdown = gr.Dropdown(
                    choices=[
                        "google/siglip-base-patch16-224",
                        "google/siglip-so400m-patch14-384",
                    ],
                    value="google/siglip-base-patch16-224",
                    label="SigLIP Backbone",
                )
                fps_slider = gr.Slider(
                    minimum=0.5, maximum=5.0, value=1.0, step=0.5, label="Sampling Rate (FPS)"
                )

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("🔍 Text-to-Moment Search"):
                        query_input = gr.Textbox(
                            value="blue ocean with water",
                            label="Natural Language Query",
                            placeholder="e.g., 'a red apple', 'goal celebration', 'car speeding'",
                        )
                        top_k_slider = gr.Slider(
                            minimum=1, maximum=10, value=3, step=1, label="Top K Moments"
                        )
                        search_btn = gr.Button("Search Video Moments", variant="primary")
                        search_status = gr.Markdown()
                        results_gallery = gr.Gallery(
                            label="Matching Frame Moments", columns=3, height="auto"
                        )
                        results_df = gr.Dataframe(label="Detailed Rankings")
                        clipped_output = gr.Video(label="Auto-Extracted Top Clip (±1.5s)")

                        search_btn.click(
                            fn=process_video_and_search,
                            inputs=[video_input, query_input, fps_slider, model_dropdown, top_k_slider],
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


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
