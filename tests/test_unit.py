import hashlib
import math
import os
import resource
import tempfile
import time
import unicodedata
import unittest
from typing import Any, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import transformers

from src.dataset import (
    LABEL_MAP,
    _resolve_stec_label,
    group_by_background_correlation,
    stratified_group_kfold,
)
from src.synthetic import generate_benchmark_video, generate_dummy_video
from src.video_processor import VideoProcessor
from src.video_search import SearchResult, VideoSearchEngine


_orig_auto_model = transformers.AutoModel.from_pretrained


def _forbidden_auto_model(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("transformers.AutoModel.from_pretrained must not be called in unit tests")


def setUpModule() -> None:
    transformers.AutoModel.from_pretrained = _forbidden_auto_model


def tearDownModule() -> None:
    transformers.AutoModel.from_pretrained = _orig_auto_model


class _StubModelConfig:
    def __init__(self, hidden_size: int = 768) -> None:
        self.hidden_size = hidden_size


class _StubVisionTextConfig:
    def __init__(self, hidden_size: int = 768) -> None:
        self.vision_config = _StubModelConfig(hidden_size)
        self.text_config = _StubModelConfig(hidden_size)


class _StubModelWrapper:
    def __init__(self, logit_scale: torch.Tensor, logit_bias: torch.Tensor, hidden_size: int = 768) -> None:
        self.logit_scale = logit_scale
        self.logit_bias = logit_bias
        self.config = _StubVisionTextConfig(hidden_size)


class StubSigLIPEngine:
    """Deterministic model stub implementing VideoSearchEngine's engine protocol."""

    def __init__(self, embed_dim: int = 768, seed: int = 42) -> None:
        self.embed_dim = embed_dim
        self.batch_sizes_log: List[int] = []
        self.recorded_texts: List[str] = []
        self.custom_image_embeds: Optional[torch.Tensor] = None
        self._custom_img_offset: int = 0
        self.custom_text_map: Dict[str, torch.Tensor] = {}

        rng = np.random.default_rng(seed)
        self._proj = torch.from_numpy(rng.standard_normal((3, embed_dim), dtype=np.float32))

        self.logit_scale = torch.tensor(4.76)
        self.logit_bias = torch.tensor(-12.93)
        self.model = _StubModelWrapper(self.logit_scale, self.logit_bias, embed_dim)

    def encode_images(self, images: List[Image.Image], batch_size: int = 16) -> torch.Tensor:
        self.batch_sizes_log.append(len(images))
        if not images:
            return torch.empty((0, self.embed_dim), dtype=torch.float32)

        if self.custom_image_embeds is not None:
            start = self._custom_img_offset
            end = start + len(images)
            self._custom_img_offset = end
            return self.custom_image_embeds[start:end]

        vectors: List[torch.Tensor] = []
        for img in images:
            mean_rgb = np.array(img.resize((1, 1), Image.Resampling.BILINEAR))[0, 0].astype(np.float32) / 255.0
            vec = torch.from_numpy(mean_rgb) @ self._proj
            vectors.append(vec)
        stacked = torch.stack(vectors, dim=0)
        return F.normalize(stacked, p=2, dim=-1)

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        self.recorded_texts.extend(texts)
        if not texts:
            return torch.empty((0, self.embed_dim), dtype=torch.float32)

        vectors: List[torch.Tensor] = []
        for txt in texts:
            if txt in self.custom_text_map:
                vectors.append(self.custom_text_map[txt].view(-1))
            else:
                digest = hashlib.sha256(txt.encode("utf-8")).digest()
                seed = int.from_bytes(digest[:8], byteorder="big")
                rng = np.random.default_rng(seed)
                raw = rng.standard_normal(self.embed_dim, dtype=np.float32)
                vectors.append(torch.from_numpy(raw))
        stacked = torch.stack(vectors, dim=0)
        return F.normalize(stacked, p=2, dim=-1)

    def score_logits(
        self, image_embeds: torch.Tensor, text_embeds: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img = image_embeds.to(dtype=torch.float32)
        txt = text_embeds.to(dtype=torch.float32)
        similarities = img @ txt.T
        scale = torch.exp(self.logit_scale)
        probabilities = torch.sigmoid(scale * similarities + self.logit_bias)
        return similarities, probabilities


def _create_synthetic_video(
    file_path: str, fps: float, duration_sec: float, width: int = 64, height: int = 48
) -> str:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(file_path, fourcc, float(fps), (width, height))
    total_frames = int(round(fps * duration_sec))
    for i in range(total_frames):
        color = (int((i * 17) % 256), int((i * 31) % 256), int((i * 53) % 256))
        frame = np.full((height, width, 3), color, dtype=np.uint8)
        out.write(frame)
    out.release()
    return file_path


class TestSamplingFrequency(unittest.TestCase):
    """Test 1: Exact sampling counts, monotonic timestamps, within-range frame indices."""

    def test_sampling_grid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            native_rates = [30, 24, 10]
            target_rates = [4.0, 5.0, 1.0, 0.5]
            duration = 5.0

            for native_fps in native_rates:
                video_path = os.path.join(tmpdir, f"video_{native_fps}fps.mp4")
                total_video_frames = int(round(native_fps * duration))
                _create_synthetic_video(video_path, float(native_fps), duration)

                for target_fps in target_rates:
                    with self.subTest(native=native_fps, target=target_fps):
                        proc = VideoProcessor(target_fps=target_fps)
                        frames = proc.extract_frames(video_path)
                        count = len(frames)

                        expected_count = int(math.ceil(duration * target_fps))
                        self.assertAlmostEqual(count, expected_count, delta=1)

                        effective_fps = count / duration
                        self.assertLessEqual(abs(effective_fps - target_fps), 1.0)

                        timestamps = [f["timestamp"] for f in frames]
                        frame_indices = [f["frame_idx"] for f in frames]

                        self.assertGreater(count, 0)
                        self.assertEqual(timestamps[0], 0.0)

                        for idx in range(1, count):
                            self.assertGreater(timestamps[idx], timestamps[idx - 1])

                        for ts in timestamps:
                            self.assertEqual(ts, round(ts, 2))

                        for fi in frame_indices:
                            self.assertGreaterEqual(fi, 0)
                            self.assertLess(fi, total_video_frames)


class TestVideoProcessorValidation(unittest.TestCase):
    """Test 2: VideoProcessor argument validation and max_frames capping."""

    def test_invalid_target_fps(self) -> None:
        for bad_fps in [0, -1.0, float("nan"), float("inf"), -float("inf")]:
            with self.subTest(bad_fps=bad_fps):
                with self.assertRaises(ValueError):
                    VideoProcessor(target_fps=bad_fps)

    def test_invalid_max_frames(self) -> None:
        for bad_max in [0, -3]:
            with self.subTest(bad_max=bad_max):
                with self.assertRaises(ValueError):
                    VideoProcessor(target_fps=1.0, max_frames=bad_max)

    def test_max_frames_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "cap_test.mp4")
            _create_synthetic_video(video_path, fps=10.0, duration_sec=2.0)
            proc = VideoProcessor(target_fps=5.0, max_frames=5)
            frames = proc.extract_frames(video_path)
            self.assertEqual(len(frames), 5)


class TestClipVideo(unittest.TestCase):
    """Test 3: clip_video exact frame count and range validation."""

    def test_clip_video_exact_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            vpath = os.path.join(tmpdir, "input_10fps.mp4")
            _create_synthetic_video(vpath, fps=10.0, duration_sec=2.0)

            out_clip = os.path.join(tmpdir, "clipped_0_1.mp4")
            VideoProcessor.clip_video(vpath, 0.0, 1.0, out_clip)
            self.assertTrue(os.path.exists(out_clip))

            cap = cv2.VideoCapture(out_clip)
            frame_count = 0
            while True:
                ret, _ = cap.read()
                if not ret:
                    break
                frame_count += 1
            cap.release()
            self.assertEqual(frame_count, 10)

    def test_clip_video_invalid_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            vpath = os.path.join(tmpdir, "input_range.mp4")
            _create_synthetic_video(vpath, fps=10.0, duration_sec=2.0)
            out_clip = os.path.join(tmpdir, "invalid_clip.mp4")

            for start, end in [(1.0, 1.0), (2.0, 1.0), (10.0, 12.0)]:
                with self.subTest(start=start, end=end):
                    with self.assertRaises(ValueError):
                        VideoProcessor.clip_video(vpath, start, end, out_clip)
                    self.assertFalse(os.path.exists(out_clip))

    def test_clip_video_bad_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_clip = os.path.join(tmpdir, "out.mp4")
            with self.assertRaises(ValueError):
                VideoProcessor.clip_video(os.path.join(tmpdir, "missing.mp4"), 0.0, 1.0, out_clip)
            self.assertFalse(os.path.exists(out_clip))


class TestTemporalNMS(unittest.TestCase):
    """Test 4: Temporal NMS suppression, score ordering, and argument validation."""

    def test_nms_suppression_and_ordering(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            vpath = os.path.join(tmpdir, "nms_video.mp4")
            _create_synthetic_video(vpath, fps=2.0, duration_sec=4.5)

            stub = StubSigLIPEngine()
            query_str = "target visual pattern"
            q_vec = torch.zeros((1, 768), dtype=torch.float32)
            q_vec[0, 0] = 1.0
            stub.custom_text_map[query_str] = q_vec

            scores = [0.95, 0.50, 0.50, 0.50, 0.90, 0.50, 0.50, 0.50, 0.85]
            crafted_embeds: List[torch.Tensor] = []
            for s in scores:
                v = torch.zeros(768, dtype=torch.float32)
                v[0] = s
                v[1] = math.sqrt(max(0.0, 1.0 - s * s))
                crafted_embeds.append(v)
            stub.custom_image_embeds = torch.stack(crafted_embeds, dim=0)

            proc = VideoProcessor(target_fps=2.0)
            searcher = VideoSearchEngine(engine=stub, processor=proc)
            searcher.index_video(vpath, batch_size=8)

            suppressed_results = searcher.search_moments(query=query_str, top_k=3, min_gap_sec=2.0)
            self.assertEqual(len(suppressed_results), 3)

            for i in range(len(suppressed_results) - 1):
                self.assertGreaterEqual(suppressed_results[i].score, suppressed_results[i + 1].score)

            ts_list = [r.timestamp for r in suppressed_results]
            for i in range(len(ts_list)):
                for j in range(i + 1, len(ts_list)):
                    self.assertGreaterEqual(abs(ts_list[i] - ts_list[j]), 2.0)

            dense_results = searcher.search_moments(query=query_str, top_k=3, min_gap_sec=0.0)
            self.assertEqual(len(dense_results), 3)
            dense_ts = [r.timestamp for r in dense_results]
            self.assertEqual(dense_ts, [0.0, 2.0, 4.0])

    def test_search_moments_validation(self) -> None:
        stub = StubSigLIPEngine()
        searcher = VideoSearchEngine(engine=stub)

        with self.assertRaises(RuntimeError):
            searcher.search_moments("unindexed query")

        searcher.frame_embeddings = torch.randn((2, 768))
        searcher.frame_metadata = [
            {"timestamp": 0.0, "frame_idx": 0, "video_path": "a.mp4"},
            {"timestamp": 1.0, "frame_idx": 1, "video_path": "a.mp4"},
        ]

        for empty_q in ["", "   ", "\t\n"]:
            with self.subTest(query=empty_q):
                with self.assertRaises(ValueError):
                    searcher.search_moments(empty_q)

        for bad_k in [0, -1]:
            with self.subTest(top_k=bad_k):
                with self.assertRaises(ValueError):
                    searcher.search_moments("query", top_k=bad_k)

        with self.assertRaises(ValueError):
            searcher.search_moments("query", min_gap_sec=-1.0)


class TestIndexVideoBatching(unittest.TestCase):
    """Test 5: Batching breakdown, index row synchronization, multi-video append, missing file."""

    def test_batching_and_append(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            vpath1 = os.path.join(tmpdir, "vid10.mp4")
            _create_synthetic_video(vpath1, fps=10.0, duration_sec=1.0)

            vpath2 = os.path.join(tmpdir, "vid5.mp4")
            _create_synthetic_video(vpath2, fps=10.0, duration_sec=0.5)

            stub = StubSigLIPEngine()
            proc = VideoProcessor(target_fps=10.0)
            searcher = VideoSearchEngine(engine=stub, processor=proc)

            indexed1 = searcher.index_video(vpath1, batch_size=4)
            self.assertEqual(indexed1, 10)
            self.assertEqual(stub.batch_sizes_log, [4, 4, 2])
            self.assertIsNotNone(searcher.frame_embeddings)
            self.assertEqual(searcher.frame_embeddings.shape[0], 10)
            self.assertEqual(len(searcher.frame_metadata), 10)

            indexed2 = searcher.index_video(vpath2, batch_size=4)
            self.assertEqual(indexed2, 5)
            self.assertEqual(stub.batch_sizes_log, [4, 4, 2, 4, 1])
            self.assertEqual(searcher.frame_embeddings.shape[0], 15)
            self.assertEqual(len(searcher.frame_metadata), 15)

            with self.assertRaises(FileNotFoundError):
                searcher.index_video(os.path.join(tmpdir, "nonexistent.mp4"), batch_size=4)

            self.assertEqual(searcher.frame_embeddings.shape[0], 15)
            self.assertEqual(len(searcher.frame_metadata), 15)


class TestClassifyVideoZeroShot(unittest.TestCase):
    """Test 6: Label validation, template handling, output keys, and probability ranges."""

    def test_classify_zero_shot_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            vpath = os.path.join(tmpdir, "classify_vid.mp4")
            _create_synthetic_video(vpath, fps=10.0, duration_sec=0.5)

            stub = StubSigLIPEngine()
            proc = VideoProcessor(target_fps=10.0)
            searcher = VideoSearchEngine(engine=stub, processor=proc)

            with self.assertRaises(ValueError) as ctx:
                searcher.classify_video_zero_shot([])
            self.assertIn("Candidate labels cannot be empty", str(ctx.exception))

            with self.assertRaises(RuntimeError):
                searcher.classify_video_zero_shot(["dog", "cat"])

            searcher.index_video(vpath, batch_size=5)

            stub.recorded_texts.clear()
            labels = ["dog", "cat", "a photo of a {}"]
            rankings = searcher.classify_video_zero_shot(labels, template="This is a photo of {}.")

            expected_encoded = ["This is a photo of dog.", "This is a photo of cat.", "a photo of a {}"]
            self.assertEqual(stub.recorded_texts, expected_encoded)

            self.assertEqual(len(rankings), 3)
            returned_labels = [r["label"] for r in rankings]
            for lbl in labels:
                self.assertIn(lbl, returned_labels)

            required_keys = {"label", "mean_score", "max_score", "mean_prob", "max_prob"}
            for entry in rankings:
                self.assertEqual(set(entry.keys()), required_keys)
                self.assertGreaterEqual(entry["mean_prob"], 0.0)
                self.assertLessEqual(entry["mean_prob"], 1.0)
                self.assertGreaterEqual(entry["max_prob"], 0.0)
                self.assertLessEqual(entry["max_prob"], 1.0)

            stub.recorded_texts.clear()
            raw_labels = ["action", "thriller"]
            rankings_raw = searcher.classify_video_zero_shot(raw_labels, template=None)
            self.assertEqual(stub.recorded_texts, raw_labels)
            self.assertEqual(len(rankings_raw), 2)


class TestTextTruncation(unittest.TestCase):
    """Test 7: Offline tokenization contract with max_length=64 truncation."""

    def test_tokenizer_truncation_contract(self) -> None:
        t_start = time.perf_counter()
        try:
            processor = transformers.AutoProcessor.from_pretrained(
                "google/siglip-base-patch16-224", local_files_only=True
            )
        except Exception as exc:
            self.skipTest(f"Offline AutoProcessor unavailable: {exc}")

        load_duration = time.perf_counter() - t_start
        if load_duration > 1.0:
            self.skipTest(f"Tokenizer load too slow ({load_duration:.2f}s > 1.0s)")

        long_query = " ".join([f"word{i}" for i in range(101)])
        tokens = processor(
            text=[long_query],
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        self.assertEqual(tokens["input_ids"].shape, torch.Size([1, 64]))


class TestMemoryBound(unittest.TestCase):
    """Test 8: Memory bounding (F12 / IS-3) streaming 300-frame 640x360 video."""

    def test_streaming_rss_bound(self) -> None:
        # Comparison measurement: 300 frames of 640x360 raw RGB buffers equal
        # 300 * 640 * 360 * 3 = 207,360,000 bytes (~197.75 MB). Retaining all frames
        # in a PIL list causes an immediate resident memory spike proportional to duration.
        with tempfile.TemporaryDirectory() as tmpdir:
            vpath = os.path.join(tmpdir, "long_stream.mp4")
            _create_synthetic_video(vpath, fps=30.0, duration_sec=10.0, width=640, height=360)

            rss_before_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

            stub = StubSigLIPEngine()
            proc = VideoProcessor(target_fps=30.0)
            searcher = VideoSearchEngine(engine=stub, processor=proc)

            indexed_count = searcher.index_video(vpath, batch_size=16)
            self.assertEqual(indexed_count, 300)

            # Assert searcher does not retain raw PIL Image objects
            for meta in searcher.frame_metadata:
                self.assertNotIn("image", meta)
            self.assertFalse(hasattr(searcher, "images"))
            self.assertFalse(hasattr(searcher, "frames"))

            rss_after_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            delta_mb = (rss_after_kb - rss_before_kb) / 1024.0
            print(f"\n[TestMemoryBound] 300 frames 640x360 streamed: peak RSS growth = {delta_mb:.2f} MB")
            self.assertLess(delta_mb, 300.0, f"Peak RSS growth ({delta_mb:.2f} MB) exceeded 300 MB limit")


class TestDatasetPureFunctions(unittest.TestCase):
    """Test 9: Pure function algorithms in src.dataset."""

    def test_group_by_background_correlation(self) -> None:
        v0 = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
        v1 = np.array([10.05, 19.95, 30.05, 39.95], dtype=np.float32)
        v2 = np.array([40.0, 30.0, 20.0, 10.0], dtype=np.float32)
        v3 = np.array([0.0, 50.0, 0.0, 50.0], dtype=np.float32)

        feats = np.stack([v0, v1, v2, v3])
        groups = group_by_background_correlation(feats, threshold=0.90)

        self.assertEqual(len(groups), 4)
        self.assertEqual(groups[0], groups[1])
        self.assertNotEqual(groups[2], groups[0])
        self.assertNotEqual(groups[3], groups[0])
        self.assertNotEqual(groups[2], groups[3])

    def test_stratified_group_kfold(self) -> None:
        labels = [
            "fight", "fight", "theft", "theft", "accident",
            "accident", "fire", "fire", "vandalism", "vandalism",
        ]
        groups = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
        n_samples = len(labels)

        splits1 = stratified_group_kfold(labels, groups, n_splits=3, seed=42)
        splits2 = stratified_group_kfold(labels, groups, n_splits=3, seed=42)

        for (tr1, te1), (tr2, te2) in zip(splits1, splits2):
            self.assertTrue(np.array_equal(tr1, tr2))
            self.assertTrue(np.array_equal(te1, te2))

        all_tested_indices: List[int] = []
        for train_idx, test_idx in splits1:
            self.assertEqual(len(set(train_idx) & set(test_idx)), 0)
            self.assertEqual(set(train_idx) | set(test_idx), set(range(n_samples)))

            tr_groups = {groups[i] for i in train_idx}
            te_groups = {groups[i] for i in test_idx}
            self.assertEqual(len(tr_groups & te_groups), 0)
            all_tested_indices.extend(test_idx.tolist())

        self.assertEqual(sorted(all_tested_indices), list(range(n_samples)))

        with self.assertRaises(ValueError):
            stratified_group_kfold(labels, groups, n_splits=10, seed=42)

    def test_unicode_label_mapping(self) -> None:
        nfd_theft = unicodedata.normalize("NFD", "Ăn trộm")
        nfc_theft = unicodedata.normalize("NFC", "Ăn trộm")
        self.assertNotEqual(nfd_theft, nfc_theft)

        self.assertEqual(_resolve_stec_label(f"dir/{nfd_theft}/clip.mp4"), "theft")
        self.assertEqual(_resolve_stec_label(f"dir/{nfc_theft}/clip.mp4"), "theft")

        nfd_fight = unicodedata.normalize("NFD", "Đánh nhau")
        self.assertEqual(_resolve_stec_label(f"camera_01/{nfd_fight}/clip.mp4"), "fighting")

        nfd_fall = unicodedata.normalize("NFD", "NGÃ")
        self.assertEqual(_resolve_stec_label(f"camera_02/{nfd_fall}/clip.mp4"), "people_falling")

        self.assertIsNone(_resolve_stec_label("smartbench_test.mp4"))


class TestSyntheticGeneration(unittest.TestCase):
    """Test 10: Synthetic dummy video and benchmark generator contracts."""

    def test_generate_dummy_video_bare_filename_in_temp_cwd(self) -> None:
        orig_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                os.chdir(tmpdir)
                out = generate_dummy_video(output_path="demo.mp4", duration_sec=1)
                self.assertEqual(out, "demo.mp4")
                self.assertTrue(os.path.exists("demo.mp4"))
                self.assertGreater(os.path.getsize("demo.mp4"), 0)
            finally:
                os.chdir(orig_cwd)

    def test_generate_benchmark_video_scene_count_and_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            p_no_text = os.path.join(tmpdir, "bench_no_text.mp4")
            p_with_text = os.path.join(tmpdir, "bench_with_text.mp4")

            scenes = generate_benchmark_video(
                p_no_text, duration_per_scene=0.2, fps=10, with_text_overlay=False
            )
            self.assertEqual(len(scenes), 8)

            generate_benchmark_video(
                p_with_text, duration_per_scene=0.2, fps=10, with_text_overlay=True
            )

            cap_no = cv2.VideoCapture(p_no_text)
            ret_no, frame_no = cap_no.read()
            cap_no.release()
            self.assertTrue(ret_no)

            cap_with = cv2.VideoCapture(p_with_text)
            ret_with, frame_with = cap_with.read()
            cap_with.release()
            self.assertTrue(ret_with)

            header_no = frame_no[10:45, 10:200]
            header_with = frame_with[10:45, 10:200]
            self.assertFalse(np.array_equal(header_no, header_with))


if __name__ == "__main__":
    unittest.main()
