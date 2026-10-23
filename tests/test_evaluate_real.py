import json
import os
import tempfile
import unittest
from typing import Any, List
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from evaluate_real import evaluate_surveillance, main, parse_args
from src.dataset import VideoSample, compute_sha256
from src.features import _write_cache_atomic, get_feature_cache_path
from src.heads import load_head
from src.metrics import (
    STEC_TO_MSAD,
    evaluate_stec_probe,
    paired_bootstrap_test,
    wilson_score_interval,
)


class MockSigLIPEngine:
    """Hermetic deterministic SigLIP engine stub for unit tests."""

    def __init__(self, hidden_size: int = 64) -> None:
        self.model_name = "mock/siglip-test"
        self.hidden_size = hidden_size
        self.model = type("StubModel", (), {
            "config": type("StubCfg", (), {
                "vision_config": type("SubCfg", (), {"hidden_size": hidden_size})(),
                "text_config": type("SubCfg", (), {"hidden_size": hidden_size})(),
            })(),
            "logit_scale": torch.tensor(4.0),
            "logit_bias": torch.tensor(-2.0),
        })()

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        vecs = []
        for t in texts:
            seed = sum(ord(c) for c in t) % (2**31)
            rng = torch.Generator().manual_seed(seed)
            vecs.append(F.normalize(torch.randn(self.hidden_size, generator=rng), p=2, dim=-1))
        return torch.stack(vecs, dim=0)

    def score_logits(self, img_embeds: torch.Tensor, txt_embeds: torch.Tensor) -> Any:
        sims = img_embeds.float() @ txt_embeds.float().T
        probs = torch.sigmoid(sims * 4.0 - 2.0)
        return sims, probs


class TestStatisticalMetrics(unittest.TestCase):
    def test_wilson_score_interval(self) -> None:
        # Empty trials
        self.assertEqual(wilson_score_interval(0, 0), (0.0, 0.0))
        # Zero successes
        low, high = wilson_score_interval(0, 10)
        self.assertEqual(low, 0.0)
        self.assertGreater(high, 0.0)
        # All successes
        low, high = wilson_score_interval(10, 10)
        self.assertLess(low, 1.0)
        self.assertEqual(high, 1.0)
        # 5 of 10
        low, high = wilson_score_interval(5, 10)
        self.assertGreater(low, 0.2)
        self.assertLess(high, 0.8)

    def test_paired_bootstrap_test(self) -> None:
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1] * 5)
        # Adapter perfect, zero-shot completely wrong
        y_ad = y_true.copy()
        y_zs = 1 - y_true
        diff, ci_low, ci_high = paired_bootstrap_test(
            y_true=y_true, y_adapter=y_ad, y_zero_shot=y_zs, num_classes=2, n_resamples=100, seed=42
        )
        self.assertGreater(diff, 0.5)
        self.assertGreater(ci_low, 0.0)
        self.assertGreaterEqual(ci_high, ci_low)


class TestEvaluateRealPipeline(unittest.TestCase):
    def test_cli_argument_parsing(self) -> None:
        args = parse_args(["--fps", "2.0", "--seed", "42", "--n-splits", "3"])
        self.assertEqual(args.fps, 2.0)
        self.assertEqual(args.seed, 42)
        self.assertEqual(args.n_splits, 3)

    def test_main_missing_root_exits_one(self) -> None:
        ret = main(["--msad-root", "/nonexistent/test/path", "--stec-root", "/nonexistent/test/path"])
        self.assertEqual(ret, 1)

    def test_mock_surveillance_cv_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            msad_dir = os.path.join(tmpdir, "msad")
            stec_dir = os.path.join(tmpdir, "stec")
            cache_dir = os.path.join(tmpdir, "cache")
            ckpt_dir = os.path.join(tmpdir, "checkpoints")
            out_json = os.path.join(tmpdir, "results.json")
            os.makedirs(msad_dir)
            os.makedirs(stec_dir)

            classes = ["Fighting", "Fire", "Robbery"]
            mock_samples_msad = []
            hidden_size = 32
            mock_engine = MockSigLIPEngine(hidden_size=hidden_size)

            # Create mock cached features for MSAD
            for i in range(12):
                cls = classes[i % 3]
                path = os.path.join(msad_dir, f"{cls}_{i}.mp4")
                with open(path, "w") as f:
                    f.write(f"mock_video_bytes_{cls}_{i}")
                sha = compute_sha256(path)
                sample = VideoSample(
                    path=path, label=cls, source="msad", sha256=sha, group_id=i // 2, duration_sec=5.0, fps=1.0
                )
                mock_samples_msad.append(sample)

                # Pre-populate feature cache
                c_path = get_feature_cache_path(cache_dir, mock_engine.model_name, sha, 1.0, None)
                embeds = np.random.randn(3, hidden_size).astype(np.float32)
                embeds = embeds / np.linalg.norm(embeds, axis=-1, keepdims=True)
                timestamps = np.array([0.0, 1.0, 2.0], dtype=np.float32)
                _write_cache_atomic(c_path, embeds, timestamps, path, 5.0, mock_engine.model_name, 1.0, None)

            # Create mock cached features for STEC
            mock_samples_stec = []
            stec_classes = ["fighting", "theft"]
            for j in range(4):
                cls = stec_classes[j % 2]
                path = os.path.join(stec_dir, f"clip_{j}.mp4")
                with open(path, "w") as f:
                    f.write(f"mock_stec_bytes_{cls}_{j}")
                sha = compute_sha256(path)
                sample = VideoSample(
                    path=path, label=cls, source="stec", sha256=sha, group_id=j, duration_sec=5.0, fps=1.0
                )
                mock_samples_stec.append(sample)
                c_path = get_feature_cache_path(cache_dir, mock_engine.model_name, sha, 1.0, None)
                embeds = np.random.randn(2, hidden_size).astype(np.float32)
                embeds = embeds / np.linalg.norm(embeds, axis=-1, keepdims=True)
                timestamps = np.array([0.0, 1.0], dtype=np.float32)
                _write_cache_atomic(c_path, embeds, timestamps, path, 5.0, mock_engine.model_name, 1.0, None)

            mock_manifest = {"msad": mock_samples_msad, "stec": mock_samples_stec}

            with patch("evaluate_real.load_surveillance_manifest", return_value=mock_manifest):
                args = parse_args([
                    "--msad-root", msad_dir,
                    "--stec-root", stec_dir,
                    "--cache-dir", cache_dir,
                    "--checkpoint-dir", ckpt_dir,
                    "--out", out_json,
                    "--model", mock_engine.model_name,
                    "--fps", "1.0",
                    "--seed", "42",
                    "--n-splits", "3",
                ])
                res = evaluate_surveillance(args, engine=mock_engine)

            # Verify report keys
            self.assertIn("msad_5fold_pooled_macro_f1", res)
            self.assertIn("zero_shot", res["msad_5fold_pooled_macro_f1"])
            self.assertIn("prompt_ensemble", res["msad_5fold_pooled_macro_f1"])
            self.assertIn("adapter", res["msad_5fold_pooled_macro_f1"])
            self.assertIn("bootstrap_ci_adapter_minus_zero_shot", res)
            self.assertIn("verdict", res)
            self.assertIn(res["verdict"], ("gain", "no_gain"))
            self.assertIn("stec_cross_probe", res)
            self.assertIn("per_class_metrics", res)
            self.assertEqual(res["n_videos"], 12)
            self.assertEqual(res["n_frames"], 12 * 3 + 4 * 2)
            self.assertEqual(res["model_id"], mock_engine.model_name)
            self.assertEqual(res["seed"], 42)

            # Check JSON file written
            self.assertTrue(os.path.exists(out_json))
            with open(out_json, "r") as f:
                saved_json = json.load(f)
            self.assertEqual(saved_json["n_videos"], 12)

            # Check checkpoint saved and loadable
            ckpt_path = os.path.join(ckpt_dir, "adapter_msad_base224.pt")
            self.assertTrue(os.path.exists(ckpt_path))
            loaded_head, meta = load_head(ckpt_path)
            self.assertEqual(loaded_head.linear.weight.shape, (3, hidden_size))
            self.assertEqual(meta["model_name"], mock_engine.model_name)


if __name__ == "__main__":
    unittest.main()
