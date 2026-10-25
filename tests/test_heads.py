import os
import tempfile
import time
import unittest
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from src.heads import (
    DEFAULT_TEMPLATES,
    LinearAdapterHead,
    confusion_matrix,
    encode_prompt_ensemble,
    load_head,
    macro_f1,
    per_class_f1,
    pool_video_embedding,
    predict,
    save_head,
    train_head,
)


class DummySigLIPEngine:
    """Mock SigLIPEngine for testing prompt ensembling without weights."""

    def __init__(self, embed_dim: int = 768) -> None:
        self.embed_dim = embed_dim
        self.recorded_queries: List[str] = []

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        self.recorded_queries.extend(texts)
        # Deterministic pseudo-embeddings based on string length and char codes
        vectors = []
        for text in texts:
            seed = sum(ord(c) for c in text)
            gen = torch.Generator().manual_seed(seed)
            vec = torch.randn(self.embed_dim, generator=gen)
            vectors.append(F.normalize(vec, p=2, dim=-1))
        return torch.stack(vectors, dim=0)


class TestLinearAdapterHead(unittest.TestCase):
    def test_forward_shape(self) -> None:
        head = LinearAdapterHead(768, 11)
        x = torch.randn(10, 768)
        y = head(x)
        self.assertEqual(y.shape, (10, 11))

    def test_forward_invalid_dim_raises_value_error(self) -> None:
        head = LinearAdapterHead(768, 11)
        # 1D input
        with self.assertRaises(ValueError) as ctx1:
            head(torch.randn(768))
        self.assertIn("Expected 2D input with 768 features", str(ctx1.exception))

        # 2D input with mismatched in_features
        with self.assertRaises(ValueError) as ctx2:
            head(torch.randn(10, 512))
        self.assertIn("Expected 2D input with 768 features", str(ctx2.exception))

        # 3D input
        with self.assertRaises(ValueError) as ctx3:
            head(torch.randn(2, 5, 768))
        self.assertIn("Expected 2D input with 768 features", str(ctx3.exception))

    def test_invalid_init_parameters(self) -> None:
        with self.assertRaises(ValueError):
            LinearAdapterHead(0, 11)
        with self.assertRaises(ValueError):
            LinearAdapterHead(768, -1)


class TestPredict(unittest.TestCase):
    def test_predict_outputs(self) -> None:
        head = LinearAdapterHead(64, 4)
        x = torch.randn(8, 64)
        preds, probs = predict(head, x)
        self.assertEqual(preds.shape, (8,))
        self.assertEqual(probs.shape, (8, 4))
        self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.ones(8), atol=1e-5))
        self.assertTrue(torch.equal(preds, torch.argmax(probs, dim=-1)))


class TestPooling(unittest.TestCase):
    def test_mean_pooling(self) -> None:
        frames = torch.randn(20, 128)
        pooled = pool_video_embedding(frames, mode="mean")
        self.assertEqual(pooled.shape, (128,))
        self.assertAlmostEqual(float(torch.norm(pooled, p=2).item()), 1.0, places=5)

    def test_topk_mean_pooling(self) -> None:
        frames = torch.randn(30, 128)
        pooled = pool_video_embedding(frames, mode="topk_mean", top_k=5)
        self.assertEqual(pooled.shape, (128,))
        self.assertAlmostEqual(float(torch.norm(pooled, p=2).item()), 1.0, places=5)

    def test_empty_or_invalid_pooling(self) -> None:
        with self.assertRaises(ValueError):
            pool_video_embedding(torch.empty((0, 128)), mode="mean")
        with self.assertRaises(ValueError):
            pool_video_embedding(torch.randn(128), mode="mean")
        with self.assertRaises(ValueError):
            pool_video_embedding(torch.randn(10, 128), mode="invalid_mode")
        with self.assertRaises(ValueError):
            pool_video_embedding(torch.randn(10, 128), mode="topk_mean", top_k=0)


class TestPromptEnsemble(unittest.TestCase):
    def test_encode_prompt_ensemble(self) -> None:
        engine = DummySigLIPEngine(embed_dim=256)
        labels = ["People_falling", "Fighting", "Water_incident"]
        embeds = encode_prompt_ensemble(labels, engine, DEFAULT_TEMPLATES)
        self.assertEqual(embeds.shape, (3, 256))
        # Unit norm rows
        norms = torch.norm(embeds, p=2, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones(3), atol=1e-5))
        # Verify cleaned labels were formatted
        self.assertIn("a surveillance video of people falling", engine.recorded_queries)
        self.assertIn("fighting in surveillance camera", engine.recorded_queries)

    def test_empty_labels(self) -> None:
        engine = DummySigLIPEngine(embed_dim=128)
        embeds = encode_prompt_ensemble([], engine)
        self.assertEqual(embeds.shape[0], 0)


class TestMetrics(unittest.TestCase):
    def test_confusion_matrix_and_f1(self) -> None:
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 2, 1])
        cm = confusion_matrix(y_true, y_pred, num_classes=3)
        self.assertEqual(cm.shape, (3, 3))
        self.assertEqual(cm[0, 0], 2)
        self.assertEqual(cm[1, 1], 1)
        self.assertEqual(cm[1, 2], 1)

        f1s = per_class_f1(y_true, y_pred, num_classes=3)
        self.assertEqual(len(f1s), 3)
        self.assertAlmostEqual(f1s[0], 1.0)
        # Class 1: TP=1, FP=1, FN=1 -> F1 = 2*1 / (2 + 1 + 1) = 0.5
        self.assertAlmostEqual(f1s[1], 0.5)
        # Class 2: TP=1, FP=1, FN=1 -> F1 = 0.5
        self.assertAlmostEqual(f1s[2], 0.5)

        mf1 = macro_f1(y_true, y_pred, num_classes=3)
        self.assertAlmostEqual(mf1, (1.0 + 0.5 + 0.5) / 3.0)

    def test_empty_class_zero_f1(self) -> None:
        y_true = np.array([0, 0])
        y_pred = np.array([0, 0])
        f1s = per_class_f1(y_true, y_pred, num_classes=3)
        self.assertEqual(f1s[1], 0.0)
        self.assertEqual(f1s[2], 0.0)


class TestHeadTraining(unittest.TestCase):
    def test_training_convergence_and_speed(self) -> None:
        torch.manual_seed(42)
        n_samples, dim, num_classes = 240, 768, 11
        centers = torch.randn(num_classes, dim)
        y_train = torch.randint(0, num_classes, (n_samples,))
        x_train = F.normalize(centers[y_train] + 0.05 * torch.randn(n_samples, dim), p=2, dim=-1)

        y_val = torch.randint(0, num_classes, (60,))
        x_val = F.normalize(centers[y_val] + 0.05 * torch.randn(60, dim), p=2, dim=-1)

        t0 = time.time()
        head, history = train_head(
            features=x_train,
            labels=y_train,
            num_classes=num_classes,
            val_features=x_val,
            val_labels=y_val,
            max_epochs=100,
            patience=5,
        )
        elapsed = time.time() - t0

        self.assertLess(elapsed, 30.0, f"Training took too long: {elapsed:.2f}s")
        self.assertLessEqual(history["epochs"], 100)
        self.assertTrue(history["converged"])
        self.assertLess(history["val_loss"][-1], history["val_loss"][0])

    def test_training_with_init_weight(self) -> None:
        torch.manual_seed(0)
        x = torch.randn(50, 64)
        y = torch.randint(0, 5, (50,))
        init_w = torch.randn(5, 64)
        head, _ = train_head(x, y, num_classes=5, init_weight=init_w, max_epochs=5)
        self.assertEqual(head.linear.weight.shape, (5, 64))


class TestSerialization(unittest.TestCase):
    def test_save_load_roundtrip(self) -> None:
        head = LinearAdapterHead(128, 4)
        classes = ["class_0", "class_1", "class_2", "class_3"]
        hyperparams = {"lr": 0.1, "weight_decay": 0.01}
        model_name = "google/siglip-base-patch16-224"

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "adapter.npz")
            saved = save_head(
                ckpt_path,
                head=head,
                class_names=classes,
                hyperparams=hyperparams,
                model_name=model_name,
            )
            self.assertTrue(os.path.exists(saved))

            # Successful load
            loaded_head, payload = load_head(
                saved, expected_model_name=model_name, expected_classes=classes
            )
            self.assertTrue(torch.allclose(loaded_head.linear.weight, head.linear.weight))
            self.assertTrue(torch.allclose(loaded_head.linear.bias, head.linear.bias))
            self.assertEqual(payload["class_names"], classes)
            self.assertEqual(payload["model_name"], model_name)

            # Model mismatch raises ValueError
            with self.assertRaises(ValueError):
                load_head(saved, expected_model_name="mismatched-model")

            # Classes mismatch raises ValueError
            with self.assertRaises(ValueError):
                load_head(saved, expected_classes=["a", "b", "c", "d"])


if __name__ == "__main__":
    unittest.main()
