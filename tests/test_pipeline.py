import os
import tempfile
import unittest
from typing import Optional

import cv2
import numpy as np

from src.siglip_engine import SigLIPEngine
from src.video_processor import VideoProcessor
from src.video_search import VideoSearchEngine


class TestSigLIPVideoPipeline(unittest.TestCase):
    temp_dir: Optional[tempfile.TemporaryDirectory] = None
    test_video_path: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        try:
            cls.test_video_path = os.path.join(cls.temp_dir.name, "test_dummy.mp4")
            fps = 10
            total_frames = 60
            width, height = 160, 120
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(cls.test_video_path, fourcc, fps, (width, height))
            try:
                blue_color = (255, 0, 0)
                red_color = (0, 0, 255)
                for i in range(total_frames):
                    color = blue_color if i < 30 else red_color
                    frame = np.full((height, width, 3), color, dtype=np.uint8)
                    out.write(frame)
            finally:
                out.release()
        except Exception:
            cls.temp_dir.cleanup()
            raise

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.temp_dir is not None:
            cls.temp_dir.cleanup()

    def test_video_processor(self) -> None:
        proc = VideoProcessor(target_fps=1.0)
        frames = proc.extract_frames(self.test_video_path)
        self.assertGreater(len(frames), 0)
        self.assertIn("image", frames[0])
        self.assertIn("timestamp", frames[0])

        img = VideoProcessor.extract_frame_at_timestamp(self.test_video_path, 0.5)
        self.assertIsNotNone(img)

        assert self.temp_dir is not None
        clip_path = os.path.join(self.temp_dir.name, "test_clip.mp4")
        try:
            VideoProcessor.clip_video(self.test_video_path, 0.0, 1.0, clip_path)
            self.assertTrue(os.path.exists(clip_path))

            cap = cv2.VideoCapture(clip_path)
            try:
                self.assertTrue(cap.isOpened())
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.assertEqual(width, 160)
                self.assertEqual(height, 120)
                self.assertEqual(frame_count, 10)
            finally:
                cap.release()
        finally:
            if os.path.exists(clip_path):
                os.remove(clip_path)

    def test_search_engine(self) -> None:
        engine = SigLIPEngine()
        proc = VideoProcessor(target_fps=1.0)
        searcher = VideoSearchEngine(engine=engine, processor=proc)

        indexed = searcher.index_video(self.test_video_path)
        self.assertGreater(indexed, 0)

        results = searcher.search_moments("blue picture", top_k=2)
        self.assertEqual(len(results), min(2, indexed))
        self.assertIsInstance(results[0].score, float)
        self.assertLess(results[0].timestamp, 3.0)
        self.assertLess(results[0].frame_idx, 30)

        blue_results = searcher.search_moments("blue picture", top_k=2, min_gap_sec=0.0)
        self.assertEqual(len(blue_results), 2)
        for res in blue_results:
            self.assertLess(res.timestamp, 3.0)
            self.assertLess(res.frame_idx, 30)

        red_results = searcher.search_moments("red picture", top_k=2, min_gap_sec=0.0)
        self.assertEqual(len(red_results), 2)
        for res in red_results:
            self.assertGreaterEqual(res.timestamp, 3.0)
            self.assertGreaterEqual(res.frame_idx, 30)

        rankings = searcher.classify_video_zero_shot(["blue picture", "red picture"])
        self.assertEqual(len(rankings), 2)
        for rank in rankings:
            self.assertIn("label", rank)
            self.assertIn("mean_score", rank)
            self.assertIn("max_score", rank)
            self.assertIn("mean_prob", rank)
            self.assertIn("max_prob", rank)
            self.assertGreaterEqual(rank["mean_prob"], 0.0)
            self.assertLessEqual(rank["mean_prob"], 1.0)
            self.assertGreaterEqual(rank["max_prob"], 0.0)
            self.assertLessEqual(rank["max_prob"], 1.0)
            self.assertLessEqual(rank["mean_prob"], rank["max_prob"])


if __name__ == "__main__":
    unittest.main()
