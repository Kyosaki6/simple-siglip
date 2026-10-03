import unittest
import numpy as np
import cv2
import os
import torch
from src.video_processor import VideoProcessor
from src.siglip_engine import SigLIPEngine
from src.video_search import VideoSearchEngine


class TestSigLIPVideoPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_video_path = "tests/test_dummy.mp4"
        fps = 10
        total_frames = 20
        width, height = 160, 120
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(cls.test_video_path, fourcc, fps, (width, height))

        for i in range(total_frames):
            color = (255, 0, 0) if i < 10 else (0, 0, 255)
            frame = np.full((height, width, 3), color, dtype=np.uint8)
            out.write(frame)
        out.release()

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_video_path):
            os.remove(cls.test_video_path)

    def test_video_processor(self):
        proc = VideoProcessor(target_fps=1.0)
        frames = proc.extract_frames(self.test_video_path)
        self.assertGreater(len(frames), 0)
        self.assertIn("image", frames[0])
        self.assertIn("timestamp", frames[0])

        img = VideoProcessor.extract_frame_at_timestamp(self.test_video_path, 0.5)
        self.assertIsNotNone(img)

        clip_path = "tests/test_clip.mp4"
        VideoProcessor.clip_video(self.test_video_path, 0.0, 1.0, clip_path)
        self.assertTrue(os.path.exists(clip_path))
        if os.path.exists(clip_path):
            os.remove(clip_path)

    def test_search_engine(self):
        engine = SigLIPEngine()
        proc = VideoProcessor(target_fps=1.0)
        searcher = VideoSearchEngine(engine=engine, processor=proc)

        indexed = searcher.index_video(self.test_video_path)
        self.assertGreater(indexed, 0)

        results = searcher.search_moments("blue picture", top_k=2)
        self.assertEqual(len(results), min(2, indexed))
        self.assertIsInstance(results[0].score, float)

        rankings = searcher.classify_video_zero_shot(["blue picture", "red picture"])
        self.assertEqual(len(rankings), 2)


if __name__ == "__main__":
    unittest.main()
