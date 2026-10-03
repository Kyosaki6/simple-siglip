from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.siglip_engine import SigLIPEngine
from src.video_processor import VideoProcessor


@dataclass
class SearchResult:
    timestamp: float
    frame_idx: int
    score: float
    video_path: str


class VideoSearchEngine:
    """Index video frames and retrieve moments matching natural language queries using SigLIP."""

    def __init__(
        self,
        engine: Optional[SigLIPEngine] = None,
        processor: Optional[VideoProcessor] = None,
    ):
        self.engine = engine or SigLIPEngine()
        self.processor = processor or VideoProcessor(target_fps=1.0)

        self.frame_embeddings: Optional[torch.Tensor] = None
        self.frame_metadata: List[Dict[str, Any]] = []

    def index_video(self, video_path: str) -> int:
        """
        Extract frames, compute SigLIP embeddings, and add to index.

        Returns:
            Number of indexed frames.
        """
        extracted = self.processor.extract_frames(video_path)
        if not extracted:
            return 0

        images = [item["image"] for item in extracted]
        embeds = self.engine.encode_images(images)

        for item in extracted:
            self.frame_metadata.append(
                {
                    "timestamp": item["timestamp"],
                    "frame_idx": item["frame_idx"],
                    "video_path": video_path,
                }
            )

        if self.frame_embeddings is None:
            self.frame_embeddings = embeds
        else:
            self.frame_embeddings = torch.cat([self.frame_embeddings, embeds], dim=0)

        return len(extracted)

    def search_moments(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Search for specific timestamps in indexed videos matching a natural language query.

        Returns:
            List of SearchResult objects sorted by highest relevance.
        """
        if self.frame_embeddings is None or len(self.frame_metadata) == 0:
            raise RuntimeError("Index is empty. Call index_video() first.")

        text_embed = self.engine.encode_texts([query])  # Shape: (1, D)
        # Cosine similarity because embeddings are L2 normalized
        similarities = (self.frame_embeddings @ text_embed.T).squeeze(-1)  # Shape: (N,)

        k = min(top_k, len(similarities))
        top_scores, top_indices = torch.topk(similarities, k=k)

        results = []
        for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            meta = self.frame_metadata[idx]
            results.append(
                SearchResult(
                    timestamp=meta["timestamp"],
                    frame_idx=meta["frame_idx"],
                    score=round(float(score), 4),
                    video_path=meta["video_path"],
                )
            )

        return results

    def classify_video_zero_shot(
        self, candidate_labels: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Perform zero-shot video classification by aggregating frame similarities.

        Returns:
            List of dicts {"label": str, "mean_score": float, "max_score": float} sorted descending.
        """
        if self.frame_embeddings is None or len(self.frame_metadata) == 0:
            raise RuntimeError("Index is empty. Call index_video() first.")

        text_embeds = self.engine.encode_texts(candidate_labels)  # Shape: (num_labels, D)
        # Similarity matrix: (num_frames, num_labels)
        sim_matrix = self.frame_embeddings @ text_embeds.T

        mean_scores = sim_matrix.mean(dim=0).tolist()
        max_scores = sim_matrix.max(dim=0).values.tolist()

        rankings = []
        for label, mean_s, max_s in zip(candidate_labels, mean_scores, max_scores):
            rankings.append(
                {
                    "label": label,
                    "mean_score": round(float(mean_s), 4),
                    "max_score": round(float(max_s), 4),
                }
            )

        rankings.sort(key=lambda x: x["mean_score"], reverse=True)
        return rankings
