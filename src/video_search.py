from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import torch
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
    ) -> None:
        self.engine = engine or SigLIPEngine()
        self.processor = processor or VideoProcessor(target_fps=1.0)

        self.frame_embeddings: Optional[torch.Tensor] = None
        self.frame_metadata: List[Dict[str, Any]] = []

    def index_video(self, video_path: str, batch_size: int = 16) -> int:
        """
        Extract frames in batches, compute SigLIP embeddings, and add to index.

        Returns:
            Number of indexed frames.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        batch_images: List[Image.Image] = []
        batch_metadata: List[Dict[str, Any]] = []
        new_embeds_list: List[torch.Tensor] = []
        new_metadata: List[Dict[str, Any]] = []
        count = 0

        for item in self.processor.iter_frames(video_path):
            batch_images.append(item["image"])
            batch_metadata.append(
                {
                    "timestamp": item["timestamp"],
                    "frame_idx": item["frame_idx"],
                    "video_path": video_path,
                }
            )
            count += 1
            if len(batch_images) >= batch_size:
                embeds = self.engine.encode_images(batch_images)
                new_embeds_list.append(embeds)
                new_metadata.extend(batch_metadata)
                batch_images = []
                batch_metadata = []

        if batch_images:
            embeds = self.engine.encode_images(batch_images)
            new_embeds_list.append(embeds)
            new_metadata.extend(batch_metadata)
            batch_images = []
            batch_metadata = []

        if count == 0:
            return 0

        combined_new_embeds = torch.cat(new_embeds_list, dim=0)

        self.frame_metadata.extend(new_metadata)
        if self.frame_embeddings is None:
            self.frame_embeddings = combined_new_embeds
        else:
            self.frame_embeddings = torch.cat([self.frame_embeddings, combined_new_embeds], dim=0)

        return count

    def search_moments(
        self, query: str, top_k: int = 5, min_gap_sec: float = 2.0
    ) -> List[SearchResult]:
        """
        Search for specific timestamps in indexed videos matching a natural language query.

        Args:
            query: Natural language search string.
            top_k: Maximum number of results to return.
            min_gap_sec: Minimum temporal gap in seconds between returned moments from the same video.
                         Set to 0.0 to disable suppression.

        Returns:
            List of SearchResult objects sorted by highest relevance.
        """
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty or whitespace.")

        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")

        if min_gap_sec < 0.0:
            raise ValueError(f"min_gap_sec must be non-negative, got {min_gap_sec}")

        if self.frame_embeddings is None or len(self.frame_metadata) == 0:
            raise RuntimeError("Index is empty. Call index_video() first.")

        clean_query = query.strip()
        text_embed = self.engine.encode_texts([clean_query])  # Shape: (1, D)
        # Cosine similarity because embeddings are L2 normalized
        similarities = (self.frame_embeddings @ text_embed.T).view(-1)  # Shape: (N,)

        sorted_indices = torch.argsort(similarities, descending=True).tolist()

        results: List[SearchResult] = []
        for idx in sorted_indices:
            score = float(similarities[idx].item())
            meta = self.frame_metadata[idx]
            vpath = meta["video_path"]
            ts = meta["timestamp"]

            if min_gap_sec > 0.0:
                if any(
                    res.video_path == vpath and abs(ts - res.timestamp) < min_gap_sec
                    for res in results
                ):
                    continue

            results.append(
                SearchResult(
                    timestamp=meta["timestamp"],
                    frame_idx=meta["frame_idx"],
                    score=round(score, 4),
                    video_path=vpath,
                )
            )
            if len(results) >= top_k:
                break

        return results

    def classify_video_zero_shot(
        self,
        candidate_labels: List[str],
        template: Optional[str] = "This is a photo of {}.",
    ) -> List[Dict[str, Any]]:
        """
        Perform zero-shot video classification by aggregating frame similarities.

        Args:
            candidate_labels: List of label strings.
            template: Optional prompt template string with "{}" placeholder.

        Returns:
            List of dicts {"label": str, "mean_score": float, "max_score": float,
                           "mean_prob": float, "max_prob": float} sorted by mean_prob descending.
        """
        if not candidate_labels:
            raise ValueError("Candidate labels cannot be empty")

        if self.frame_embeddings is None or len(self.frame_metadata) == 0:
            raise RuntimeError("Index is empty. Call index_video() first.")

        if template is not None:
            if "{}" not in template:
                raise ValueError(f"template must contain a '{{}}' placeholder, got {template!r}")
            # A label that already contains "{}" is treated as a ready-made prompt and kept verbatim.
            formatted_labels = [
                label if "{}" in label else template.replace("{}", label)
                for label in candidate_labels
            ]
        else:
            formatted_labels = candidate_labels

        text_embeds = self.engine.encode_texts(formatted_labels)
        sim_matrix, prob_matrix = self.engine.score_logits(
            self.frame_embeddings, text_embeds
        )

        mean_scores = sim_matrix.mean(dim=0).tolist()
        max_scores = sim_matrix.max(dim=0).values.tolist()
        mean_probs = prob_matrix.mean(dim=0).tolist()
        max_probs = prob_matrix.max(dim=0).values.tolist()

        # Sort on unrounded values: sigmoid probs sit near 0, so rounded ties are common.
        order = sorted(
            range(len(candidate_labels)),
            key=lambda i: (mean_probs[i], mean_scores[i]),
            reverse=True,
        )
        return [
            {
                "label": candidate_labels[i],
                "mean_score": round(float(mean_scores[i]), 4),
                "max_score": round(float(max_scores[i]), 4),
                "mean_prob": round(float(mean_probs[i]), 4),
                "max_prob": round(float(max_probs[i]), 4),
            }
            for i in order
        ]
