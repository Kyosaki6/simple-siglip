from typing import Any, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor


class SigLIPEngine:
    """Wrapper around HuggingFace SigLIP model for vision and text feature extraction."""

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.dtype = dtype if dtype is not None else (
            torch.float16 if self.device.type == "cuda" else torch.float32
        )

        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            dtype=self.dtype,
        ).to(self.device)
        self.model.eval()

    @staticmethod
    def _pool_features(features: Union[torch.Tensor, Any]) -> torch.Tensor:
        """Extract pooled feature tensor from model output."""
        if hasattr(features, "pooler_output") and features.pooler_output is not None:
            features = features.pooler_output
        elif not isinstance(features, torch.Tensor):
            features = features[0]
        return features

    @torch.no_grad()
    def encode_images(
        self, images: List[Image.Image], batch_size: int = 16
    ) -> torch.Tensor:
        """
        Encode batch of PIL images into normalized feature vectors.

        Returns:
            Tensor of shape (N, feature_dim), L2-normalized.
        """
        all_embeds = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(
                device=self.device, dtype=self.dtype
            )
            features = self.model.get_image_features(**inputs)
            features = self._pool_features(features)
            features = F.normalize(features.float(), p=2, dim=-1)
            all_embeds.append(features.cpu())

        if not all_embeds:
            return torch.empty(
                (0, self.model.config.vision_config.hidden_size), dtype=torch.float32
            )

        return torch.cat(all_embeds, dim=0)

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """
        Encode text queries into normalized feature vectors.

        Returns:
            Tensor of shape (M, feature_dim), L2-normalized.
        """
        if not texts:
            return torch.empty(
                (0, self.model.config.text_config.hidden_size), dtype=torch.float32
            )

        inputs = self.processor(
            text=texts,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        ).to(device=self.device)
        features = self.model.get_text_features(**inputs)
        features = self._pool_features(features)
        features = F.normalize(features.float(), p=2, dim=-1)
        return features.cpu()

    @torch.no_grad()
    def score_logits(
        self, image_embeds: torch.Tensor, text_embeds: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cosine similarities and calibrated SigLIP sigmoid probabilities.

        Args:
            image_embeds: Tensor of shape (N, D), L2-normalized image embeddings.
            text_embeds: Tensor of shape (M, D), L2-normalized text embeddings.

        Returns:
            Tuple of:
                similarities: Tensor of shape (N, M), cosine similarity in float32.
                probabilities: Tensor of shape (N, M), sigmoid probability in [0, 1] in float32.
        """
        target_device = image_embeds.device
        img = image_embeds.to(device=target_device, dtype=torch.float32)
        txt = text_embeds.to(device=target_device, dtype=torch.float32)
        similarities = img @ txt.T

        scale = self.model.logit_scale.to(device=target_device, dtype=torch.float32).exp()
        bias = self.model.logit_bias.to(device=target_device, dtype=torch.float32)

        probabilities = torch.sigmoid(scale * similarities + bias)
        return similarities, probabilities
