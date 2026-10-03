from typing import List, Optional, Union
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
    ):
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
            if hasattr(features, "pooler_output") and features.pooler_output is not None:
                features = features.pooler_output
            elif not isinstance(features, torch.Tensor):
                features = features[0]
            features = F.normalize(features, p=2, dim=-1)
            all_embeds.append(features.cpu())

        if not all_embeds:
            return torch.empty((0, self.model.config.vision_config.hidden_size))

        return torch.cat(all_embeds, dim=0)

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """
        Encode text queries into normalized feature vectors.

        Returns:
            Tensor of shape (M, feature_dim), L2-normalized.
        """
        inputs = self.processor(
            text=texts, padding="max_length", return_tensors="pt"
        ).to(device=self.device)
        features = self.model.get_text_features(**inputs)
        if hasattr(features, "pooler_output") and features.pooler_output is not None:
            features = features.pooler_output
        elif not isinstance(features, torch.Tensor):
            features = features[0]
        features = F.normalize(features, p=2, dim=-1)
        return features.cpu()

    @torch.no_grad()
    def compute_zero_shot_logits(
        self, images: List[Image.Image], candidate_labels: List[str]
    ) -> torch.Tensor:
        """
        Compute SigLIP zero-shot probabilities using SigLIP's learned temperature and bias.
        SigLIP uses sigmoid activation instead of CLIP's softmax:
            P(image, text) = sigmoid(scale * (image @ text.T) + bias)

        Returns:
            Tensor of shape (num_images, num_labels) with sigmoid probabilities in [0, 1].
        """
        inputs = self.processor(
            text=candidate_labels,
            images=images,
            padding="max_length",
            return_tensors="pt",
        ).to(device=self.device)

        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = torch.sigmoid(logits_per_image)
        return probs.cpu()
