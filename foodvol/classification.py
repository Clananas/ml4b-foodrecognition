"""Stage C — food classification.

Wraps a pretrained image classifier (a ViT fine-tuned on Food-101 by default) and
returns the top-k predicted food classes. The predicted class label is the key used
by :mod:`foodvol.nutrition` to look up density and energy/macros, so the label
vocabulary (Food-101 class names, e.g. ``"apple_pie"``, ``"sushi"``) is shared with
the bundled nutrition table.

The model is loaded lazily; if it cannot be downloaded the classifier degrades to a
single ``"unknown"`` prediction so the pipeline keeps running (nutrition then uses a
documented default density/energy).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import cv2
import numpy as np
from PIL import Image

from . import config


@dataclass
class Prediction:
    """A single (label, score) classification result."""

    label: str
    score: float


def _to_pil(image: np.ndarray | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if image.ndim == 2:
        return Image.fromarray(image).convert("RGB")
    # assume BGR (OpenCV) -> RGB
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


class FoodClassifier:
    """Predicts the food class of an image crop. Heavy model is loaded lazily."""

    def __init__(self, model_id: str = config.FOOD_CLASSIFIER_HF, device: Optional[str] = None):
        self.model_id = model_id
        self.device = device or config.get_device()
        self._pipe = None
        self.available = True

    def _ensure_model(self) -> bool:
        if self._pipe is not None:
            return True
        if not self.available:
            return False
        try:
            import torch
            from transformers import pipeline

            # transformers expects an int index or a torch.device.
            device = torch.device(self.device) if self.device in ("mps", "cuda") else -1
            self._pipe = pipeline("image-classification", model=self.model_id, device=device)
            return True
        except Exception as exc:
            print(f"[classification] classifier unavailable ({exc}); predictions will be 'unknown'.")
            self.available = False
            return False

    def classify(self, image: Union[np.ndarray, Image.Image], top_k: int = 3) -> list[Prediction]:
        """Return the top-k predictions (highest score first)."""
        if not self._ensure_model():
            return [Prediction("unknown", 0.0)]
        try:
            preds = self._pipe(_to_pil(image), top_k=top_k)
            return [Prediction(p["label"], float(p["score"])) for p in preds]
        except Exception as exc:
            print(f"[classification] inference failed ({exc}); returning 'unknown'.")
            return [Prediction("unknown", 0.0)]

    def classify_top1(self, image: Union[np.ndarray, Image.Image]) -> Prediction:
        """Convenience wrapper returning only the most likely class."""
        return self.classify(image, top_k=1)[0]
