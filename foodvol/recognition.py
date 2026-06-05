"""Open-vocabulary food recognition with a non-food gate (CLIP zero-shot).

The earlier classifier had two problems that made the app fail on real photos:

1. It could only output one of the 101 Food-101 *dishes*, so a plain apple was
   mislabelled, and it had **no way to say "this isn't food"** — every checkerboard
   square or placemat patch became some dish.
2. That meant background patterns were segmented and labelled as food.

This module fixes both with **CLIP zero-shot classification**. CLIP scores an image
against arbitrary text labels, so we score each candidate region against:

* a broad **food vocabulary** (every class in the nutrition table, fruits included), and
* a set of **non-food sentinels** ("a checkerboard pattern", "a plate", "a table"…).

If a non-food sentinel wins, the region is rejected. This both **recognises real
dishes and fruit** and **gates out non-food**, which is exactly what the segmentation
stage needs. CLIP is a strong pretrained internet model; we only train the *portion*
(volume) model ourselves.

Text features for the fixed label set are computed once and cached, so recognising a
region is a single image encode plus a matrix multiply — fast enough for many regions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import cv2
import numpy as np
from PIL import Image

from . import config, nutrition

# Non-food labels used purely as a gate. If one of these wins, the region is not food.
NONFOOD_LABELS = [
    "a checkerboard pattern", "a polka dot mat", "a calibration board",
    "an empty plate", "a plate", "a bowl", "a table surface", "a placemat",
    "a fork", "a knife", "a spoon", "cutlery", "a napkin", "a hand",
    "fabric", "a wall", "the floor", "plain background",
]

HYPOTHESIS = "a photo of {}."


@dataclass
class Recognition:
    """Result of recognising one region."""

    label: str          # nutrition-table key (e.g. "apple", "apple_pie"); "unknown" if gated
    score: float        # probability of the winning label
    is_food: bool
    top: list[tuple[str, float]]  # top-k (label, score) for transparency


def _to_pil(image: Union[np.ndarray, Image.Image]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if image.ndim == 2:
        return Image.fromarray(image).convert("RGB")
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


class FoodRecognizer:
    """CLIP zero-shot recogniser with a non-food gate. Model loaded lazily."""

    def __init__(self, model_id: str = config.CLIP_MODEL_HF, device: Optional[str] = None):
        self.model_id = model_id
        self.device = device or config.get_device()
        self._model = None
        self._processor = None
        self._text_features = None      # (N, D) normalised
        self._labels: list[str] = []     # nutrition keys aligned with text features
        self._is_food: np.ndarray = np.array([])
        self.available = True
        self._fallback = None            # FoodClassifier if CLIP unavailable

    # --- label set -------------------------------------------------------------
    def _build_labels(self) -> tuple[list[str], list[str], np.ndarray]:
        """Return (prompt_texts, nutrition_keys, is_food_mask)."""
        food_keys = nutrition.known_classes()
        food_texts = [k.replace("_", " ") for k in food_keys]
        keys = food_keys + NONFOOD_LABELS
        texts = food_texts + NONFOOD_LABELS
        is_food = np.array([True] * len(food_keys) + [False] * len(NONFOOD_LABELS))
        return texts, keys, is_food

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        if not self.available:
            return False
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            self._model = CLIPModel.from_pretrained(self.model_id).to(self.device).eval()
            self._processor = CLIPProcessor.from_pretrained(self.model_id)

            texts, keys, is_food = self._build_labels()
            with torch.no_grad():
                prompts = [HYPOTHESIS.format(t) for t in texts]
                inputs = self._processor(text=prompts, return_tensors="pt", padding=True).to(self.device)
                feats = self._embeds(self._model.get_text_features(**inputs))
                feats = feats / feats.norm(dim=-1, keepdim=True)
            self._labels = keys
            self._is_food = is_food
            self._text_features = feats
            return True
        except Exception as exc:
            print(f"[recognition] CLIP unavailable ({exc}); falling back to the Food-101 classifier "
                  "(no non-food gate).")
            self._model = self._processor = self._text_features = None
            self.available = False
            return False

    @staticmethod
    def _embeds(out):
        """Return joint-space embeddings, tolerating transformers API drift.

        Older transformers return the projected embedding tensor directly; newer ones
        return a base output whose ``pooler_output`` already holds the projected
        joint-space embedding.
        """
        import torch
        if torch.is_tensor(out):
            return out
        return out.pooler_output

    # --- inference -------------------------------------------------------------
    def recognize(self, image: Union[np.ndarray, Image.Image], top_k: int = 5) -> Recognition:
        if not self._ensure_model():
            return self._fallback_recognize(image)
        import torch

        pil = _to_pil(image)
        with torch.no_grad():
            inputs = self._processor(images=pil, return_tensors="pt").to(self.device)
            feat = self._embeds(self._model.get_image_features(**inputs))
            feat = feat / feat.norm(dim=-1, keepdim=True)
            logit_scale = self._model.logit_scale.exp()
            logits = (logit_scale * feat @ self._text_features.t()).squeeze(0)
            probs = logits.softmax(dim=-1).detach().cpu().numpy()

        order = np.argsort(probs)[::-1]
        top = [(self._labels[i], float(probs[i])) for i in order[:top_k]]
        best = int(order[0])
        return Recognition(
            label=self._labels[best] if self._is_food[best] else "unknown",
            score=float(probs[best]),
            is_food=bool(self._is_food[best]),
            top=top,
        )

    def _fallback_recognize(self, image) -> Recognition:
        """Without CLIP, use the supervised Food-101 classifier (cannot gate non-food)."""
        if self._fallback is None:
            from .classification import FoodClassifier
            self._fallback = FoodClassifier(device=self.device)
        pred = self._fallback.classify_top1(image)
        return Recognition(label=pred.label, score=pred.score, is_food=True,
                           top=[(pred.label, pred.score)])
