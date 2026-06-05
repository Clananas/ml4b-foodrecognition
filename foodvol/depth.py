"""Stage D (optional) — monocular relative depth.

Wraps Depth Anything V2 (small) to produce a single-image depth map. The output is
**relative** (affine-invariant inverse depth): larger values are nearer, and it is
*not* metric on its own. We use it only as an auxiliary **height cue** — for example
to estimate food height from a top-down view when no dedicated side view is provided.

When a metric side view *is* available, the geometric height from
:mod:`foodvol.volume` is preferred and this module is not needed. Depth is therefore
strictly optional and loaded lazily.
"""
from __future__ import annotations

from typing import Optional, Union

import cv2
import numpy as np
from PIL import Image

from . import config


def _to_pil(image: np.ndarray | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if image.ndim == 2:
        return Image.fromarray(image).convert("RGB")
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


class DepthEstimator:
    """Estimates a relative depth map for an image. Heavy model is loaded lazily."""

    def __init__(self, model_id: str = config.DEPTH_MODEL_HF, device: Optional[str] = None):
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

            device = torch.device(self.device) if self.device in ("mps", "cuda") else -1
            self._pipe = pipeline("depth-estimation", model=self.model_id, device=device)
            return True
        except Exception as exc:
            print(f"[depth] depth model unavailable ({exc}); height will use geometry/priors only.")
            self.available = False
            return False

    def relative_depth(self, image: Union[np.ndarray, Image.Image]) -> Optional[np.ndarray]:
        """Return a float32 relative-depth map (H, W), normalised to [0, 1], or None.

        Values near 1 are closer to the camera; values near 0 are farther. The map is
        resized to the input resolution.
        """
        if not self._ensure_model():
            return None
        try:
            pil = _to_pil(image)
            out = self._pipe(pil)
            depth = out["predicted_depth"]
            depth = depth.squeeze().detach().cpu().numpy().astype(np.float32)
            depth = cv2.resize(depth, (pil.width, pil.height), interpolation=cv2.INTER_CUBIC)
            dmin, dmax = float(depth.min()), float(depth.max())
            if dmax - dmin < 1e-6:
                return np.zeros_like(depth)
            return (depth - dmin) / (dmax - dmin)
        except Exception as exc:
            print(f"[depth] inference failed ({exc}).")
            return None

    def height_above_plane(
        self,
        image: Union[np.ndarray, Image.Image],
        plate_mask: np.ndarray,
        food_mask: np.ndarray,
    ) -> Optional[float]:
        """Rough relative height of food above the plate surface, in [0, 1].

        Compares the median relative depth of the food region against the plate ring.
        This is a *relative* cue (not centimetres); :mod:`foodvol.volume` calibrates it
        against ground-truth data. Returns None if depth is unavailable.
        """
        depth = self.relative_depth(image)
        if depth is None:
            return None
        plate_ring = plate_mask & ~food_mask
        if plate_ring.sum() < 20 or food_mask.sum() < 20:
            return None
        plate_level = float(np.median(depth[plate_ring]))
        food_level = float(np.median(depth[food_mask]))
        return max(0.0, food_level - plate_level)   # nearer (higher) food => positive
