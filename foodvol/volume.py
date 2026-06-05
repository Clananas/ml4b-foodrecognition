"""Stage E — volume estimation: footprint area + height -> physical volume.

This is the part we *train*. The geometry gives two metric measurements:

* **footprint area** (cm^2) from the top view — pixels inside the food mask scaled
  by the calibration;
* **height** (cm) from the side view — the food mask's vertical extent scaled by the
  calibration.

A bounding prism has volume ``area * height``; the true volume is some fraction of
that (a "shape factor") which depends on how the food piles up. Rather than guess the
factor, we **learn** ``volume = f(area, height)`` from ECUSTFD's ground-truth volumes
with a small, robust regressor. Crucially the model is **class-agnostic**: volume is
pure geometry, and per-class density/energy are handled separately in
:mod:`foodvol.nutrition`, which helps it generalise to unseen foods.

Upgrade path (GPU): replace the area/height features + linear model with a deep
multi-view network that regresses volume directly from the two images. The
:class:`VolumeEstimator` API (``fit``/``predict_volume``/``save``/``load``) is the
seam to swap in such a model without touching the rest of the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

from . import config
from .calibration import Calibration

if TYPE_CHECKING:  # avoid importing the perception stack at module load
    from .segmentation import FoodSegmenter, InstanceMask


# --- geometric measurements ----------------------------------------------------
@dataclass
class Measurement:
    """A single geometric measurement plus the mask it came from."""

    value: float                      # cm^2 (area) or cm (height)
    mask: Optional["InstanceMask"]
    ok: bool


def measure_footprint_area_cm2(
    image_bgr: np.ndarray,
    food_box: tuple[int, int, int, int],
    calib: Calibration,
    segmenter: "FoodSegmenter",
) -> Measurement:
    """Top-view footprint area in cm^2 (segment the food, scale its pixel area)."""
    inst = segmenter.segment_box(image_bgr, food_box)
    if inst is None:
        return Measurement(float("nan"), None, ok=False)
    return Measurement(calib.pixel_area_to_cm2(inst.area_px), inst, ok=True)


def measure_height_cm(
    image_bgr: np.ndarray,
    food_box: tuple[int, int, int, int],
    calib: Calibration,
    segmenter: "FoodSegmenter",
) -> Measurement:
    """Side-view height in cm (vertical extent of the food mask, scaled)."""
    inst = segmenter.segment_box(image_bgr, food_box)
    if inst is None:
        return Measurement(float("nan"), None, ok=False)
    _, _, _, h_px = inst.bbox
    return Measurement(calib.pixel_length_to_cm(h_px), inst, ok=True)


# --- feature engineering -------------------------------------------------------
FEATURE_NAMES = ("area_cm2", "height_cm", "area_x_height")


def volume_features(area_cm2: float, height_cm: float) -> np.ndarray:
    """Feature vector for the regressor.

    ``area * height`` (a bounding-prism volume) is the physically dominant term;
    ``area`` and ``height`` let the model correct for shapes that scale differently.
    """
    return np.array([area_cm2, height_cm, area_cm2 * height_cm], dtype=np.float64)


def feature_matrix(areas: Sequence[float], heights: Sequence[float]) -> np.ndarray:
    return np.vstack([volume_features(a, h) for a, h in zip(areas, heights)])


# --- estimator -----------------------------------------------------------------
class VolumeEstimator:
    """Predicts food volume (mL) from footprint area and height.

    Before training (or if loading fails) it falls back to the physics estimate
    ``volume = shape_factor * area * height``.
    """

    def __init__(self, shape_factor: float = 0.5):
        self.shape_factor = shape_factor
        self.model = None              # sklearn regressor once fitted (None for 'proportional')
        self.fitted = False            # True once fit() has run (model or learned shape factor)
        self.metrics: dict[str, float] = {}

    # --- training ---
    def fit(
        self,
        areas: Sequence[float],
        heights: Sequence[float],
        volumes_ml: Sequence[float],
        model_kind: str = "huber",
    ) -> "VolumeEstimator":
        """Fit ``volume = f(area, height)`` on measured ground-truth volumes."""
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        X = feature_matrix(areas, heights)
        y = np.asarray(volumes_ml, dtype=np.float64)

        if model_kind == "proportional":
            # Physics with a *learned* shape factor: volume = k * area * height.
            # Single parameter through the origin -> robust and extrapolates cleanly,
            # which matters for an open-world app with portions larger than training.
            ah = X[:, 2]
            self.shape_factor = float(np.sum(ah * y) / np.sum(ah * ah))
            self.model = None
            self.fitted = True
            return self
        if model_kind == "huber":
            from sklearn.linear_model import HuberRegressor
            self.model = make_pipeline(StandardScaler(), HuberRegressor(max_iter=500))
        elif model_kind == "gbr":
            from sklearn.ensemble import HistGradientBoostingRegressor
            self.model = HistGradientBoostingRegressor(max_depth=3, max_iter=200)
        elif model_kind == "linear":
            from sklearn.linear_model import LinearRegression
            self.model = make_pipeline(StandardScaler(), LinearRegression())
        else:
            raise ValueError(f"unknown model_kind: {model_kind!r}")

        self.model.fit(X, y)
        self.fitted = True
        return self

    # --- inference ---
    def predict_volume(self, area_cm2: float, height_cm: float) -> float:
        """Predict volume in mL for a single (area, height) pair."""
        if self.model is not None:
            X = volume_features(area_cm2, height_cm).reshape(1, -1)
            return float(max(0.0, self.model.predict(X)[0]))
        return float(self.shape_factor * area_cm2 * height_cm)

    def predict_many(self, areas: Sequence[float], heights: Sequence[float]) -> np.ndarray:
        if self.model is not None:
            preds = self.model.predict(feature_matrix(areas, heights))
            return np.clip(preds, 0.0, None)
        return np.array([self.shape_factor * a * h for a, h in zip(areas, heights)])

    @property
    def is_trained(self) -> bool:
        return self.fitted

    # --- persistence ---
    def save(self, path: Path = config.VOLUME_MODEL_PATH) -> Path:
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"shape_factor": self.shape_factor, "model": self.model,
                     "fitted": self.fitted, "metrics": self.metrics}, path)
        return path

    @classmethod
    def load(cls, path: Path = config.VOLUME_MODEL_PATH) -> "VolumeEstimator":
        import joblib
        est = cls()
        try:
            blob = joblib.load(path)
            est.shape_factor = blob.get("shape_factor", 0.5)
            est.model = blob.get("model")
            est.fitted = blob.get("fitted", est.model is not None)
            est.metrics = blob.get("metrics", {})
        except Exception as exc:
            print(f"[volume] could not load trained model ({exc}); using physics fallback.")
        return est


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Standard regression error metrics (MAE, RMSE, MAPE, R^2)."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mape = float(np.mean(np.abs(err) / np.clip(np.abs(y_true), 1e-6, None)) * 100.0)
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "MAPE_percent": mape, "R2": r2}
