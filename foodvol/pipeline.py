"""End-to-end orchestration: images + plate diameter -> per-item mass and calories.

Wires the stages together:

    top view  --calibrate--> scale --segment--> items --classify--> class
                                          |                    |
                                          +--> footprint area  +--> density/energy
    side view --calibrate--> scale --segment--> food height
                                          |
              area + height --VolumeEstimator--> volume --x density--> mass --> calories

Height handling: with a side view we measure the dominant food's height and apply it
per item. This is exact for a single dish and an approximation for multi-item plates
with very different heights (documented limitation); the rigorous per-portion
evaluation lives in ``notebooks/00_feasibility.ipynb``. Without a side view, an
optional monocular depth cue or a coarse area-based prior is used instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from . import config, nutrition
from .calibration import Calibration, CalibrationError, calibrate
from .nutrition import NutritionEstimate
from .recognition import FoodRecognizer, Recognition
from .segmentation import FoodSegmenter, InstanceMask
from .volume import VolumeEstimator, measure_height_cm

ImageInput = Union[str, Path, np.ndarray]

MAX_SEGMENTS = 40        # cap how many regions the recogniser scores per image
MAX_HEIGHT_CM = 12.0     # clamp per-item height to a physically plausible range


@dataclass
class ItemEstimate:
    """Per-food-item result."""

    food_class: str
    confidence: float
    area_cm2: float
    height_cm: float
    volume_ml: float
    nutrition: NutritionEstimate
    mask: InstanceMask
    height_source: str = "geometric_prior"   # 'side_view' | 'class_prior' | 'geometric_prior'
    shape_source: str = "trained"            # 'class' | 'trained'
    alternatives: list[tuple[str, float]] = field(default_factory=list)  # CLIP top-k

    @property
    def mass_g(self) -> float:
        return self.nutrition.mass_g


@dataclass
class PlateEstimate:
    """Full result for one plate."""

    items: list[ItemEstimate] = field(default_factory=list)
    calibration_top: Optional[Calibration] = None
    calibration_side: Optional[Calibration] = None
    height_source: str = "none"
    notes: list[str] = field(default_factory=list)

    @property
    def total_mass_g(self) -> float:
        return sum(i.mass_g for i in self.items)

    @property
    def total_kcal(self) -> float:
        return sum(i.nutrition.kcal for i in self.items)

    @property
    def total_protein_g(self) -> float:
        return sum(i.nutrition.protein_g for i in self.items)

    @property
    def total_carbs_g(self) -> float:
        return sum(i.nutrition.carbs_g for i in self.items)

    @property
    def total_fat_g(self) -> float:
        return sum(i.nutrition.fat_g for i in self.items)

    def summary(self) -> str:
        lines = [f"{'Food':<22}{'mass(g)':>10}{'kcal':>9}{'vol(mL)':>10}"]
        lines.append("-" * 51)
        for i in sorted(self.items, key=lambda x: x.nutrition.kcal, reverse=True):
            lines.append(f"{i.food_class:<22}{i.mass_g:>10.0f}{i.nutrition.kcal:>9.0f}{i.volume_ml:>10.0f}")
        lines.append("-" * 51)
        lines.append(f"{'TOTAL':<22}{self.total_mass_g:>10.0f}{self.total_kcal:>9.0f}")
        return "\n".join(lines)


def _load_bgr(image: ImageInput) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return image
    img = cv2.imread(str(image))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image}")
    return img


class FoodVolumePipeline:
    """High-level API. Heavy models are shared and loaded lazily by their stages."""

    def __init__(
        self,
        volume_model_path: Path = config.VOLUME_MODEL_PATH,
        use_depth: bool = False,
        device: Optional[str] = None,
    ):
        self.segmenter = FoodSegmenter(device=device)
        self.recognizer = FoodRecognizer(device=device)
        self.volume = VolumeEstimator.load(volume_model_path)
        self.use_depth = use_depth
        self._depth = None  # created on demand
        self.device = device

    # --- main entry point ------------------------------------------------------
    def estimate(
        self,
        top_image: ImageInput,
        side_image: Optional[ImageInput] = None,
        plate_diameter_cm: float = config.REFERENCE_DIAMETERS_CM["plate_dinner"],
        min_confidence: float = 0.0,
    ) -> PlateEstimate:
        """Estimate per-item mass and calories from a top (and optional side) image."""
        top = _load_bgr(top_image)
        result = PlateEstimate()

        # A. calibrate the top view from the plate (used only for the metric scale).
        try:
            calib_top = calibrate(top, plate_diameter_cm, expect="largest")
        except CalibrationError as exc:
            raise CalibrationError(
                f"{exc} Provide a clearer top-down photo with the whole plate visible."
            ) from exc
        result.calibration_top = calib_top
        interior = calib_top.interior_mask(top.shape)

        # Sanity check: if the "plate" fills the frame, the reference is probably not a
        # real plate -> scale unreliable, and we can't trust the interior to bound food.
        reference_ok = not (interior.mean() > 0.8 or calib_top.diameter_px > 0.95 * max(top.shape[:2]))
        if not reference_ok:
            result.notes.append(
                "The detected reference fills most of the frame — make sure a round plate "
                "of the entered diameter is fully visible. Scale and masses may be wrong."
            )

        # B. height from the side view (or a fallback cue).
        plate_height_cm, calib_side, height_src = self._estimate_height(
            side_image, plate_diameter_cm, top, interior
        )
        result.calibration_side = calib_side
        result.height_source = height_src
        if height_src == "prior":
            result.notes.append("No usable side view: height estimated from a coarse prior.")

        # C. segment everything, then keep only regions CLIP recognises as food. The food
        # gate (not the plate interior) is what separates dishes from background/patterns.
        segments = self.segmenter.segment(
            top, interior_mask=interior if reference_ok else None)[:MAX_SEGMENTS]
        if not segments:
            result.notes.append("No distinct regions detected.")
            return result

        # D. recognise each region and gate out non-food.
        candidates: list[tuple[InstanceMask, "Recognition"]] = []
        n_rejected = 0
        for inst in segments:
            rec = self.recognizer.recognize(inst.crop(top))
            if rec.is_food and rec.score >= min_confidence:
                candidates.append((inst, rec))
            else:
                n_rejected += 1

        # FastSAM emits the same object at several scales (apple, apple+plate, a slice);
        # keep the highest-scoring mask per object and drop the others it overlaps.
        kept = self._suppress_nested(candidates)

        # E. measure area, height, volume, nutrition for each kept item.
        for inst, rec in kept:
            info = nutrition.lookup(rec.label)
            area_cm2 = calib_top.pixel_area_to_cm2(inst.area_px)

            # Height priority: measured side view > class-specific prior > geometric prior.
            if plate_height_cm is not None and plate_height_cm > 0:
                height_cm, height_src = float(plate_height_cm), "side_view"
            elif info.typical_height_cm is not None:
                height_cm, height_src = float(info.typical_height_cm), "class_prior"
            else:
                height_cm, height_src = self._geometric_height_prior(area_cm2), "geometric_prior"
            height_cm = float(np.clip(height_cm, 0.3, MAX_HEIGHT_CM))

            # Volume: class-specific shape factor when known, else the trained regressor.
            if info.shape_factor is not None:
                volume_ml = float(info.shape_factor * area_cm2 * height_cm)
                shape_src = "class"
            else:
                volume_ml = float(self.volume.predict_volume(area_cm2, height_cm))
                shape_src = "trained"

            nut = info.for_volume(volume_ml)
            result.items.append(ItemEstimate(
                food_class=rec.label, confidence=rec.score, area_cm2=area_cm2,
                height_cm=height_cm, volume_ml=volume_ml, nutrition=nut, mask=inst,
                height_source=height_src, shape_source=shape_src,
                alternatives=[(l, s) for l, s in rec.top if l != rec.label][:3],
            ))

        if not result.items:
            result.notes.append(
                f"None of the {len(segments)} detected regions were recognised as food."
            )
        elif n_rejected:
            result.notes.append(f"{n_rejected} non-food region(s) were filtered out.")
        return result

    @staticmethod
    def _suppress_nested(candidates, containment_thresh: float = 0.6):
        """Greedy NMS by containment: keep higher-scoring masks, drop ones they overlap.

        Two masks of the same physical item (e.g. the apple and the apple+plate region)
        overlap heavily even if their IoU is low, so we compare against the smaller mask.
        """
        candidates = sorted(candidates, key=lambda c: c[1].score, reverse=True)
        kept: list[tuple[InstanceMask, "Recognition"]] = []
        for inst, rec in candidates:
            duplicate = False
            for kinst, _ in kept:
                inter = int(np.logical_and(inst.mask, kinst.mask).sum())
                if inter / max(1, min(inst.area_px, kinst.area_px)) > containment_thresh:
                    duplicate = True
                    break
            if not duplicate:
                kept.append((inst, rec))
        return kept

    # --- height helpers --------------------------------------------------------
    def _estimate_height(self, side_image, plate_diameter_cm, top, interior):
        """Return (representative_food_height_cm, side_calibration, source)."""
        if side_image is not None:
            side = _load_bgr(side_image)
            try:
                calib_side = calibrate(side, plate_diameter_cm, expect="largest")
            except CalibrationError:
                calib_side = None
            if calib_side is not None:
                side_interior = calib_side.interior_mask(side.shape)
                food = self.segmenter.segment(side, interior_mask=side_interior)
                if food:
                    tallest = max(food, key=lambda i: i.bbox[3])  # largest vertical extent
                    h_cm = calib_side.pixel_length_to_cm(tallest.bbox[3])
                    return h_cm, calib_side, "side_view"

        # Fallback: optional monocular depth cue on the top view.
        if self.use_depth:
            h = self._depth_height(top, interior)
            if h is not None:
                return h, None, "depth"

        # Last resort: a coarse prior (see _item_height for the per-item version).
        return None, None, "prior"

    def _depth_height(self, top, interior) -> Optional[float]:
        if self._depth is None:
            from .depth import DepthEstimator
            self._depth = DepthEstimator(device=self.device)
        rel = self._depth.relative_depth(top)
        if rel is None or interior.sum() < 50:
            return None
        ring = interior & ~cv2.erode(interior.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)
        base = float(np.median(rel[ring])) if ring.sum() else float(np.median(rel[interior]))
        rise = max(0.0, float(np.percentile(rel[interior], 95)) - base)
        # Empirical: full relative range ~ a few cm of food relief. Coarse by design.
        return float(np.clip(rise * 8.0, 0.5, 8.0))

    @staticmethod
    def _geometric_height_prior(area_cm2: float) -> float:
        """Fallback height (cm) when neither a side view nor a class prior is available.

        Assumes a round-ish object: takes the footprint's equivalent radius and uses a
        fraction of it (clamped to a plausible range). For non-round dishes (pizza,
        soup, …) the class-specific prior in the nutrition table takes precedence.
        """
        radius_cm = float(np.sqrt(max(area_cm2, 1e-3) / np.pi))
        return float(np.clip(0.8 * radius_cm, 0.5, 6.0))
