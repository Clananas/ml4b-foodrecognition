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
from .nutrition import NutritionEstimate
from .recognition import FoodRecognizer, Recognition
from .segmentation import FoodSegmenter, InstanceMask
from .volume import VolumeEstimator

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
    scale_source: str = "class_prior"        # 'class_prior' | 'fallback'
    mass_source: str = "areal_density"       # 'areal_density' | 'clamped_min' | 'clamped_max'
    cm_per_px: float = 0.0
    typical_mass_g: float = 0.0
    mass_range_g: tuple[float, float] = (0.0, 0.0)
    alternatives: list[tuple[str, float]] = field(default_factory=list)  # CLIP top-k

    @property
    def mass_g(self) -> float:
        return self.nutrition.mass_g


@dataclass
class PlateEstimate:
    """Full result for one frame (one or more food items)."""

    items: list[ItemEstimate] = field(default_factory=list)
    height_source: str = "none"           # kept for backward compat with old UI code
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
        device: Optional[str] = None,
    ):
        self.segmenter = FoodSegmenter(device=device)
        self.recognizer = FoodRecognizer(device=device)
        self.volume = VolumeEstimator.load(volume_model_path)
        self.device = device

    # --- main entry point ------------------------------------------------------
    def estimate(
        self,
        top_image: ImageInput,
        side_image: Optional[ImageInput] = None,
        min_confidence: float = 0.0,
    ) -> PlateEstimate:
        """Estimate per-item mass and calories — no metric reference required.

        The pipeline runs in this order:
          1. Segment the image into region candidates (FastSAM).
          2. Recognise each candidate (CLIP) and drop non-food regions.
          3. **Self-calibrate per item**: convert pixels to centimetres using the
             recognised class's ``typical_long_cm`` from the nutrition table.
          4. Apply class-specific shape factor + height (or the side view) to
             compute volume → mass → calories & macros.

        ``side_image`` is no longer used for plate calibration; it is still
        accepted for compatibility but currently ignored.
        """
        del side_image  # accepted for backward compatibility, not used yet
        top = _load_bgr(top_image)
        result = PlateEstimate()
        result.height_source = "class_prior"

        # 1. Segment the whole frame; no plate-interior restriction.
        segments = self.segmenter.segment(top, interior_mask=None)[:MAX_SEGMENTS]
        if not segments:
            result.notes.append("No distinct regions detected.")
            return result

        # 2. Recognise each candidate; keep food only.
        candidates: list[tuple[InstanceMask, "Recognition"]] = []
        n_rejected = 0
        for inst in segments:
            rec = self.recognizer.recognize(inst.crop(top))
            if rec.is_food and rec.score >= min_confidence:
                candidates.append((inst, rec))
            else:
                n_rejected += 1

        # FastSAM emits the same object at several scales; collapse overlapping ones.
        kept = self._suppress_nested(candidates)

        # 3 + 4. For each item: self-calibrate, predict mass, clamp to a sanity range.
        for inst, rec in kept:
            info = nutrition.lookup(rec.label)
            cm_per_px, scale_src = self._per_item_scale(inst, info)
            area_cm2 = (cm_per_px ** 2) * inst.area_px

            # Direct area→mass via the per-class areal density. This *is* the model.
            if info.mass_per_cm2 is not None:
                raw_mass = float(info.mass_per_cm2 * area_cm2)
            elif info.typical_mass_g is not None:
                # No areal density learned yet — fall back to the typical mass and
                # scale by how much the measured area deviates from the typical area.
                typical_area = (info.typical_long_cm or 10.0) ** 2 * 0.59  # ≈ ellipse area at 0.75 aspect
                raw_mass = float(info.typical_mass_g * (area_cm2 / typical_area))
            else:
                raw_mass = float(info.density_g_per_ml * 100.0 * area_cm2 / 50.0)  # weak default

            # Clamp to the realistic whole-serving range and remember whether we did.
            lo = info.mass_min_g if info.mass_min_g is not None else 0.0
            hi = info.mass_max_g if info.mass_max_g is not None else float("inf")
            if raw_mass < lo:
                mass_g, mass_src = lo, "clamped_min"
            elif raw_mass > hi:
                mass_g, mass_src = hi, "clamped_max"
            else:
                mass_g, mass_src = raw_mass, "areal_density"

            volume_ml = (float(mass_g / info.density_g_per_ml)
                         if info.density_g_per_ml else float("nan"))

            item = ItemEstimate(
                food_class=rec.label, confidence=rec.score, area_cm2=area_cm2,
                height_cm=float("nan"), volume_ml=volume_ml,
                nutrition=info.for_mass(mass_g), mask=inst,
                alternatives=[(l, s) for l, s in rec.top if l != rec.label][:3],
            )
            item.scale_source = scale_src
            item.mass_source = mass_src
            item.cm_per_px = cm_per_px
            item.typical_mass_g = info.typical_mass_g or 0.0
            item.mass_range_g = (lo, hi if hi != float("inf") else 0.0)
            result.items.append(item)
            if mass_src in ("clamped_min", "clamped_max"):
                result.notes.append(
                    f"{rec.label}: raw estimate {raw_mass:.0f} g was outside the "
                    f"plausible range ({lo:.0f}-{hi:.0f} g); clamped to {mass_g:.0f} g."
                )

        if not result.items:
            result.notes.append(
                f"None of the {len(segments)} detected regions were recognised as food."
            )
        elif n_rejected:
            result.notes.append(f"{n_rejected} non-food region(s) were filtered out.")
        if any(it.scale_source == "fallback" for it in result.items):
            result.notes.append(
                "Some items had no typical_long_cm in the nutrition table — their masses "
                "use a generic fallback scale and may be off."
            )
        return result

    @staticmethod
    def _per_item_scale(inst: "InstanceMask", info) -> tuple[float, str]:
        """Return (cm/px, source) for a single item.

        Compares the item's bounding-box long side in pixels to the class's
        ``typical_long_cm``. Falls back to a coarse default if the class has no
        size entry (the resulting mass should then be treated as a guess).
        """
        long_side_px = max(inst.bbox[2], inst.bbox[3])  # max(width, height) in px
        if info.typical_long_cm is not None and long_side_px > 0:
            return float(info.typical_long_cm / long_side_px), "class_prior"
        # Fallback: assume the item is roughly 10 cm long. Very rough.
        return float(10.0 / max(long_side_px, 1)), "fallback"

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

    @staticmethod
    def _geometric_height_prior(area_cm2: float) -> float:
        """Fallback height (cm) when neither a side view nor a class prior is available.

        Assumes a round-ish object: takes the footprint's equivalent radius and uses a
        fraction of it (clamped to a plausible range). For non-round dishes (pizza,
        soup, …) the class-specific prior in the nutrition table takes precedence.
        """
        radius_cm = float(np.sqrt(max(area_cm2, 1e-3) / np.pi))
        return float(np.clip(0.8 * radius_cm, 0.5, 6.0))
