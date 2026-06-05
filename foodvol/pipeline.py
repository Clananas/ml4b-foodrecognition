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
    scale_source: str = "class_prior"        # 'class_prior' | 'chessboard' | 'reranked'
    mass_source: str = "areal_density"       # 'areal_density' | 'clamped_min' | 'clamped_max'
    cm_per_px: float = 0.0
    typical_mass_g: float = 0.0
    mass_range_g: tuple[float, float] = (0.0, 0.0)
    quantity_confidence: float = 0.0         # 0..1, how trustworthy *the mass* is
    alternatives: list[tuple[str, float]] = field(default_factory=list)  # CLIP top-k

    @property
    def mass_g(self) -> float:
        return self.nutrition.mass_g


@dataclass
class PlateEstimate:
    """Full result for one frame (one or more food items)."""

    items: list[ItemEstimate] = field(default_factory=list)
    height_source: str = "none"           # kept for backward compat with old UI code
    chessboard_scale_cm_per_px: float = 0.0  # >0 if a chessboard was used as scale
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
        chessboard_square_cm: float = 2.0,
    ):
        self.segmenter = FoodSegmenter(device=device)
        self.recognizer = FoodRecognizer(device=device)
        self.volume = VolumeEstimator.load(volume_model_path)
        self.device = device
        self.chessboard_square_cm = float(chessboard_square_cm)

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

        # 0. Opportunistic: detect a chessboard. If found, it gives a real cm/px
        #    independent of the recognised class — far more reliable than
        #    deriving the scale from class priors.
        from .chessboard import detect_scale as _detect_chessboard_scale
        cb = _detect_chessboard_scale(top, square_cm=self.chessboard_square_cm)
        if cb is not None:
            result.chessboard_scale_cm_per_px = cb.cm_per_px
            result.notes.append(
                f"Chessboard detected ({cb.pattern[0]}×{cb.pattern[1]} corners, "
                f"square = {cb.square_cm:.1f} cm) — using it as the metric scale."
            )

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

        # 3 + 4. For each item: self-calibrate, predict mass, sanity-rerank, clamp.
        for inst, rec in kept:
            # Try the top-1 class; if its plausible range can't contain the measurement,
            # re-evaluate the same area against every food candidate in CLIP's top-k
            # and prefer one whose plausible range *does* contain the raw estimate.
            chosen_label, chosen_info, cm_per_px, scale_src, area_cm2, raw_mass, was_reranked \
                = self._pick_class_and_scale(inst, rec, cb)

            info = chosen_info
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

            # Quantity confidence: how trustworthy is *the mass*?
            q_conf = self._quantity_confidence(
                clip_score=rec.score, raw_mass=raw_mass, lo=lo, hi=hi,
                typical=info.typical_mass_g or (lo + hi) / 2 if hi != float("inf") else lo,
                scale_src=scale_src,
                was_reranked=was_reranked,
            )

            item = ItemEstimate(
                food_class=chosen_label, confidence=rec.score, area_cm2=area_cm2,
                height_cm=float("nan"), volume_ml=volume_ml,
                nutrition=info.for_mass(mass_g), mask=inst,
                alternatives=[(l, s) for l, s in rec.top if l != chosen_label][:3],
            )
            item.scale_source = "reranked" if was_reranked else scale_src
            item.mass_source = mass_src
            item.cm_per_px = cm_per_px
            item.typical_mass_g = info.typical_mass_g or 0.0
            item.mass_range_g = (lo, hi if hi != float("inf") else 0.0)
            item.quantity_confidence = q_conf
            result.items.append(item)

            if was_reranked:
                result.notes.append(
                    f"Top-1 class '{rec.label}' didn't fit the measured size; "
                    f"re-ranked to '{chosen_label}' from CLIP's alternatives."
                )
            if mass_src in ("clamped_min", "clamped_max"):
                result.notes.append(
                    f"{chosen_label}: raw estimate {raw_mass:.0f} g was outside the "
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
        """Return (cm/px, source) for a single item, *class-only* fallback.

        Compares the item's bounding-box long side in pixels to the class's
        ``typical_long_cm``. Used when no chessboard is detected.
        """
        long_side_px = max(inst.bbox[2], inst.bbox[3])
        if info.typical_long_cm is not None and long_side_px > 0:
            return float(info.typical_long_cm / long_side_px), "class_prior"
        return float(10.0 / max(long_side_px, 1)), "fallback"

    def _pick_class_and_scale(self, inst, rec, chessboard):
        """Decide on (class, cm/px) using all evidence.

        Strategy:
        1. Compute cm/px. If a chessboard is present, use that scale (real metric);
           else fall back to the recognised class's typical_long_cm.
        2. Compute raw_mass = mass_per_cm2[top-1] × area_cm2.
        3. If raw_mass fits the top-1 plausible [min, max] range → keep top-1.
        4. Otherwise look through CLIP's other top-k candidates. If one of them
           has a plausible range that *contains* the raw_mass (re-scaled to its
           own areal density), pick it. This is what saves a Muffin from being
           called a Blueberry: when the recognised "blueberry" range [1,2]g can
           never explain a 30 cm² object, but "cupcakes" [60,180]g can.
        5. If no candidate fits, keep top-1 and let the clamp + warning fire.
        """
        candidates = [(rec.label, rec.score)] + [(l, s) for l, s in rec.top if l != rec.label]
        candidates = candidates[:5]   # at most 5 candidates

        def evaluate(label):
            info = nutrition.lookup(label)
            if chessboard is not None:
                cm_per_px = chessboard.cm_per_px
                scale_src = "chessboard"
            else:
                cm_per_px, scale_src = self._per_item_scale(inst, info)
            area_cm2 = (cm_per_px ** 2) * inst.area_px
            raw_mass = self._raw_mass(info, area_cm2)
            return info, cm_per_px, scale_src, area_cm2, raw_mass

        # Evaluate top-1 first.
        top_label = candidates[0][0]
        info, cm_per_px, scale_src, area_cm2, raw_mass = evaluate(top_label)
        lo = info.mass_min_g if info.mass_min_g is not None else 0.0
        hi = info.mass_max_g if info.mass_max_g is not None else float("inf")
        if lo <= raw_mass <= hi:
            return top_label, info, cm_per_px, scale_src, area_cm2, raw_mass, False

        # Top-1 fails plausibility. Try alternatives.
        for alt_label, _ in candidates[1:]:
            a_info, a_cm, a_src, a_area, a_mass = evaluate(alt_label)
            a_lo = a_info.mass_min_g if a_info.mass_min_g is not None else 0.0
            a_hi = a_info.mass_max_g if a_info.mass_max_g is not None else float("inf")
            if a_lo <= a_mass <= a_hi:
                return alt_label, a_info, a_cm, a_src, a_area, a_mass, True

        # No candidate fits; stay with top-1 and let the clamp fire.
        return top_label, info, cm_per_px, scale_src, area_cm2, raw_mass, False

    @staticmethod
    def _raw_mass(info, area_cm2: float) -> float:
        """The 'before-clamp' mass estimate for one (class, area)."""
        if info.mass_per_cm2 is not None:
            return float(info.mass_per_cm2 * area_cm2)
        if info.typical_mass_g is not None:
            typical_area = (info.typical_long_cm or 10.0) ** 2 * 0.59
            return float(info.typical_mass_g * (area_cm2 / typical_area))
        return float(info.density_g_per_ml * 100.0 * area_cm2 / 50.0)

    @staticmethod
    def _quantity_confidence(*, clip_score, raw_mass, lo, hi, typical,
                             scale_src, was_reranked):
        """0..1 confidence in the *mass*, not just the class label.

        Combines:
        - CLIP score (how sure is the recogniser about *some* class)
        - Plausibility: did the raw estimate fall inside the plausible range,
          and how close was it to the class's typical value
        - Scale source: a chessboard scale is much more trustworthy than a
          class-prior scale (which is just a guess about typical size)
        - Re-ranking penalty: a re-ranked class is by construction less certain
          than CLIP's top pick
        """
        # Plausibility.
        if hi == float("inf"):
            plaus = 0.5
        elif lo <= raw_mass <= hi:
            band = max(hi - lo, 1e-6)
            dist = abs(raw_mass - typical) / band
            plaus = float(np.clip(1.0 - 0.6 * dist, 0.3, 1.0))
        else:
            plaus = 0.15   # clamped — we know the answer is wrong, just bounded

        scale_factor = {"chessboard": 1.0, "class_prior": 0.7,
                        "reranked": 0.7, "fallback": 0.4}.get(scale_src, 0.5)

        clip_factor = float(np.clip(clip_score, 0.0, 1.0))
        rerank_factor = 0.75 if was_reranked else 1.0
        return float(np.clip(clip_factor * plaus * scale_factor * rerank_factor, 0.0, 1.0))

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
