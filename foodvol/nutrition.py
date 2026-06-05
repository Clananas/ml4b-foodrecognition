"""Stage F — nutrition lookup: class -> density and energy/macros -> mass & calories.

Reads the bundled :data:`~foodvol.config.NUTRITION_DB_PATH` table. The ``density``
column converts an estimated **volume** (mL) into **mass** (g); the per-100 g energy
and macro columns then convert mass into **calories** and **macronutrients**.

Lookups are normalised (lower-case, spaces/hyphens -> underscores) so they tolerate
small label differences, and fall back to a documented generic value for unknown
classes.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from . import config

# Generic cooked-food fallback used when a class is not in the table. Portion priors
# are intentionally None for the default — unknown classes fall back to the trained
# class-agnostic volume regressor + the geometric height prior.
DEFAULT_NUTRITION = ("__default__", 0.90, 200.0, 8.0, 25.0, 8.0)


def _normalise(label: str) -> str:
    return label.strip().lower().replace("-", "_").replace(" ", "_")


@dataclass(frozen=True)
class NutritionInfo:
    """Per-class nutritional + portion reference.

    The portion fields are *priors* used by the volume stage when class-specific
    shape information is known. They are None for classes without an entry, in which
    case the trained class-agnostic regressor and the generic height prior are used.

    Attributes
    ----------
    typical_height_cm : a sensible thickness for a typical serving (e.g. ~1.5 cm for
        pizza, ~3.5 cm for a bowl of soup). Used as the height prior when no side view
        is available.
    shape_factor : the ratio of true volume to the bounding prism ``area × height``.
        Pizza ≈ 0.9 (near a prism), a half-spherical apple ≈ 0.55, a bowl ≈ 1.0.
    """

    food_class: str
    density_g_per_ml: float
    kcal_per_100g: float
    protein_g_per_100g: float
    carbs_g_per_100g: float
    fat_g_per_100g: float
    typical_height_cm: Optional[float] = None
    shape_factor: Optional[float] = None
    is_default: bool = False

    def mass_from_volume(self, volume_ml: float) -> float:
        """Convert a volume in millilitres to mass in grams."""
        return float(volume_ml) * self.density_g_per_ml

    def for_mass(self, mass_g: float) -> "NutritionEstimate":
        """Scale energy and macros to a given mass in grams."""
        f = mass_g / 100.0
        return NutritionEstimate(
            food_class=self.food_class,
            mass_g=mass_g,
            kcal=self.kcal_per_100g * f,
            protein_g=self.protein_g_per_100g * f,
            carbs_g=self.carbs_g_per_100g * f,
            fat_g=self.fat_g_per_100g * f,
            is_default=self.is_default,
        )

    def for_volume(self, volume_ml: float) -> "NutritionEstimate":
        """Convenience: volume -> mass -> energy/macros."""
        return self.for_mass(self.mass_from_volume(volume_ml))


@dataclass(frozen=True)
class NutritionEstimate:
    """Concrete nutrition values for a specific portion."""

    food_class: str
    mass_g: float
    kcal: float
    protein_g: float
    carbs_g: float
    fat_g: float
    is_default: bool = False


def _optional_float(row: dict, key: str) -> Optional[float]:
    """Read an optional numeric column from a CSV row (blank cell -> None)."""
    val = row.get(key, "")
    if val is None or val == "":
        return None
    try:
        return float(val)
    except ValueError:
        return None


@lru_cache(maxsize=1)
def _table() -> dict[str, NutritionInfo]:
    table: dict[str, NutritionInfo] = {}
    with open(config.NUTRITION_DB_PATH, newline="") as fh:
        for row in csv.DictReader(fh):
            name = _normalise(row["class"])
            table[name] = NutritionInfo(
                food_class=name,
                density_g_per_ml=float(row["density_g_per_ml"]),
                kcal_per_100g=float(row["kcal_per_100g"]),
                protein_g_per_100g=float(row["protein_g_per_100g"]),
                carbs_g_per_100g=float(row["carbs_g_per_100g"]),
                fat_g_per_100g=float(row["fat_g_per_100g"]),
                typical_height_cm=_optional_float(row, "typical_height_cm"),
                shape_factor=_optional_float(row, "shape_factor"),
            )
    return table


def lookup(food_class: str) -> NutritionInfo:
    """Return the :class:`NutritionInfo` for a class, or a generic default."""
    info = _table().get(_normalise(food_class))
    if info is not None:
        return info
    _, density, kcal, protein, carbs, fat = DEFAULT_NUTRITION
    return NutritionInfo(food_class=food_class, density_g_per_ml=density,
                         kcal_per_100g=kcal, protein_g_per_100g=protein,
                         carbs_g_per_100g=carbs, fat_g_per_100g=fat, is_default=True)


def known_classes() -> list[str]:
    return sorted(_table().keys())
