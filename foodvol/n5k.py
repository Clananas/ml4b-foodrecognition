"""Nutrition5k subset reader.

Single-ingredient standard-food dishes from the Nutrition5k dataset
(Thames et al., CVPR 2021). Each portion has a top-down RGB photo and gold-
standard per-dish nutrition values measured on a scale.

The download lives at ``data/Nutrition5k/overhead_rgb/<dish_id>.png`` and the
companion manifest at ``data/n5k_meta/n5k_subset_manifest.csv``. The cafeteria
class names are mapped onto our nutrition-table vocabulary (e.g.
``"cheese pizza"`` → ``"pizza"``, ``"sweet potato"`` → ``"sweet_potato"``).

See ``data/download_nutrition5k.py`` for licence and citation.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Optional

from . import config

DATA_DIR = config.DATA_DIR / "Nutrition5k"
IMAGE_DIR = DATA_DIR / "overhead_rgb"
MANIFEST_PATH = config.DATA_DIR / "n5k_meta" / "n5k_subset_manifest.csv"

# Map the raw cafeteria ingredient names to keys of our nutrition table.
# Entries here are deliberately conservative; everything not listed maps to
# its lowercase, underscored form (e.g. "carrot" → "carrot").
_CLASS_MAP = {
    "cheese pizza":     "pizza",
    "white rice":       "white_rice",
    "brown rice":       "white_rice",   # nutrition_db lacks brown_rice — same prior
    "wild rice":        "white_rice",
    "sweet potato":     "sweet_potato",
    "grilled chicken":  "chicken_breast",
    "chicken":          "chicken_breast",
    "chicken breast":   "chicken_breast",
    "fried egg":        "egg",
    "boiled egg":       "egg",
    "bell pepper":      "bell_pepper",
    "french fries":     "french_fries",
    "ice cream":        "ice_cream",
    "lettuce":          "salad",
    "noodles":          "pasta",
    "yoghurt":          "yogurt",
    "ham":              "bacon",
}


def map_to_nutrition_class(raw_name: str) -> str:
    """Translate a Nutrition5k ingredient name into our nutrition-table vocabulary."""
    name = raw_name.strip().lower()
    return _CLASS_MAP.get(name, name.replace(" ", "_").replace("-", "_"))


@dataclass
class N5kPortion:
    """One Nutrition5k single-ingredient portion."""

    dish_id: str
    n5k_class: str           # raw Nutrition5k label (e.g. "cheese pizza")
    food_class: str          # our nutrition-table key (e.g. "pizza")
    image_path: Path
    total_mass_g: float
    total_kcal: float
    total_fat_g: float
    total_carb_g: float
    total_protein_g: float

    @property
    def density_g_per_ml(self) -> float:
        """Nutrition5k doesn't measure volume — return NaN to keep the schema."""
        return float("nan")


class Nutrition5k:
    """Typed reader for the downloaded subset."""

    def __init__(self, manifest_path: Path = MANIFEST_PATH, image_dir: Path = IMAGE_DIR):
        self.manifest_path = Path(manifest_path)
        self.image_dir = Path(image_dir)

    def is_available(self) -> bool:
        return (self.manifest_path.exists()
                and self.image_dir.is_dir()
                and len(list(self.image_dir.glob("*.png"))) >= 50)

    @lru_cache(maxsize=1)
    def portions(self) -> list[N5kPortion]:
        if not self.manifest_path.exists():
            return []
        out: list[N5kPortion] = []
        with open(self.manifest_path, newline="") as f:
            for row in csv.DictReader(f):
                img = self.image_dir / f"{row['dish_id']}.png"
                if not img.exists():
                    continue
                out.append(N5kPortion(
                    dish_id=row["dish_id"],
                    n5k_class=row["n5k_class"],
                    food_class=map_to_nutrition_class(row["n5k_class"]),
                    image_path=img,
                    total_mass_g=float(row["total_mass_g"]),
                    total_kcal=float(row["total_kcal"]),
                    total_fat_g=float(row["total_fat_g"]),
                    total_carb_g=float(row["total_carb_g"]),
                    total_protein_g=float(row["total_protein_g"]),
                ))
        return out

    def iter_portions(self) -> Iterator[N5kPortion]:
        yield from self.portions()

    def by_class(self) -> dict[str, list[N5kPortion]]:
        out: dict[str, list[N5kPortion]] = {}
        for p in self.portions():
            out.setdefault(p.food_class, []).append(p)
        return out
