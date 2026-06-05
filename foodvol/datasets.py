"""ECUSTFD dataset access: ground truth, bounding boxes and top/side pairing.

Provides typed access to the dataset downloaded by ``data/download_ecustfd.py`` so
the feasibility notebook and tests can stay short. Nothing here depends on the
heavy perception models.

Unit note: ``density.xls`` labels its volume column ``volume(mm^3)``, but the values
are actually in **millilitres (cm^3)** — e.g. apple001 = 310 mL at 244.5 g gives a
density of ~0.79 g/mL, which is physically correct for an apple. We expose the
column as ``volume_ml``.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Optional

from . import config

# Images flagged by the dataset authors as missing the coin (unusable for scale).
EXCLUDED_STEMS = {"mix002T(2)", "mix005S(4)"}

COIN_DIAMETER_CM = config.REFERENCE_DIAMETERS_CM["coin_ecustfd"]  # 2.5 cm

_NAME_RE = re.compile(r"^(?P<base>.+?)(?P<view>[ST])\((?P<n>\d+)\)$")


@dataclass
class BBox:
    name: str
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def box(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def diameter_px(self) -> float:
        """Un-foreshortened diameter estimate for a circular object (max side)."""
        return float(max(self.width, self.height))

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


@dataclass
class Portion:
    """One food portion with its ground truth and its top/side image paths."""

    portion_id: str          # e.g. "apple001"
    food_type: str           # e.g. "apple"
    volume_ml: float
    weight_g: float
    top_images: list[Path] = field(default_factory=list)
    side_images: list[Path] = field(default_factory=list)

    @property
    def density_g_per_ml(self) -> float:
        return self.weight_g / self.volume_ml if self.volume_ml else float("nan")


class ECUSTFD:
    """Typed reader for a downloaded ECUSTFD dataset directory."""

    def __init__(self, root: Path = config.ECUSTFD_DIR):
        self.root = Path(root)
        self.images_dir = self.root / "JPEGImages"
        self.annotations_dir = self.root / "Annotations"
        self.density_path = self.root / "density.xls"

    def is_available(self) -> bool:
        return self.density_path.exists() and self.images_dir.is_dir()

    # --- ground truth ----------------------------------------------------------
    @lru_cache(maxsize=1)
    def ground_truth(self) -> dict[str, list[tuple[str, float, float]]]:
        """Map portion id -> list of (food_type, volume_ml, weight_g).

        Single-food portions have a one-element list; ``mix*`` portions have two.
        """
        import xlrd  # reads the legacy .xls directly (pandas 3 rejects xlrd 1.2)

        wb = xlrd.open_workbook(str(self.density_path))
        gt: dict[str, list[tuple[str, float, float]]] = {}
        for sheet_name in wb.sheet_names():
            sh = wb.sheet_by_name(sheet_name)
            for r in range(1, sh.nrows):  # row 0 is the header
                pid = str(sh.cell_value(r, 0)).strip()
                ftype = str(sh.cell_value(r, 1)).strip()
                try:
                    volume_ml = float(sh.cell_value(r, 2))
                    weight_g = float(sh.cell_value(r, 3))
                except (TypeError, ValueError):
                    continue
                if not pid:
                    continue
                gt.setdefault(pid, []).append((ftype, volume_ml, weight_g))
        return gt

    # --- annotations -----------------------------------------------------------
    def annotation(self, image_stem: str) -> list[BBox]:
        """Parse the VOC XML for an image stem (e.g. 'apple001T(1)')."""
        xml_path = self.annotations_dir / f"{image_stem}.xml"
        if not xml_path.exists():
            return []
        root = ET.parse(xml_path).getroot()
        boxes: list[BBox] = []
        for obj in root.findall("object"):
            name = obj.findtext("name", default="").strip()
            bb = obj.find("bndbox")
            if bb is None:
                continue
            boxes.append(BBox(
                name,
                int(float(bb.findtext("xmin", "0"))),
                int(float(bb.findtext("ymin", "0"))),
                int(float(bb.findtext("xmax", "0"))),
                int(float(bb.findtext("ymax", "0"))),
            ))
        return boxes

    def coin_box(self, image_stem: str) -> Optional[BBox]:
        for b in self.annotation(image_stem):
            if b.name == "coin":
                return b
        return None

    def food_boxes(self, image_stem: str) -> list[BBox]:
        return [b for b in self.annotation(image_stem) if b.name != "coin"]

    # --- portions --------------------------------------------------------------
    def portions(self, single_food_only: bool = True) -> list[Portion]:
        """Group images into portions with both a top and a side view available."""
        gt = self.ground_truth()
        tops: dict[str, list[Path]] = {}
        sides: dict[str, list[Path]] = {}
        for img in sorted(self.images_dir.glob("*.JPG")):
            if img.stem in EXCLUDED_STEMS:
                continue
            m = _NAME_RE.match(img.stem)
            if not m:
                continue
            base, view = m.group("base"), m.group("view")
            (tops if view == "T" else sides).setdefault(base, []).append(img)

        portions: list[Portion] = []
        for pid, entries in gt.items():
            if single_food_only and len(entries) != 1:
                continue
            if pid not in tops or pid not in sides:
                continue
            ftype, volume_ml, weight_g = entries[0]
            portions.append(Portion(
                portion_id=pid, food_type=ftype, volume_ml=volume_ml, weight_g=weight_g,
                top_images=tops[pid], side_images=sides[pid],
            ))
        portions.sort(key=lambda p: p.portion_id)
        return portions

    def iter_portions(self, single_food_only: bool = True) -> Iterator[Portion]:
        yield from self.portions(single_food_only=single_food_only)
