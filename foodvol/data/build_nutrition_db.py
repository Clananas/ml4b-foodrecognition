"""Generate the bundled nutrition + portion table ``nutrition_db.csv``.

The table is the single source of truth for everything class-specific:

* Nutrition: density (g/mL), energy and macros per 100 g.
* Portion priors: typical_long_cm (the long-side dimension of a typical serving
  in the top-down photo), typical_mass_g (a realistic whole-portion mass for that
  class), and a plausible mass_min_g / mass_max_g range used as a sanity bound.
* mass_per_cm2 derived from typical_mass_g / typical_area, used as the direct
  area-to-mass predictor.

All values are hand-curated to represent **realistic, whole servings as a normal
person would photograph them** (a whole apple, a whole sliced pizza, a whole
avocado — not cafeteria-cut quarters). Where Nutrition5k or ECUSTFD measurements
informed a value, that is noted in the comment.

Half / sliced / wedge variants are deliberately separate classes (e.g.
``half_avocado``, ``pizza_slice``, ``apple_slice``) so CLIP can choose between
them; their priors reflect those serving sizes.

Run::

    python -m foodvol.data.build_nutrition_db
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

# ---------------------------------------------------------------------------
# Nutrition: per 100 g, plus average density (g/mL).
# Sourced from typical food-composition values; replace with USDA/BLS for production.
# ---------------------------------------------------------------------------
NUTRITION: dict[str, tuple[float, float, float, float, float]] = {
    # --- Whole fruit (a normal person photographs the whole fruit) ---
    # Note: we deliberately do NOT add a separate apple_slice/banana_slice class.
    # CLIP confuses small whole apples with slices, leading to bad portions. The
    # whole-fruit class with a clamp to [100, 280] g handles both reasonably.
    "apple":            (0.85, 52,   0.3, 14.0, 0.2),
    "banana":           (0.94, 89,   1.1, 23.0, 0.3),
    "orange":           (0.95, 47,   0.9, 12.0, 0.1),
    "pear":             (1.00, 57,   0.4, 15.0, 0.1),
    "peach":            (0.95, 39,   0.9, 10.0, 0.3),
    "plum":             (1.00, 46,   0.7, 11.0, 0.3),
    "mango":            (1.00, 60,   0.8, 15.0, 0.4),
    "lemon":            (0.90, 29,   1.1,  9.0, 0.3),
    "lime":             (0.90, 30,   0.7,  11.0, 0.2),
    "tomato":           (0.95, 18,   0.9,  3.9, 0.2),
    "kiwi":             (1.00, 61,   1.1, 15.0, 0.5),
    "litchi":           (1.00, 66,   0.8, 17.0, 0.4),
    "avocado":          (1.00, 160,  2.0,  9.0, 15.0),
    "half_avocado":     (1.00, 160,  2.0,  9.0, 15.0),
    "watermelon_wedge": (0.95, 30,   0.6,  8.0, 0.2),
    "pineapple_slice":  (1.00, 50,   0.5, 13.0, 0.1),
    "strawberry":       (0.90, 32,   0.7,  7.7, 0.3),
    "blueberry":        (0.95, 57,   0.7, 14.0, 0.3),
    "raspberry":        (0.80, 52,   1.2, 12.0, 0.7),
    "cherry":           (1.00, 50,   1.0, 12.0, 0.3),
    "grape":            (1.00, 69,   0.7, 18.0, 0.2),
    # --- Vegetables ---
    "carrot":           (1.00, 41,   0.9, 10.0, 0.2),
    "cucumber":         (0.95, 15,   0.7,  3.6, 0.1),
    "broccoli":         (0.55, 34,   2.8,  7.0, 0.4),
    "potato":           (1.05, 87,   2.0, 20.0, 0.1),
    "sweet_potato":     (1.05, 90,   2.0, 21.0, 0.2),
    "corn":             (0.90, 86,   3.2, 19.0, 1.2),
    "bell_pepper":      (0.90, 31,   1.0,  6.0, 0.3),
    "mushroom":         (0.60, 22,   3.1,  3.3, 0.3),
    "salad":            (0.40, 20,   1.5,  3.7, 0.2),
    # --- Eggs / dairy / basics ---
    "egg":              (1.03, 143, 13.0,  1.1, 9.5),
    "fried_egg":        (1.03, 196, 14.0,  0.8, 15.0),
    "boiled_egg":       (1.03, 155, 13.0,  1.1, 11.0),
    "omelette":         (1.00, 154, 11.0,  0.6, 11.7),
    "cheese":           (1.05, 402, 25.0,  1.3, 33.0),
    "yogurt":           (1.03,  61,  3.5,  4.7, 3.3),
    "milk":             (1.03,  60,  3.2,  4.8, 3.3),
    "bread":            (0.30, 265,  9.0, 49.0, 3.2),
    "toast":            (0.30, 290,  9.0, 50.0, 4.0),
    # --- Cooked staples ---
    "white_rice":       (0.85, 130,  2.7, 28.0, 0.3),
    "brown_rice":       (0.85, 110,  2.6, 23.0, 0.9),
    "pasta":            (0.90, 158,  6.0, 31.0, 0.9),
    "oatmeal":          (1.00,  71,  2.5, 12.0, 1.5),
    # --- Meats / fish ---
    "chicken_breast":   (1.05, 165, 31.0,  0.0, 3.6),
    "grilled_chicken":  (1.05, 200, 30.0,  0.0, 8.0),
    "steak":            (1.05, 270, 25.0,  0.0, 19.0),
    "beef":             (1.05, 250, 26.0,  0.0, 15.0),
    "bacon":            (1.00, 540, 37.0,  1.4, 42.0),
    "sausage":          (1.00, 300, 12.0,  2.0, 27.0),
    "grilled_salmon":   (1.00, 210, 23.0,  0.0, 13.0),
    "tuna":             (1.00, 130, 28.0,  0.0,  1.0),
    # --- Dishes & sandwiches ---
    "pizza_slice":      (0.70, 270, 11.0, 33.0, 10.0),
    "pizza_whole":      (0.70, 270, 11.0, 33.0, 10.0),
    "hamburger":        (0.90, 250, 13.0, 28.0, 10.0),
    "sandwich":         (0.60, 250, 12.0, 28.0, 10.0),
    "hot_dog":          (0.80, 290, 11.0, 23.0, 17.0),
    "taco":             (0.80, 220,  9.0, 20.0, 11.0),
    "burrito":          (0.80, 210,  9.0, 22.0,  9.0),
    "sushi":            (1.00, 150,  6.0, 28.0,  2.0),
    "dumplings":        (0.90, 200,  8.0, 25.0,  8.0),
    # --- Bowls (served in a bowl, you photograph from above) ---
    "soup":             (1.00,  50,  2.5,  6.0, 2.0),
    "ramen":            (1.00, 110,  5.0, 14.0, 4.0),
    "pho":              (1.00,  70,  5.0,  9.0, 2.0),
    "fried_rice":       (0.85, 170,  5.0, 24.0, 6.0),
    "pad_thai":         (0.85, 180,  9.0, 22.0, 7.0),
    # --- Snacks & sweets ---
    "ice_cream":        (0.60, 210,  4.0, 24.0, 11.0),
    "chocolate":        (1.10, 546,  4.9, 61.0, 31.0),
    "cookie":           (0.50, 480,  5.0, 64.0, 23.0),
    "donut":            (0.40, 410,  5.0, 47.0, 23.0),
    "muffin":           (0.45, 365,  6.0, 47.0, 17.0),
    "blueberry_muffin": (0.45, 360,  6.0, 50.0, 16.0),
    "cupcake":          (0.50, 370,  4.0, 53.0, 16.0),
    "croissant":        (0.40, 406,  8.0, 45.0, 21.0),
    "pancakes":         (0.60, 230,  6.0, 28.0, 10.0),
    "waffles":          (0.50, 290,  7.0, 33.0, 14.0),
    "french_fries":     (0.50, 310,  3.4, 41.0, 15.0),
}

# ---------------------------------------------------------------------------
# Portion priors: (typical_long_cm, typical_mass_g, mass_min_g, mass_max_g)
# typical_long_cm — the long side of the top-down bounding box for a typical
#                   serving as photographed by a normal user.
# typical_mass_g  — the typical mass of one such serving (whole apple,
#                   one slice of pizza, half avocado, …).
# mass_min/max_g  — a plausible range used as a sanity bound for the estimate.
# ---------------------------------------------------------------------------
PORTIONS: dict[str, tuple[float, float, float, float]] = {
    # --- Whole fruit ---
    "apple":            ( 8.0, 180,  100, 280),
    "banana":           (18.0, 120,   80, 180),
    "orange":           ( 8.0, 180,   90, 280),
    "pear":             ( 9.0, 180,  100, 280),
    "peach":            ( 7.5, 150,   80, 220),
    "plum":             ( 5.5,  70,   40, 110),
    "mango":            (11.0, 250,  150, 400),
    "lemon":            ( 6.5,  85,   50, 130),
    "lime":             ( 5.0,  60,   30,  90),
    "tomato":           ( 7.0, 120,   60, 200),
    "kiwi":             ( 7.0,  75,   40, 120),
    "litchi":           ( 3.5,  20,   10,  35),
    "avocado":          (10.0, 200,  140, 280),
    "half_avocado":     ( 9.0, 100,   60, 150),
    "watermelon_wedge": (18.0, 280,  150, 500),
    "pineapple_slice":  (10.0, 100,   60, 180),
    "strawberry":       ( 4.0,  18,    8,  30),
    "blueberry":        ( 1.2,   1,    1,   2),
    "raspberry":        ( 2.0,   4,    2,   8),
    "cherry":           ( 2.5,   8,    4,  15),
    "grape":            ( 2.0,   6,    3,  12),
    # --- Vegetables ---
    "carrot":           (15.0,  90,   40, 200),
    "cucumber":         (20.0, 200,  100, 400),
    "broccoli":         (12.0, 120,   60, 250),
    "potato":           ( 9.0, 170,   80, 350),
    "sweet_potato":     (14.0, 180,   90, 350),
    "corn":             (18.0, 150,   80, 250),
    "bell_pepper":      ( 9.0, 160,   90, 250),
    "mushroom":         ( 5.0,  20,    8,  60),
    "salad":            (18.0, 100,   40, 250),
    # --- Eggs / dairy ---
    "egg":              ( 5.5,  55,   45,  70),
    "fried_egg":        ( 9.0,  55,   45,  70),
    "boiled_egg":       ( 5.5,  55,   45,  70),
    "omelette":         (16.0, 180,  100, 300),
    "cheese":           ( 8.0,  30,   10,  80),
    "yogurt":           ( 9.0, 150,  100, 250),
    "milk":             ( 7.0, 240,  150, 350),
    "bread":            (12.0,  40,   25,  80),
    "toast":            (10.0,  30,   20,  50),
    # --- Staples (a portion = what fits on a plate) ---
    "white_rice":       (14.0, 180,  100, 350),
    "brown_rice":       (14.0, 180,  100, 350),
    "pasta":            (15.0, 200,  120, 400),
    "oatmeal":          (12.0, 220,  150, 350),
    # --- Meats / fish ---
    "chicken_breast":   (14.0, 180,  100, 280),
    "grilled_chicken":  (14.0, 180,  100, 280),
    "steak":            (14.0, 200,  120, 350),
    "beef":             (12.0, 180,  100, 300),
    "bacon":            (12.0,  30,   10,  80),    # strips, light
    "sausage":          (13.0, 100,   50, 200),
    "grilled_salmon":   (14.0, 180,  100, 280),
    "tuna":             (12.0, 150,   80, 250),
    # --- Dishes ---
    "pizza_slice":      (18.0, 130,   80, 220),
    "pizza_whole":      (28.0, 700,  400,1200),
    "hamburger":        (12.0, 230,  150, 380),
    "sandwich":         (15.0, 200,  120, 350),
    "hot_dog":          (16.0, 130,   80, 200),
    "taco":             (15.0, 150,   80, 250),
    "burrito":          (18.0, 300,  180, 500),
    "sushi":            ( 4.0,  20,   12,  35),
    "dumplings":        (12.0, 150,   80, 300),
    # --- Bowls ---
    "soup":             (16.0, 300,  200, 500),
    "ramen":            (18.0, 450,  300, 700),
    "pho":              (18.0, 450,  300, 700),
    "fried_rice":       (16.0, 250,  150, 450),
    "pad_thai":         (18.0, 280,  180, 500),
    # --- Sweets / snacks ---
    "ice_cream":        (10.0, 100,   60, 200),
    "chocolate":        (12.0,  35,   10,  80),
    "cookie":           ( 6.5,  25,   10,  60),
    "donut":            (10.0,  60,   40,  90),
    "muffin":           ( 8.0,  90,   60, 150),
    "blueberry_muffin": ( 8.0,  90,   60, 150),
    "cupcake":          ( 7.0,  70,   40, 130),
    "croissant":        (14.0,  60,   40, 100),
    "pancakes":         (14.0, 100,   60, 180),
    "waffles":          (14.0, 100,   60, 180),
    "french_fries":     (12.0, 150,   80, 300),
}

HEADER = ["class", "density_g_per_ml", "kcal_per_100g",
          "protein_g_per_100g", "carbs_g_per_100g", "fat_g_per_100g",
          "typical_long_cm", "typical_mass_g", "mass_min_g", "mass_max_g",
          "mass_per_cm2"]


def _mass_per_cm2(typical_long_cm: float, typical_mass_g: float) -> float:
    """Derive areal density from the typical long-side and typical mass.

    Models the food's footprint as an ellipse whose long-side is ``typical_long_cm``
    and short-side ~ 0.75 × long_side (a generic aspect ratio that fits most foods
    well enough; per-class fine-tuning can come later if needed).
    """
    long_side = typical_long_cm
    short_side = 0.75 * long_side
    typical_area = math.pi * (long_side / 2) * (short_side / 2)
    return round(typical_mass_g / typical_area, 3)


def build(out_path: Path | None = None) -> Path:
    out_path = out_path or (Path(__file__).resolve().parent / "nutrition_db.csv")
    with out_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(HEADER)
        for name in sorted(NUTRITION):
            density, kcal, protein, carbs, fat = NUTRITION[name]
            portion = PORTIONS.get(name)
            if portion is None:
                long_cm = mass_g = m_min = m_max = mpc2 = ""
            else:
                long_cm, mass_g, m_min, m_max = portion
                mpc2 = _mass_per_cm2(long_cm, mass_g)
            writer.writerow([name, density, kcal, protein, carbs, fat,
                             long_cm, mass_g, m_min, m_max, mpc2])
    print(f"Wrote {len(NUTRITION)} classes to {out_path} "
          f"({len(PORTIONS)} with full portion priors).")
    return out_path


if __name__ == "__main__":
    build()
