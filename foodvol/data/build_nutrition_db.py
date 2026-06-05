"""Generate the bundled nutrition table ``nutrition_db.csv``.

The table maps a food class to an approximate **density** (g/mL) and **energy &
macronutrient** content (per 100 g). It covers the 101 Food-101 classes predicted by
the default classifier plus the 19 ECUSTFD classes used in the feasibility study.

Values are deliberately *approximate*, sourced from typical food-composition figures,
and intended as a sensible default. For production use, replace them with an
authoritative source such as USDA FoodData Central or the German
Bundeslebensmittelschluessel (BLS); the CSV schema is the integration point.

Regenerate with::

    python -m foodvol.data.build_nutrition_db
"""
from __future__ import annotations

import csv
from pathlib import Path

# class -> (density_g_per_ml, kcal_per_100g, protein_g, carbs_g, fat_g)  [per 100 g]
NUTRITION: dict[str, tuple[float, float, float, float, float]] = {
    # --- Food-101 ---
    "apple_pie": (0.70, 265, 2.4, 37.0, 12.5),
    "baby_back_ribs": (1.00, 290, 21.0, 0.0, 22.0),
    "baklava": (0.80, 430, 6.0, 45.0, 25.0),
    "beef_carpaccio": (1.00, 190, 21.0, 1.0, 11.0),
    "beef_tartare": (1.00, 190, 20.0, 2.0, 11.0),
    "beet_salad": (0.60, 90, 2.0, 12.0, 4.0),
    "beignets": (0.40, 380, 6.0, 45.0, 19.0),
    "bibimbap": (0.80, 130, 6.0, 18.0, 4.0),
    "bread_pudding": (0.70, 260, 6.0, 38.0, 9.0),
    "breakfast_burrito": (0.80, 210, 9.0, 22.0, 9.0),
    "bruschetta": (0.50, 200, 5.0, 28.0, 7.0),
    "caesar_salad": (0.50, 180, 6.0, 8.0, 14.0),
    "cannoli": (0.70, 380, 7.0, 40.0, 21.0),
    "caprese_salad": (0.70, 190, 11.0, 5.0, 14.0),
    "carrot_cake": (0.60, 360, 4.0, 47.0, 18.0),
    "ceviche": (1.00, 110, 17.0, 4.0, 2.0),
    "cheesecake": (0.90, 320, 6.0, 26.0, 22.0),
    "cheese_plate": (1.05, 380, 23.0, 3.0, 30.0),
    "chicken_curry": (1.00, 150, 12.0, 6.0, 9.0),
    "chicken_quesadilla": (0.90, 280, 16.0, 22.0, 14.0),
    "chicken_wings": (1.00, 250, 24.0, 2.0, 16.0),
    "chocolate_cake": (0.70, 370, 5.0, 50.0, 16.0),
    "chocolate_mousse": (0.60, 230, 4.0, 22.0, 14.0),
    "churros": (0.50, 360, 4.0, 43.0, 19.0),
    "clam_chowder": (1.00, 90, 4.0, 9.0, 4.0),
    "club_sandwich": (0.60, 250, 14.0, 22.0, 12.0),
    "crab_cakes": (0.90, 230, 14.0, 12.0, 14.0),
    "creme_brulee": (0.90, 300, 4.0, 25.0, 21.0),
    "croque_madame": (0.80, 280, 15.0, 20.0, 16.0),
    "cup_cakes": (0.50, 370, 4.0, 53.0, 16.0),
    "deviled_eggs": (0.90, 200, 9.0, 1.0, 17.0),
    "donuts": (0.40, 410, 5.0, 47.0, 23.0),
    "dumplings": (0.90, 200, 8.0, 25.0, 8.0),
    "edamame": (0.70, 120, 11.0, 10.0, 5.0),
    "eggs_benedict": (0.90, 230, 12.0, 12.0, 15.0),
    "escargots": (1.00, 180, 14.0, 3.0, 13.0),
    "falafel": (0.70, 330, 13.0, 32.0, 18.0),
    "filet_mignon": (1.05, 270, 25.0, 0.0, 19.0),
    "fish_and_chips": (0.60, 230, 12.0, 23.0, 11.0),
    "foie_gras": (1.00, 460, 11.0, 5.0, 44.0),
    "french_fries": (0.50, 310, 3.4, 41.0, 15.0),
    "french_onion_soup": (1.00, 60, 3.0, 6.0, 3.0),
    "french_toast": (0.60, 230, 8.0, 25.0, 11.0),
    "fried_calamari": (0.70, 280, 15.0, 22.0, 14.0),
    "fried_rice": (0.85, 170, 5.0, 24.0, 6.0),
    "frozen_yogurt": (0.70, 160, 4.0, 25.0, 4.0),
    "garlic_bread": (0.40, 350, 8.0, 40.0, 17.0),
    "gnocchi": (0.90, 150, 4.0, 30.0, 2.0),
    "greek_salad": (0.60, 130, 3.0, 7.0, 10.0),
    "grilled_cheese_sandwich": (0.60, 350, 13.0, 30.0, 20.0),
    "grilled_salmon": (1.00, 210, 23.0, 0.0, 13.0),
    "guacamole": (0.95, 160, 2.0, 9.0, 14.0),
    "gyoza": (0.90, 210, 8.0, 24.0, 9.0),
    "hamburger": (0.90, 250, 13.0, 28.0, 10.0),
    "hot_and_sour_soup": (1.00, 50, 3.0, 6.0, 2.0),
    "hot_dog": (0.80, 290, 11.0, 23.0, 17.0),
    "huevos_rancheros": (0.90, 160, 8.0, 12.0, 9.0),
    "hummus": (1.00, 230, 8.0, 20.0, 14.0),
    "ice_cream": (0.60, 210, 4.0, 24.0, 11.0),
    "lasagna": (0.95, 150, 9.0, 13.0, 7.0),
    "lobster_bisque": (1.00, 100, 5.0, 7.0, 6.0),
    "lobster_roll_sandwich": (0.70, 220, 13.0, 20.0, 10.0),
    "macaroni_and_cheese": (0.90, 190, 8.0, 20.0, 9.0),
    "macarons": (0.70, 400, 7.0, 60.0, 16.0),
    "miso_soup": (1.00, 40, 3.0, 4.0, 1.0),
    "mussels": (1.00, 170, 24.0, 7.0, 4.0),
    "nachos": (0.40, 340, 9.0, 36.0, 18.0),
    "omelette": (0.80, 160, 11.0, 2.0, 12.0),
    "onion_rings": (0.40, 330, 4.0, 38.0, 18.0),
    "oysters": (1.00, 80, 9.0, 5.0, 2.0),
    "pad_thai": (0.85, 180, 9.0, 22.0, 7.0),
    "paella": (0.85, 160, 8.0, 20.0, 5.0),
    "pancakes": (0.60, 230, 6.0, 28.0, 10.0),
    "panna_cotta": (0.90, 250, 4.0, 20.0, 17.0),
    "peking_duck": (1.00, 340, 19.0, 5.0, 28.0),
    "pho": (1.00, 70, 5.0, 9.0, 2.0),
    "pizza": (0.70, 270, 11.0, 33.0, 10.0),
    "pork_chop": (1.05, 230, 26.0, 0.0, 14.0),
    "poutine": (0.70, 230, 6.0, 25.0, 12.0),
    "prime_rib": (1.05, 340, 22.0, 0.0, 28.0),
    "pulled_pork_sandwich": (0.80, 250, 16.0, 24.0, 10.0),
    "ramen": (1.00, 110, 5.0, 14.0, 4.0),
    "ravioli": (0.95, 170, 7.0, 24.0, 5.0),
    "red_velvet_cake": (0.70, 370, 4.0, 50.0, 18.0),
    "risotto": (0.90, 170, 4.0, 25.0, 6.0),
    "samosa": (0.70, 310, 5.0, 32.0, 18.0),
    "sashimi": (1.00, 130, 22.0, 0.0, 4.0),
    "scallops": (1.00, 110, 20.0, 3.0, 1.0),
    "seaweed_salad": (0.60, 70, 1.0, 9.0, 3.0),
    "shrimp_and_grits": (0.90, 170, 10.0, 16.0, 7.0),
    "spaghetti_bolognese": (0.90, 150, 7.0, 18.0, 5.0),
    "spaghetti_carbonara": (0.90, 200, 8.0, 22.0, 9.0),
    "spring_rolls": (0.60, 220, 5.0, 28.0, 10.0),
    "steak": (1.05, 270, 25.0, 0.0, 19.0),
    "strawberry_shortcake": (0.60, 290, 3.0, 42.0, 13.0),
    "sushi": (1.00, 150, 6.0, 28.0, 2.0),
    "tacos": (0.80, 220, 9.0, 20.0, 11.0),
    "takoyaki": (0.80, 190, 8.0, 20.0, 9.0),
    "tiramisu": (0.80, 280, 5.0, 30.0, 16.0),
    "tuna_tartare": (1.00, 150, 22.0, 2.0, 6.0),
    "waffles": (0.50, 290, 7.0, 33.0, 14.0),
    # --- ECUSTFD classes (fruit / snacks) ---
    "apple": (0.85, 52, 0.3, 14.0, 0.2),
    "banana": (0.94, 89, 1.1, 23.0, 0.3),
    "bread": (0.30, 265, 9.0, 49.0, 3.2),
    "bun": (0.35, 290, 8.0, 52.0, 5.0),
    "doughnut": (0.40, 410, 5.0, 47.0, 23.0),
    "egg": (1.03, 143, 13.0, 1.1, 9.5),
    "fired_dough_twist": (0.50, 430, 7.0, 50.0, 22.0),
    "grape": (1.00, 69, 0.7, 18.0, 0.2),
    "lemon": (0.90, 29, 1.1, 9.0, 0.3),
    "litchi": (1.00, 66, 0.8, 17.0, 0.4),
    "mango": (1.00, 60, 0.8, 15.0, 0.4),
    "mooncake": (1.10, 420, 6.0, 60.0, 17.0),
    "orange": (0.95, 47, 0.9, 12.0, 0.1),
    "peach": (0.95, 39, 0.9, 10.0, 0.3),
    "pear": (1.00, 57, 0.4, 15.0, 0.1),
    "plum": (1.00, 46, 0.7, 11.0, 0.3),
    "qiwi": (1.00, 61, 1.1, 15.0, 0.5),
    "sachima": (0.40, 460, 6.0, 60.0, 22.0),
    "tomato": (0.95, 18, 0.9, 3.9, 0.2),
    # --- Common everyday foods (fruit / vegetables / basics) for broader recognition ---
    "strawberry": (0.90, 32, 0.7, 7.7, 0.3),
    "blueberry": (0.95, 57, 0.7, 14.0, 0.3),
    "raspberry": (0.80, 52, 1.2, 12.0, 0.7),
    "watermelon": (0.95, 30, 0.6, 8.0, 0.2),
    "pineapple": (1.00, 50, 0.5, 13.0, 0.1),
    "kiwi": (1.00, 61, 1.1, 15.0, 0.5),
    "cherry": (1.00, 50, 1.0, 12.0, 0.3),
    "carrot": (1.00, 41, 0.9, 10.0, 0.2),
    "broccoli": (0.55, 34, 2.8, 7.0, 0.4),
    "cucumber": (0.95, 15, 0.7, 3.6, 0.1),
    "potato": (1.00, 87, 2.0, 20.0, 0.1),
    "sweet_potato": (1.00, 90, 2.0, 21.0, 0.2),
    "white_rice": (0.85, 130, 2.7, 28.0, 0.3),
    "pasta": (0.90, 158, 6.0, 31.0, 0.9),
    "chicken_breast": (1.05, 165, 31.0, 0.0, 3.6),
    "beef": (1.05, 250, 26.0, 0.0, 15.0),
    "salad": (0.40, 20, 1.5, 3.7, 0.2),
    "cheese": (1.05, 402, 25.0, 1.3, 33.0),
    "yogurt": (1.03, 61, 3.5, 4.7, 3.3),
    "milk": (1.03, 60, 3.2, 4.8, 3.3),
    "avocado": (0.95, 160, 2.0, 9.0, 15.0),
    "corn": (0.90, 86, 3.2, 19.0, 1.2),
    "peas": (0.80, 81, 5.4, 14.0, 0.4),
    "bell_pepper": (0.90, 31, 1.0, 6.0, 0.3),
    "mushroom": (0.60, 22, 3.1, 3.3, 0.3),
    "bacon": (1.00, 540, 37.0, 1.4, 42.0),
    "sausage": (1.00, 300, 12.0, 2.0, 27.0),
    "soup": (1.00, 50, 2.5, 6.0, 2.0),
    "oatmeal": (1.00, 71, 2.5, 12.0, 1.5),
    "chocolate": (1.10, 546, 4.9, 61.0, 31.0),
    "cookie": (0.50, 480, 5.0, 64.0, 23.0),
    "toast": (0.30, 290, 9.0, 50.0, 4.0),
}

HEADER = ["class", "density_g_per_ml", "kcal_per_100g",
          "protein_g_per_100g", "carbs_g_per_100g", "fat_g_per_100g",
          "typical_height_cm", "shape_factor"]


# Per-class portion priors used when no side view is available and as the shape factor
# in volume = shape_factor * area * height. Organised by category to avoid repetition;
# a class without an entry falls back to the trained class-agnostic regressor.
#
# Picking values:
#   typical_height_cm — how thick a serving usually is, measured top-to-bottom
#   shape_factor      — ratio of true volume to its bounding prism (area * height);
#                       1.0 for a perfect prism, ~0.5 for a half-sphere

_ROUND_FRUIT = [
    "apple", "orange", "peach", "mango", "plum", "lemon", "tomato", "pear",
    "qiwi", "kiwi", "litchi", "watermelon", "pineapple",
]
_BERRIES = ["strawberry", "blueberry", "raspberry", "cherry", "grape"]
_ELONGATED = ["banana", "cucumber", "carrot", "corn", "bell_pepper"]

_FLAT_DISHES = [          # roughly a thin prism: area × thin height × ~prism
    "pizza", "pancakes", "waffles", "tacos", "french_toast",
    "garlic_bread", "bruschetta", "toast", "spring_rolls",
]
_SLICE_DESSERTS = [       # wedge cut from a cake: still ~prism
    "apple_pie", "cheesecake", "chocolate_cake", "carrot_cake",
    "red_velvet_cake", "tiramisu", "bread_pudding", "strawberry_shortcake",
]
_STACKED = [              # tall, vertically built items
    "hamburger", "club_sandwich", "breakfast_burrito", "pulled_pork_sandwich",
    "hot_dog", "lobster_roll_sandwich", "grilled_cheese_sandwich",
    "chicken_quesadilla",
]
_BOWL_DISHES = [          # served in a bowl, depth ≈ bowl depth
    "soup", "ramen", "pho", "miso_soup", "lobster_bisque", "clam_chowder",
    "french_onion_soup", "hot_and_sour_soup", "oatmeal", "yogurt",
    "ice_cream", "frozen_yogurt", "panna_cotta", "chocolate_mousse",
    "creme_brulee", "bibimbap", "pad_thai", "fried_rice", "paella",
    "risotto", "macaroni_and_cheese", "shrimp_and_grits",
]
_PASTA_PILE = [           # tangled / piled, irregular packing
    "spaghetti_bolognese", "spaghetti_carbonara", "gnocchi", "ravioli", "pasta",
    "fish_and_chips", "french_fries", "nachos", "onion_rings", "chicken_wings",
    "fried_calamari", "samosa", "dumplings", "gyoza", "edamame",
]
_MEAT_BLOCK = [           # roughly a flat block (steak, fish fillet)
    "steak", "filet_mignon", "pork_chop", "baby_back_ribs", "prime_rib",
    "grilled_salmon", "sashimi", "chicken_breast", "beef", "tuna_tartare",
]
_FLAT_SALAD = ["caesar_salad", "greek_salad", "caprese_salad", "beet_salad",
               "seaweed_salad", "ceviche", "salad", "guacamole", "hummus"]

_SPECIFIC = {             # one-off overrides where the category default isn't right
    "sushi": (2.5, 0.9),       # bite-sized prisms
    "donuts": (2.5, 0.7),      # torus, ~half a prism's volume
    "doughnut": (2.5, 0.7),
    "bun": (4.0, 0.7),
    "egg": (4.5, 0.6),         # hard-boiled standing up; for fried egg side view helps
    "omelette": (1.5, 0.9),
    "deviled_eggs": (2.5, 0.6),
    "mooncake": (3.0, 0.85),
    "macarons": (2.0, 0.85),
    "cookie": (1.2, 0.9),
    "chocolate": (1.0, 1.0),   # bar
    "cheese_plate": (1.5, 0.9),
    "cheese": (1.5, 0.9),
    "bread": (4.0, 0.7),
    "bacon": (0.5, 0.9),       # strips
    "sausage": (2.5, 0.7),     # cylindrical
    "fired_dough_twist": (3.0, 0.5),
    "sachima": (2.5, 0.45),    # porous Chinese sponge snack
    "banana": (2.5, 0.55),     # elongated, lies flat on the plate
    "potato": (5.0, 0.55),
    "sweet_potato": (5.0, 0.55),
    "broccoli": (4.0, 0.45),   # florets are very porous
    "mushroom": (3.0, 0.5),
    "avocado": (4.5, 0.55),
    "white_rice": (3.0, 0.7),
    "milk": (5.0, 1.0),        # glass
    "peas": (2.0, 0.6),
    "blueberry": (1.0, 0.6),   # tiny
    "raspberry": (1.5, 0.5),
    "cherry": (2.0, 0.6),
    "strawberry": (3.0, 0.55),
    "grape": (1.8, 0.6),
}

PORTION_PRIORS: dict[str, tuple[float, float]] = {}
for c in _ROUND_FRUIT:     PORTION_PRIORS[c] = (6.5, 0.55)
for c in _BERRIES:         PORTION_PRIORS[c] = (2.0, 0.55)
for c in _ELONGATED:       PORTION_PRIORS[c] = (3.0, 0.6)
for c in _FLAT_DISHES:     PORTION_PRIORS[c] = (1.5, 0.9)
for c in _SLICE_DESSERTS:  PORTION_PRIORS[c] = (4.5, 0.9)
for c in _STACKED:         PORTION_PRIORS[c] = (5.5, 0.75)
for c in _BOWL_DISHES:     PORTION_PRIORS[c] = (3.5, 1.0)
for c in _PASTA_PILE:      PORTION_PRIORS[c] = (3.0, 0.6)
for c in _MEAT_BLOCK:      PORTION_PRIORS[c] = (2.0, 0.85)
for c in _FLAT_SALAD:      PORTION_PRIORS[c] = (3.0, 0.5)
PORTION_PRIORS.update(_SPECIFIC)


def build(out_path: Path | None = None) -> Path:
    out_path = out_path or (Path(__file__).resolve().parent / "nutrition_db.csv")
    with out_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(HEADER)
        for name in sorted(NUTRITION):
            density, kcal, protein, carbs, fat = NUTRITION[name]
            h, k = PORTION_PRIORS.get(name, (None, None))
            writer.writerow([name, density, kcal, protein, carbs, fat,
                             "" if h is None else h, "" if k is None else k])
    print(f"Wrote {len(NUTRITION)} entries to {out_path} "
          f"({len(PORTION_PRIORS)} with portion priors)")
    return out_path


if __name__ == "__main__":
    build()
