"""Tests for the curated per-class portion priors."""
import pytest

from foodvol import nutrition


def test_apple_has_realistic_whole_fruit_prior():
    info = nutrition.lookup("apple")
    assert info.typical_long_cm == pytest.approx(8.0)
    assert 150 < info.typical_mass_g < 220              # a whole apple
    assert info.mass_min_g < info.typical_mass_g < info.mass_max_g


def test_avocado_is_whole_not_sliced():
    """Regression: previously avocado prior came from cafeteria slices (~50 g).
    A whole avocado must be in the right ballpark."""
    info = nutrition.lookup("avocado")
    assert 140 < info.typical_mass_g < 280, "avocado prior must be a whole fruit"
    assert info.mass_per_cm2 is not None and info.mass_per_cm2 > 2.0


def test_half_avocado_is_separate_class():
    half = nutrition.lookup("half_avocado")
    whole = nutrition.lookup("avocado")
    assert half.typical_mass_g < whole.typical_mass_g
    assert 60 < half.typical_mass_g < 150


def test_pizza_slice_vs_whole_pizza():
    sl = nutrition.lookup("pizza_slice")
    wh = nutrition.lookup("pizza_whole")
    assert sl.typical_long_cm < wh.typical_long_cm
    assert sl.typical_mass_g < wh.typical_mass_g
    assert sl.kcal_per_100g == wh.kcal_per_100g       # same food, different portion


def test_sanity_ranges_are_ordered():
    """For every class with a range, min < typical < max."""
    for cls in nutrition.known_classes():
        info = nutrition.lookup(cls)
        if info.typical_mass_g is None: continue
        assert info.mass_min_g <= info.typical_mass_g <= info.mass_max_g, \
            f"{cls}: range out of order"


def test_unknown_class_has_no_prior():
    info = nutrition.lookup("definitely_not_a_food_xyz")
    assert info.is_default
    assert info.typical_mass_g is None
    assert info.mass_per_cm2 is None


def test_table_size_is_reasonable():
    classes = nutrition.known_classes()
    assert 50 <= len(classes) <= 200
    for must_have in ["apple", "avocado", "pizza_slice", "banana", "soup",
                      "hamburger", "egg", "salad", "broccoli"]:
        assert must_have in classes
