"""Tests for class-aware portion priors (height + shape factor)."""
import pytest

from foodvol import nutrition


def test_pizza_has_flat_prior():
    info = nutrition.lookup("pizza")
    assert info.typical_height_cm == pytest.approx(1.5)
    assert info.shape_factor == pytest.approx(0.9)


def test_apple_has_round_prior():
    info = nutrition.lookup("apple")
    assert 5.0 < info.typical_height_cm < 8.0   # apple is a few cm tall
    assert 0.4 < info.shape_factor < 0.7        # roughly hemispherical


def test_soup_uses_bowl_depth():
    info = nutrition.lookup("soup")
    assert info.typical_height_cm == pytest.approx(3.5)
    assert info.shape_factor == pytest.approx(1.0)   # a bowl is essentially a cylinder


def test_unknown_class_has_no_portion_prior():
    info = nutrition.lookup("definitely_not_a_food_xyz")
    assert info.is_default
    assert info.typical_height_cm is None
    assert info.shape_factor is None


def test_pizza_volume_against_old_pipeline():
    """The user's pizza screenshot: area 129 cm², no side view.

    The old pipeline produced an absurd 235 g / 636 kcal from a 5.1 cm prior height
    and a fruit shape factor (k=0.51). With class-aware priors the same area should
    land in the slice-of-pizza range (~100–150 g, ~270–400 kcal).
    """
    info = nutrition.lookup("pizza")
    A = 129.0  # cm²
    volume_ml = info.shape_factor * A * info.typical_height_cm
    mass_g = volume_ml * info.density_g_per_ml
    kcal = mass_g * info.kcal_per_100g / 100.0
    assert 100 <= mass_g <= 180, f"pizza mass out of range: {mass_g:.0f} g"
    assert 250 <= kcal <= 500, f"pizza kcal out of range: {kcal:.0f}"


def test_flat_dishes_use_thin_height():
    """All categorised flat dishes share the same flat-prism prior."""
    for cls in ["pizza", "pancakes", "waffles", "tacos"]:
        info = nutrition.lookup(cls)
        assert info.typical_height_cm is not None and info.typical_height_cm <= 2.0
        assert info.shape_factor is not None and info.shape_factor >= 0.8


def test_bowl_dishes_use_bowl_depth():
    for cls in ["soup", "ramen", "pho", "ice_cream"]:
        info = nutrition.lookup(cls)
        assert info.typical_height_cm is not None and 2.5 <= info.typical_height_cm <= 5.0
        assert info.shape_factor is not None and info.shape_factor >= 0.9
