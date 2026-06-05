"""Tests for the nutrition lookup stage."""
import pytest

from foodvol import nutrition


def test_known_class_lookup():
    info = nutrition.lookup("pizza")
    assert not info.is_default
    assert info.kcal_per_100g > 0
    assert info.density_g_per_ml > 0


def test_label_normalisation():
    # Different casing / separators resolve to the same entry.
    a = nutrition.lookup("French_Fries")
    b = nutrition.lookup("french fries")
    c = nutrition.lookup(" french-fries ")
    assert a.food_class == b.food_class == c.food_class
    assert not a.is_default


def test_unknown_class_falls_back_to_default():
    info = nutrition.lookup("definitely_not_a_food_12345")
    assert info.is_default
    assert info.kcal_per_100g > 0  # still usable


def test_mass_scaling_is_linear():
    info = nutrition.lookup("pizza")
    est = info.for_mass(200.0)
    assert est.mass_g == pytest.approx(200.0)
    assert est.kcal == pytest.approx(info.kcal_per_100g * 2.0)
    assert est.protein_g == pytest.approx(info.protein_g_per_100g * 2.0)


def test_volume_to_mass_uses_density():
    info = nutrition.lookup("apple")
    est = info.for_volume(100.0)  # 100 mL
    assert est.mass_g == pytest.approx(100.0 * info.density_g_per_ml)


def test_table_is_populated():
    classes = nutrition.known_classes()
    assert len(classes) >= 100
    assert "sushi" in classes and "apple" in classes
