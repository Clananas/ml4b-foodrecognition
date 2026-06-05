"""Tests for the ECUSTFD dataset reader (skipped if the dataset is not downloaded)."""
import pytest

from foodvol.datasets import ECUSTFD

ds = ECUSTFD()
pytestmark = pytest.mark.skipif(not ds.is_available(),
                                reason="ECUSTFD not downloaded (run data/download_ecustfd.py)")


def test_ground_truth_parses():
    gt = ds.ground_truth()
    assert len(gt) > 100
    # apple001 is a known entry with a physically sensible density.
    ftype, vol, weight = gt["apple001"][0]
    assert ftype == "apple"
    assert vol > 0 and weight > 0
    assert 0.5 < weight / vol < 1.3   # density g/mL in a plausible food range


def test_portions_have_both_views():
    portions = ds.portions(single_food_only=True)
    assert len(portions) > 100
    for p in portions[:20]:
        assert p.top_images and p.side_images
        assert 0.1 < p.density_g_per_ml < 1.5


def test_annotation_has_coin_and_food_boxes():
    boxes = ds.annotation("apple001T(1)")
    names = {b.name for b in boxes}
    assert "coin" in names
    coin = ds.coin_box("apple001T(1)")
    assert coin is not None and coin.diameter_px > 0
    assert ds.food_boxes("apple001T(1)")
