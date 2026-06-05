"""Tests for the metric calibration stage."""
import cv2
import numpy as np
import pytest

from foodvol.calibration import CalibrationError, calibrate, calibrate_manual


def _disk_image(diameter_px=400, size=(600, 800)):
    img = np.zeros((*size, 3), np.uint8)
    cv2.circle(img, (size[1] // 2, size[0] // 2), diameter_px // 2, (230, 230, 230), -1)
    return img


def test_calibrate_largest_recovers_scale():
    img = _disk_image(diameter_px=400)
    calib = calibrate(img, real_diameter_cm=26.0, expect="largest")
    # Detected diameter should be within a few percent of the drawn 400 px.
    assert calib.diameter_px == pytest.approx(400, rel=0.05)
    assert calib.cm_per_px == pytest.approx(26.0 / 400, rel=0.05)


def test_calibrate_manual_is_exact():
    calib = calibrate_manual(diameter_px=200, real_diameter_cm=20.0)
    assert calib.cm_per_px == pytest.approx(0.1)
    assert calib.pixel_length_to_cm(50) == pytest.approx(5.0)
    assert calib.pixel_area_to_cm2(100) == pytest.approx(100 * 0.01)


def test_interior_mask_area_matches_disk():
    img = _disk_image(diameter_px=400)
    calib = calibrate(img, real_diameter_cm=26.0, expect="largest")
    mask = calib.interior_mask(img.shape)
    expected = np.pi * (calib.diameter_px / 2) ** 2
    assert mask.dtype == bool
    assert mask.sum() == pytest.approx(expected, rel=0.05)


def test_calibrate_raises_without_reference():
    blank = np.zeros((400, 400, 3), np.uint8)
    with pytest.raises(CalibrationError):
        calibrate(blank, real_diameter_cm=26.0, expect="largest")


def test_invalid_diameter_rejected():
    with pytest.raises(ValueError):
        calibrate(_disk_image(), real_diameter_cm=0.0)
