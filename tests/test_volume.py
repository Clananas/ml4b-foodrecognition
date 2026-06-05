"""Tests for the volume estimator and its metrics."""
import numpy as np
import pytest

from foodvol.volume import VolumeEstimator, evaluate, volume_features


def test_physics_fallback_before_training():
    est = VolumeEstimator(shape_factor=0.5)
    assert not est.is_trained
    assert est.predict_volume(60.0, 5.0) == pytest.approx(0.5 * 60.0 * 5.0)


def test_features_dominant_term():
    f = volume_features(60.0, 5.0)
    assert f.tolist() == [60.0, 5.0, 300.0]


def test_proportional_recovers_shape_factor():
    rng = np.random.default_rng(0)
    areas = rng.uniform(20, 120, 200)
    heights = rng.uniform(2, 9, 200)
    volumes = 0.62 * areas * heights  # ground-truth shape factor
    est = VolumeEstimator().fit(areas, heights, volumes, model_kind="proportional")
    assert est.is_trained
    assert est.shape_factor == pytest.approx(0.62, rel=1e-6)
    assert est.predict_volume(100.0, 6.0) == pytest.approx(0.62 * 600.0, rel=1e-6)


def test_linear_model_fits_linear_data():
    rng = np.random.default_rng(1)
    areas = rng.uniform(20, 120, 300)
    heights = rng.uniform(2, 9, 300)
    volumes = 0.5 * areas * heights
    est = VolumeEstimator().fit(areas, heights, volumes, model_kind="huber")
    preds = est.predict_many(areas, heights)
    assert evaluate(volumes, preds)["MAPE_percent"] < 5.0


def test_save_load_roundtrip(tmp_path):
    areas = np.linspace(20, 120, 50)
    heights = np.linspace(2, 9, 50)
    volumes = 0.55 * areas * heights
    est = VolumeEstimator().fit(areas, heights, volumes, model_kind="proportional")
    path = est.save(tmp_path / "vol.joblib")
    loaded = VolumeEstimator.load(path)
    assert loaded.is_trained
    assert loaded.shape_factor == pytest.approx(est.shape_factor)


def test_evaluate_perfect_prediction():
    y = np.array([10.0, 20.0, 30.0])
    m = evaluate(y, y)
    assert m["MAE"] == 0 and m["MAPE_percent"] == 0
    assert m["R2"] == pytest.approx(1.0)
