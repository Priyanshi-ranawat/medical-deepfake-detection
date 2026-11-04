import numpy as np
import pytest

from src.gradcam_alignment import alignment_metrics, sample_bilinear


def test_perfect_peak_alignment():
    cam = np.zeros((224, 224), dtype=np.float32)
    cam[100, 90] = 1
    result = alignment_metrics(cam, 90, 100)
    assert result["peak_distance_px"] == 0
    assert result["hit_16px"]
    assert result["top_1pct_hit"]
    assert result["annotation_percentile"] == 1


def test_known_distance_and_radius_hits():
    cam = np.zeros((64, 64), dtype=np.float32)
    cam[40, 30] = 1
    result = alignment_metrics(cam, 6, 22)
    assert result["peak_distance_px"] == pytest.approx(30)
    assert not result["hit_16px"]
    assert result["hit_32px"]


def test_bilinear_sampling_fractional_point():
    cam = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert sample_bilinear(cam, 0.5, 0.5) == pytest.approx(0.5)


def test_zero_heatmap_has_safe_center_metrics():
    result = alignment_metrics(np.zeros((8, 8)), 2, 3)
    assert np.isnan(result["center_distance_px"])
    assert np.isfinite(result["peak_distance_px"])
    assert not result["cam_valid"]
    assert not result["hit_16px"]
    assert not result["top_10pct_hit"]


def test_invalid_heatmap_rejected():
    with pytest.raises(ValueError):
        alignment_metrics(np.array([[np.nan]]), 0, 0)

