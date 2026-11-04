"""Spatial alignment metrics between Grad-CAM and CSV point annotations."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

DEFAULT_RADII = (16.0, 32.0, 48.0)
DEFAULT_TOP_PCTS = (1.0, 5.0, 10.0, 20.0)


def sample_bilinear(array: np.ndarray, x: float, y: float) -> float:
    """Sample a 2-D pixel-center array; edge coordinates use border values."""
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("heatmap must be a finite 2-D array")
    h, w = array.shape
    x = float(np.clip(x, 0.0, w - 1.0))
    y = float(np.clip(y, 0.0, h - 1.0))
    x0, y0 = int(math.floor(x)), int(math.floor(y))
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    dx, dy = x - x0, y - y0
    return float(
        array[y0, x0] * (1 - dx) * (1 - dy)
        + array[y0, x1] * dx * (1 - dy)
        + array[y1, x0] * (1 - dx) * dy
        + array[y1, x1] * dx * dy
    )


def alignment_metrics(
    heatmap: np.ndarray,
    x: float,
    y: float,
    radii: Iterable[float] = DEFAULT_RADII,
    top_pcts: Iterable[float] = DEFAULT_TOP_PCTS,
):
    """Measure peak, center-of-mass and high-activation alignment to one point."""
    cam = np.asarray(heatmap, dtype=np.float64)
    if cam.ndim != 2 or not np.isfinite(cam).all():
        raise ValueError("heatmap must be a finite 2-D array")
    if not np.isfinite([x, y]).all():
        raise ValueError("annotation coordinates must be finite")
    h, w = cam.shape
    peak_y, peak_x = np.unravel_index(int(np.argmax(cam)), cam.shape)
    peak_distance = math.hypot(float(peak_x) - x, float(peak_y) - y)
    cam_valid = bool(float(np.ptp(cam)) > 1e-12 and float(cam.sum()) > 0)

    mass = float(cam.sum())
    if mass > 0:
        yy, xx = np.indices(cam.shape, dtype=np.float64)
        center_x = float((cam * xx).sum() / mass)
        center_y = float((cam * yy).sum() / mass)
        center_distance = math.hypot(center_x - x, center_y - y)
    else:
        center_x = center_y = center_distance = float("nan")

    at_point = sample_bilinear(cam, x, y)
    result = {
        "annotation_x": float(x),
        "annotation_y": float(y),
        "peak_x": int(peak_x),
        "peak_y": int(peak_y),
        "peak_distance_px": float(peak_distance),
        "normalized_peak_distance": float(peak_distance / math.hypot(w, h)),
        "center_x": center_x,
        "center_y": center_y,
        "center_distance_px": center_distance,
        "annotation_cam_value": at_point,
        "annotation_percentile": float(np.mean(cam <= at_point)),
        "cam_valid": cam_valid,
    }
    for radius in radii:
        result[f"hit_{int(radius)}px"] = bool(cam_valid and peak_distance <= radius)
    for pct in top_pcts:
        if not 0 < pct <= 100:
            raise ValueError("top percentages must be in (0, 100]")
        cutoff = float(np.quantile(cam, 1.0 - pct / 100.0))
        result[f"top_{int(pct)}pct_hit"] = bool(cam_valid and at_point >= cutoff)
    return result


def summarize_alignment(frame, group_columns):
    """Return transparent descriptive aggregates without treating rows as patients."""
    metrics = {
        "path": "size",
        "peak_distance_px": "median",
        "normalized_peak_distance": "median",
        "center_distance_px": "median",
        "annotation_percentile": "mean",
        "cam_valid": "mean",
        "hit_16px": "mean",
        "hit_32px": "mean",
        "hit_48px": "mean",
        "top_1pct_hit": "mean",
        "top_5pct_hit": "mean",
        "top_10pct_hit": "mean",
        "top_20pct_hit": "mean",
    }
    out = frame.groupby(list(group_columns), dropna=False).agg(metrics).reset_index()
    return out.rename(columns={"path": "n_rows"})

