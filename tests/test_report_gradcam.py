import pandas as pd
import pytest

from src.report_gradcam import build_cross_reference


def test_cross_reference_joins_classifier_metrics():
    alignment = pd.DataFrame([
        {"architecture": "resnet18", "modality": "CT", "task": "injection",
         "train_method": "CTGAN", "split": "test_same", "path": "a.npy", "scan_id": "s1",
         "cam_valid": True, "peak_distance_px": 10.0, "annotation_percentile": .9,
         "hit_32px": True, "top_10pct_hit": True},
        {"architecture": "resnet18", "modality": "CT", "task": "injection",
         "train_method": "CTGAN", "split": "test_same", "path": "b.npy", "scan_id": "s2",
         "cam_valid": True, "peak_distance_px": 50.0, "annotation_percentile": .4,
         "hit_32px": False, "top_10pct_hit": False},
    ])
    results = pd.DataFrame([{
        "arch": "resnet18", "modality": "CT", "task": "injection", "train_method": "CTGAN",
        "split": "test_same", "auc": "0.95 ± 0.01", "recall": "0.87 ± 0.02", "fn": "42 ± 5",
    }])
    report = build_cross_reference(alignment, results)
    assert len(report) == 1
    assert report.loc[0, "gradcam_rows"] == 2
    assert report.loc[0, "median_peak_distance_px"] == 30
    assert report.loc[0, "hit_32px_rate"] == .5
    assert report.loc[0, "classifier_auc"] == "0.95 ± 0.01"


def test_cross_reference_requires_alignment_schema():
    with pytest.raises(ValueError, match="missing columns"):
        build_cross_reference(pd.DataFrame({"path": ["a"]}), pd.DataFrame())

