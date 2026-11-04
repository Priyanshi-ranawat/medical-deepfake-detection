from argparse import Namespace

import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import src.explain as explain


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Conv2d(3, 4, 3, padding=1)
        self.head = nn.Linear(4, 1)

    def forward(self, x):
        return self.head(torch.relu(self.features(x)).mean((2, 3)))


class TinyEvaluationDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        return torch.rand(3, 16, 16), 1, {
            "path": f"sample_{index}.npy",
            "scan_id": "scan_a",
            "x": 7.0 + index,
            "y": 8.0,
            "source_x": 50 + index,
            "source_y": 60,
            "crop_left": 43,
            "crop_top": 52,
            "coordinate_convention": "pixel_center",
            "coordinates_valid": True,
        }


def test_analysis_writes_tables_manifest_and_overlays(tmp_path, monkeypatch):
    model = TinyClassifier()
    checkpoint = {
        "state_dict": model.state_dict(),
        "args": {"arch": "tiny", "modality": "CT", "task": "injection",
                 "train_method": "CTGAN", "dropout": 0.0, "seed": 0},
        "epoch": 2,
        "val_auc": 0.9,
        "thresholds": {"youden_val": 0.4},
    }
    checkpoint_path = tmp_path / "best.pt"
    torch.save(checkpoint, checkpoint_path)
    loader = DataLoader(TinyEvaluationDataset(), batch_size=2)

    monkeypatch.setattr(explain, "build_model", lambda *_args, **_kwargs: TinyClassifier())
    monkeypatch.setattr(explain, "gradcam_target_layer", lambda current, _arch: current.features)
    monkeypatch.setattr(explain, "make_loaders", lambda *_args, **_kwargs: {"val": loader})

    output = tmp_path / "gradcam"
    args = Namespace(checkpoint=str(checkpoint_path), root=None, output=str(output), splits=["val"],
                     batch_size=2, num_workers=0, device="cpu", save_overlays=True,
                     max_overlays=1, limit_batches=0)
    explain.analyze(args)

    alignment = pd.read_csv(output / "alignment.csv")
    assert len(alignment) == 2
    assert set(alignment["label"]) == {1}
    assert (output / "summary.csv").is_file()
    assert (output / "manifest.json").is_file()
    assert len(list((output / "overlays").glob("*.png"))) == 1
    assert len(list((output / "heatmaps").glob("*.png"))) == 1


def test_coordinate_contract_rejects_augmented_or_unknown_metadata():
    with pytest.raises(ValueError, match="invalid/augmented"):
        explain._validate_coordinate_contract({"coordinates_valid": torch.tensor([False])}, "train")

    with pytest.raises(ValueError, match="pixel_center"):
        explain._validate_coordinate_contract(
            {"coordinates_valid": torch.tensor([True]), "coordinate_convention": ["corner"]}, "val"
        )
