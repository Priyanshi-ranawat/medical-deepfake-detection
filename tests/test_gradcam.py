import pytest
import torch
import torch.nn as nn

from src.gradcam import GradCAM
from src.models import build_model, gradcam_target_layer


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Conv2d(3, 4, 3, padding=1)
        self.head = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.relu(self.features(x))
        return self.head(x.mean((2, 3)))


def test_gradcam_shape_range_and_hook_cleanup():
    model = TinyClassifier().eval()
    before = len(model.features._forward_hooks)
    with GradCAM(model, model.features) as gradcam:
        logits, cams = gradcam(torch.randn(2, 3, 16, 20))
        assert logits.shape == (2,)
        assert cams.shape == (2, 16, 20)
        assert torch.isfinite(cams).all()
        assert torch.all((0 <= cams) & (cams <= 1))
    assert len(model.features._forward_hooks) == before


def test_gradcam_rejects_wrong_input_rank():
    model = TinyClassifier().eval()
    with GradCAM(model, model.features) as gradcam:
        with pytest.raises(ValueError, match="B,C,H,W"):
            gradcam(torch.randn(3, 16, 16))


@pytest.mark.parametrize("arch", ["efficientnet_b0", "resnet18", "resnet34", "resnet50"])
def test_repository_models_have_gradcam_target(arch):
    model = build_model(arch, pretrained=False)
    assert isinstance(gradcam_target_layer(model, arch), nn.Module)

