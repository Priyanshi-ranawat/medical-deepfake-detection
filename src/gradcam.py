"""Minimal Grad-CAM for the repository's one-logit image classifiers."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """Generate a normalized CAM from one convolutional target layer.

    The model's single output logit represents the manipulated class.  Passing
    ``score_sign=-1`` instead explains evidence for the authentic class.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.activations: Optional[torch.Tensor] = None
        self._handle = target_layer.register_forward_hook(self._capture)

    def _capture(self, _module, _inputs, output):
        if not isinstance(output, torch.Tensor):
            raise TypeError("Grad-CAM target layer must return a tensor")
        self.activations = output
        output.retain_grad()

    def close(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        self.close()

    def __call__(self, images: torch.Tensor, score_sign: float = 1.0):
        """Return ``(logits, cams)`` with CAMs shaped ``[B, H, W]`` in [0, 1]."""
        if images.ndim != 4:
            raise ValueError(f"expected [B,C,H,W] images, got {tuple(images.shape)}")
        self.model.zero_grad(set_to_none=True)
        self.activations = None

        # Grad-CAM needs gradients even when a caller has disabled them globally.
        with torch.enable_grad():
            logits = self.model(images).reshape(-1)
            if len(logits) != len(images):
                raise ValueError(f"expected one logit per image, got {tuple(logits.shape)}")
            (score_sign * logits.sum()).backward()

        if self.activations is None or self.activations.grad is None:
            raise RuntimeError("target layer did not produce Grad-CAM activations/gradients")
        gradients = self.activations.grad
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cams = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cams = F.interpolate(cams, size=images.shape[-2:], mode="bilinear", align_corners=False)

        flat = cams.flatten(1)
        low = flat.min(dim=1).values[:, None, None, None]
        high = flat.max(dim=1).values[:, None, None, None]
        cams = (cams - low) / (high - low).clamp_min(1e-8)
        return logits.detach(), cams[:, 0].detach()

