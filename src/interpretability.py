"""Public Team C interpretability API.

This facade keeps the requested ``src/interpretability.py`` entry point while
the implementation remains separated into Grad-CAM, alignment, and CLI modules.
"""
from src.explain import analyze, main, save_heatmap, save_overlay
from src.gradcam import GradCAM
from src.gradcam_alignment import alignment_metrics, sample_bilinear, summarize_alignment

__all__ = [
    "GradCAM",
    "alignment_metrics",
    "sample_bilinear",
    "summarize_alignment",
    "save_heatmap",
    "save_overlay",
    "analyze",
    "main",
]


if __name__ == "__main__":
    main()

