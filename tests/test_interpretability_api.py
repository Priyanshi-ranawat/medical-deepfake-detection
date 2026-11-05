from src.interpretability import GradCAM, alignment_metrics, analyze, save_heatmap


def test_requested_interpretability_module_exposes_public_api():
    assert GradCAM is not None
    assert callable(alignment_metrics)
    assert callable(analyze)
    assert callable(save_heatmap)

