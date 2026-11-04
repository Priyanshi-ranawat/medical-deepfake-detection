# Team C Grad-CAM protocol

`src.explain` reconstructs a trained one-logit detector from `best.pt`, uses the
architecture-specific target returned by `src.models.gradcam_target_layer`, and
computes Grad-CAM for the manipulated logit. It analyzes manipulated rows only.

This implementation is based on `feature/evaluation`, which contains the complete
`feature/data-loader` commit `70bd040`. Team C reuses Team A's public loaders
directly; it does not independently parse manifests, recrop images, or reconstruct
coordinates.

Run from the repository root:

```bash
python -m src.explain \
  --checkpoint runs/resnet18_CT_injection_CTGAN_pre_seed0/best.pt \
  --root "$BTD_DATA_ROOT" \
  --output runs/resnet18_CT_injection_CTGAN_pre_seed0/gradcam \
  --splits val test_same test_other \
  --save-overlays --max-overlays 100
```

The command writes `alignment.csv`, `summary.csv`, `manifest.json`, standalone
heatmaps, and overlays. Green crosses are CSV points; cyan circles are CAM maxima.
`--max-overlays` applies per split; set it to `0` to save every manipulated image.

## Valid comparisons

- Use validation/test loaders only. Training metadata precedes random spatial
  augmentation and therefore has `coordinates_valid=False`.
- Use `meta.x/y`, which are pixel-center coordinates in the resized 224x224 ROI,
  rather than the original-slice `source_x/y`.
- Restrict localization claims to manipulated rows. Authentic coordinates are
  ROI sampling points and do not denote manipulated tissue.
- Keep injection and removal experiments separate, and retain checkpoint-selected
  validation thresholds when labeling true positives and false negatives.
- CT exposes `val`, `test_same`, and `test_other`; the latter switches between
  CTGAN and Stable Diffusion while retaining the source allocation. MRI exposes
  `val` and `test` and is always reported as Stable Diffusion (`SD`).
- Preserve every manifest row. Repeated paths or coordinates remain distinct
  ROI samples under Team A's frozen contract.

## Metrics and interpretation

The output includes peak distance, normalized peak distance, CAM center-of-mass
distance, activation percentile at the annotation, hits within 16/32/48 pixels,
and containment in the top 1/5/10/20 percent of activation. Multiple metrics are
reported because the final target feature map is coarse and an upsampled CAM does
not have true 224x224 localization precision.

The CSV annotation is a point, not a segmentation mask. Grad-CAM indicates image
regions that influenced a score; it does not prove that the highlighted pixels
are the complete manipulated region or a clinically meaningful explanation.

## Cross-reference with Team B results

After analyzing one or more checkpoints, join the localization results to the
committed validation-Youden classifier summary:

```bash
python -m src.report_gradcam \
  --alignment runs/*/gradcam/alignment.csv \
  --model-results results/arch_comparison_youden_val.csv \
  --output results/gradcam_cross_reference
```

This writes `cross_reference.csv` and `cross_reference.md`, combining AUC, recall,
and false negatives with median peak distance, hit-at-32-pixel rate, and top-10%
activation containment. It cannot generate localization rows without checkpoints.
