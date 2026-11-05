# Medical Deepfake Detection

A collaborative research project for developing a 2D deep-learning pipeline that distinguishes authentic CT/MRI scans from synthetically manipulated scans.

## Project status

The data loaders, ResNet/EfficientNet classifiers, evaluation pipeline, and Team C
interpretability workflow are implemented. Team C evaluated all 29 supplied
checkpoints (21 EfficientNet and 8 ResNet), producing 11,546 Grad-CAM alignment
rows and 340 true-positive heatmap/overlay pairs. See the
[Team C results report](docs/TEAM_C_RESULTS.md) and [PROJECT_TRACKER.md](PROJECT_TRACKER.md).

**Team A's authoritative preprocessing specification is
[contract v1](docs/PREPROCESSING_CONTRACT.md)**, with a
[machine-readable snapshot](docs/preprocessing_contract_v1.json). It records the
frozen defaults and interface; audit reports retain their historical scope.
The [complete mechanical audit](docs/FULL_PREPROCESSING_AUDIT.md) passed all
5,791 CT/MRI regions and all public loader splits; the combined repository suite
has 94 passing tests.

## Implemented approach

- Load CT `.npy` slices and MRI grayscale PNGs from the BTD test sets.
- Train ResNet and EfficientNet binary classifiers.
- Evaluate generalization across GAN- and diffusion-based manipulations.
- Generate Grad-CAM heatmaps for model interpretation.
- Track accuracy, precision, recall, F1 score, and false negatives.

## Explainability and reporting

`src/interpretability.py` implements Grad-CAM for both model families, saves
heatmaps/overlays for correctly detected manipulated samples, and measures whether
the hottest heatmap pixel is close to the CSV manipulation coordinate. It reports
Euclidean distance, Hit@16, Hit@32, and top-region containment.

Run one checkpoint:

```bash
python -m src.interpretability \
  --checkpoint /path/to/checkpoint.pt \
  --root "$BTD_DATA_ROOT" \
  --output outputs/gradcam_run \
  --save-overlays \
  --visual-selection true-positive
```

Run and summarize a directory of checkpoints:

```bash
python scripts/run_all_gradcam.py \
  --checkpoint-root /path/to/checkpoint-export \
  --data-root "$BTD_DATA_ROOT" \
  --output-root outputs/all_checkpoints

python -m src.summarize_all_gradcam \
  --output-root outputs/all_checkpoints \
  --checkpoint-root /path/to/checkpoint-export \
  --evaluation-export /path/to/evaluation-export/runs \
  --report-output outputs/all_checkpoints_report
```

Lightweight, path-free aggregate results are tracked in
[the cross-reference table](results/gradcam_all_cross_reference.csv),
[per-run localization summary](results/gradcam_localization_per_run.csv), and
[per-run classifier summary](results/gradcam_classifier_per_run.csv).

## Repository layout

```text
src/       Source code
docs/      Project documentation and reports
results/   Generated summaries and lightweight result artifacts
```

## Data pipeline (CT)

For the verified setup, follow the [reproducibility guide](docs/REPRODUCIBILITY.md).
Python and all 34 dependencies are pinned for macOS arm64; other platforms need
separate validation. After setup, `python -m src.check_environment` checks the
installed versions and frozen preprocessing fingerprints.

```bash
# Exact direct versions; use the platform lock in the reproducibility guide
# to reproduce the verified macOS arm64 environment including transitive packages.
python -m pip install -r requirements.txt
# point the code at the dataset folder (the one containing CT/ and MRI/)
export BTD_DATA_ROOT=/path/to/archive          # PowerShell: $env:BTD_DATA_ROOT = "C:\data\btd\archive"

python -m src.check_data --task both           # splits, leakage, shortcut checks, sample images
python -m pytest tests -q                      # unit tests (synthetic data, no real scans needed)
```

```python
from src.data_loader import get_dataloaders

loaders = get_dataloaders("injection", train_method="CTGAN", batch_size=32)
# keys: train, val, test_same (CTGAN), test_other (Stable Diffusion = generalization test)
# each batch: images [B, 3, 224, 224], labels (1 = manipulated), meta (path, scan_id, x, y, ...)
```

Splits are made by scan ID (never by slice), stratified by scanner. Settings live in `src/config.py`.

CT preprocessing follows the BTD reference [evaluation notebook](https://github.com/FreddieMG/BTD--Unsupervised-Detection-of-Medical-Deepfakes/blob/0fa74daef6802d305dcca4e5e37b4393fdf89c5b/BTD_evaluation.ipynb):
clip to **[-700, 2000]**, scale to [0, 1], and extract a **96x96 patch at each CSV (x, y)**
for both authentic and manipulated rows. Patches near an edge shift inward rather than
being padded. Each CSV row remains a sample, including repeated paths and exact duplicate
rows, matching the reference loader. Counts therefore refer to ROI samples, not unique files.

The CNN adapter then resizes the patch to 224x224, duplicates it to three channels, applies
training-only augmentation, and optionally applies ImageNet normalization. These are CNN
adaptations, not part of the reference implementation's one-channel diffusion-model input. To inspect the native
patch values, use `DataConfig(image_size=96, imagenet_norm=False)`; each channel then contains
the reference patch, without augmentation when `train=False`.

`meta.x/y` are ROI coordinates within the resized patch for **both** classes; authentic
removal coordinates identify healthy tissue rather than a lesion. `source_x/y` and
`crop_left/top` preserve original-slice geometry. Patch coordinates are valid only without
augmentation (validation/test). Grad-CAM checks on manipulated regions should use fake rows.

Coordinates use zero-indexed **pixel centers**, matching `align_corners=False`:
`x = (source_x - crop_left + 0.5) * image_size / patch_size - 0.5`, likewise for y.
Both CT and MRI expose `coordinate_convention="pixel_center"` and
`coordinates_valid=False` during training. Downsampled edge coordinates may be
slightly negative within the outer pixel area; they are deliberately not clamped.

Whole-slice classification is not valid for the sampled CTGAN removal pairs: TB and FB files
can contain identical full arrays with different ROI coordinates and labels. Preserve the
coordinate-defined samples when changing this pipeline.

The full CT audit and downstream contract are documented in [Team A's CT handoff](docs/CT_DATA_HANDOFF.md).
Run `python -m src.audit_data --root "$BTD_DATA_ROOT" --out results/ct_audit` for exhaustive
reference, coordinate, hash, split, and batch checks. The handoff also records remaining
dataset biases and why injection/removal splits must not be concatenated.

The current [preprocessing correctness review](docs/PREPROCESSING_VERIFICATION.md)
records verified behavior and pending contract details. Resized `meta.x/y` now
match the resize operation's pixel-center geometry; training metadata precedes spatial augmentation.
The split seed does not by itself seed training augmentation or shuffle.

## Data pipeline (MRI)

```python
from src.mri_data_loader import MRIDataConfig, get_mri_dataloaders

loaders = get_mri_dataloaders("injection", root="/path/to/archive", batch_size=32)
# keys: train, val, test. MRI contains SD fakes only; no test_other population.
images, labels, meta = next(iter(loaders["val"]))
```

MRI uses grayscale PNG scaling to [0, 1] and 128x128 ROI patches. The default
**preserves native PNG orientation**, avoiding the reference implementation's unstable border-intensity
flip rule. For explicit reference comparisons, use
`MRIDataConfig(orientation_policy="author_border")`; that option uses standard
reflection **`height - 1 - y`**, correcting the notebook's `512 - y`.
Both MRI manifests are required: source groups are assigned once across injection
and removal, then each task is selected. The default CNN adapter returns
`float32 [B, 3, 224, 224]` with ImageNet normalization and training-only augmentation.
ROI metadata includes the orientation decision; `coordinates_valid=False` during
training because returned coordinates precede random augmentation.

See [MRI handoff](docs/MRI_DATA_HANDOFF.md) for the full contract, audit results,
and remaining limitations. Reference agreement applies to the explicit
`author_border` option, not the preserve-orientation default. The
[mitigation study](docs/MRI_MITIGATION_STUDY.md) documents the default change and
background observations. Further mitigation is paused; background AUC is not a
preprocessing acceptance gate. The tested intensity floor is not active.

## Dataset and artifact policy

Do not commit medical-image data, `.npy` files, dataset CSV exports containing
sensitive paths or restricted information, trained model weights, checkpoints, or
image-level experiment artifacts. Obtain the dataset through its authorized source
and keep it outside version control. Path-free aggregate summary tables are allowed
under `results/`.

## Collaboration

See [CONTRIBUTING.md](CONTRIBUTING.md) for the branch and pull-request workflow.

