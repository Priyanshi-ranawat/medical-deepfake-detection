# Team C: Grad-CAM Interpretability Results

## Deliverables

Team C's implementation is complete:

- `src/interpretability.py` loads ResNet18 or EfficientNet-B0 checkpoints, computes
  Grad-CAM, exports heatmaps/overlays, and checks localization against dataset CSV
  coordinates.
- `scripts/run_all_gradcam.py` discovers and evaluates a collection of checkpoints.
- `src/summarize_all_gradcam.py` combines individual runs into cross-reference and
  per-run summaries.
- The repository includes unit/integration coverage for Grad-CAM, checkpoint loading,
  coordinate comparison, image export, orchestration, and reporting.

## Method

Grad-CAM targets the final convolutional block of each architecture. The activation
map is resized to the model input size. For manipulated validation/test samples, the
evaluation records the hottest pixel and compares it with the manipulation coordinate
provided by the dataset loader. Localization is summarized using Euclidean distance,
Hit@16, Hit@32, and whether the annotation falls within a high-activation region.

Visual exports are restricted to true-positive predictions, as required by the
project plan. Each selected sample produces a heatmap and an image/heatmap overlay.

## Evaluation coverage

- Checkpoints evaluated: **29** (21 EfficientNet-B0, 8 ResNet18)
- Grad-CAM alignment rows: **11,546**
- True-positive heatmaps: **340**
- True-positive overlays: **340**
- Automated tests: **94 passed**

Aggregate, path-free results are available in:

- [Cross-reference table](../results/gradcam_all_cross_reference.csv)
- [Localization by run](../results/gradcam_localization_per_run.csv)
- [Classifier metrics by run](../results/gradcam_classifier_per_run.csv)

## Main findings

- Localization generalization drops substantially between same-method and
  other-method CT injection tests. EfficientNet Hit@32 falls from about **0.884**
  to **0.527**, while ResNet falls from about **0.871** to **0.305**.
- CT removal shows the same pattern: EfficientNet Hit@32 falls from about **0.534**
  to **0.201**, and ResNet from about **0.799** to **0.292**.
- Stable-Diffusion removal is the hardest cross-method condition. EfficientNet has
  roughly **0.027 Hit@32** and **0.005 recall**; ResNet has zero for both in the
  evaluated runs.
- Pretraining materially improves localization for EfficientNet CT removal. A
  from-scratch run has about **0.064 Hit@32** and **0.571 AUC**, compared with about
  **0.534 Hit@32** and **0.914 AUC** for the corresponding pretrained run.
- Strong classification does not guarantee coordinate localization. MRI removal can
  retain high classifier AUC while producing only about **0.083 Hit@32**.

These values summarize the evaluated supplied checkpoints; the CSV files retain the
per-run numbers needed to reproduce comparisons.

## Interpretation and limitations

The CSV annotation is a point, not a segmentation mask, so a miss does not prove that
the model attended to clinically irrelevant tissue. Conversely, a hit only establishes
spatial proximity to the provided point. Grad-CAM is also coarse because it originates
from a low-resolution final convolutional feature map.

The results should therefore be reported alongside classifier metrics, not as a
standalone proof of model validity. Center bias, ROI cropping, coordinate transforms,
and repeated samples can influence the measured distance. The implementation preserves
the loader's pixel-center convention and only evaluates coordinates marked valid.

Medical images, source paths, checkpoints, and image-level exports are intentionally
excluded from Git. They remain local; only aggregate path-free summaries are committed.
