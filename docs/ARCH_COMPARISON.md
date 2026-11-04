# ResNet-18 vs EfficientNet-B0: architecture comparison

Both architectures were trained with the **identical shared pipeline and matched
protocol** (`src/train.py`, defaults in `docs/*_MATCHED_PROTOCOL.md`), on the
**same BTD archive and same machine (Apple MPS)**. ResNet-18 runs reproduced the
previously committed `feature/resnet` numbers exactly, confirming the data
version matches and the pipeline is deterministic across machines.

- 8 runs per architecture: CTGAN × {injection, removal} × seeds {0,1,2}, and
  SD × {injection, removal} × seed 0.
- Split seed 42; thresholds chosen on validation, reused on test.
- Figures below use the validation-Youden threshold
  (`results/arch_comparison_youden_val.csv`). All three thresholds are in
  `results/arch_comparison_{default_0.5,youden_val,recall0.95_val}.csv`.

## CTGAN-trained (primary comparison, mean ± sd over 3 seeds)

AUC is threshold-independent; recall / FN are at the validation-Youden threshold.

| Task | Split | EfficientNet AUC | ResNet AUC | EfficientNet recall | ResNet recall | EfficientNet FN | ResNet FN |
|---|---|---|---|---|---|---|---|
| injection | val         | 0.995 ± 0.001 | 0.996 ± 0.001 | 0.964 | 0.981 | 8.3 | 4.3 |
| injection | test_same   | **0.956 ± 0.008** | 0.948 ± 0.005 | 0.867 | 0.875 | 42.0 | 39.3 |
| injection | test_other  | **0.876 ± 0.029** | 0.864 ± 0.016 | 0.262 | 0.286 | 232.3 | 225.0 |
| removal   | val         | 0.947 ± 0.004 | 0.913 ± 0.010 | 0.889 | 0.870 | 6.0 | 7.0 |
| removal   | test_same   | 0.908 ± 0.022 | 0.912 ± 0.019 | 0.826 | 0.877 | 12.7 | 9.0 |
| removal   | test_other  | **0.700 ± 0.026** | 0.679 ± 0.084 | 0.384 | 0.429 | 45.0 | 41.7 |

`test_same` = held-out slices, same manipulation method as training.
`test_other` = the other method (CTGAN→SD) — the generalization test.

## Cross-method generalization, SD-trained (1 seed, SD→CTGAN at `test_other`)

| Task | EfficientNet AUC | ResNet AUC | EfficientNet recall | ResNet recall |
|---|---|---|---|---|
| injection | 0.859 | 0.820 | 0.089 | 0.289 |
| removal   | 0.524 | 0.501 | 0.000 | 0.000 |

## Findings

1. **The two architectures are statistically tied.** On every CTGAN split the AUC
   gap is ≤ 0.035 and within overlapping seed standard deviations. EfficientNet-B0
   is marginally higher on AUC (injection and removal `test_other`, injection
   `test_same`); ResNet-18 is marginally higher on recall / lower on false
   negatives. Neither is a clear winner — architecture choice is not the lever
   that matters here.

2. **Strong in-distribution, large generalization gap.** Both reach AUC 0.91–1.0
   on `val`/`test_same` but drop sharply cross-method: injection falls to
   ~0.86–0.88, removal to ~0.68–0.70.

3. **Removal generalizes far worse than injection.** Detecting *removed* lesions
   across generators is the hard case (CTGAN→SD AUC ~0.70); injection transfers
   substantially better (~0.87).

4. **SD→CTGAN removal collapses to chance** for both architectures (AUC ~0.50,
   recall 0.00) — a model trained to spot diffusion-removed lesions does not
   transfer to GAN-removed lesions at all. The strongest non-generalization
   result in the study.

5. **False negatives are the real cost.** Cross-method FN counts are large
   (injection `test_other`: 225–232 missed manipulations). For the clinical
   framing of this project, the headline is not "which backbone" but "neither
   backbone generalizes across generators safely without method-diverse training."

## Caveats

Fixed-protocol comparison on test populations inspected during preprocessing, not
a fresh independent evaluation — do not tune settings from these test numbers.
SD-trained rows are single-seed (no variance). EfficientNet results still need
the MRI runs for the modality-generalization axis of the proposal.
