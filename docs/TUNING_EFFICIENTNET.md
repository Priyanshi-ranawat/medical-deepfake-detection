# Hyperparameter tuning — EfficientNet-B0, CT removal/CTGAN

**Goal:** reduce false negatives (missed manipulations). A higher-AUC model has a
better ROC, so it reaches higher recall (fewer FN) at any operating point.

**Selection rule:** validation AUC only. The test splits are never used to choose
hyperparameters; they are reported afterwards as held-out observations.

**Scope:** EfficientNet-B0 on CT removal/CTGAN — the shipped config with the most
headroom (injection was already at val AUC ~0.995). One task/architecture; the
procedure extends directly to the others.

**Search:** 8-point grid over learning rate × dropout × weight decay at seed 0,
around the baseline `(lr 1e-4, dropout 0.3, weight_decay 1e-2)`. The best config
by validation AUC was then re-run at seeds 1 and 2. Reproduce with
`python scripts/tune_efficientnet.py` (needs `BTD_DATA_ROOT`).

## Grid (seed 0), ranked by validation AUC

| lr | dropout | wd | val AUC | test_same AUC | test_other AUC |
|---|---|---|---|---|---|
| **2e-4** | **0.3** | **1e-2** | **0.9523** | 0.9116 | 0.7176 |
| 2e-4 | 0.5 | 1e-2 | 0.9496 | 0.9454 | 0.7320 |
| 1e-4 | 0.3 | 1e-1 | 0.9438 | 0.8833 | 0.6866 |
| 1e-4 | 0.3 | 1e-2 (baseline) | 0.9431 | 0.8823 | 0.6874 |
| 1e-4 | 0.5 | 1e-2 | 0.9342 | 0.9309 | 0.6771 |
| 1e-4 | 0.5 | 1e-1 | 0.9342 | 0.9313 | 0.6772 |
| 5e-5 | 0.3 | 1e-2 | 0.9091 | 0.8782 | 0.6082 |
| 5e-5 | 0.5 | 1e-2 | 0.8909 | 0.8572 | 0.5481 |

Learning rate is the dominant factor: 2e-4 > 1e-4 > 5e-5. Dropout and weight decay
have minor effect. **Winner: `lr 2e-4, dropout 0.3, weight_decay 1e-2`.**

## Winner vs baseline (3 seeds, at the validation 95%-recall threshold)

| Split | Baseline AUC | Tuned AUC | Baseline FN | Tuned FN | Baseline recall | Tuned recall |
|---|---|---|---|---|---|---|
| val | 0.947 ± 0.004 | **0.957 ± 0.004** | 2.0 | 2.0 | 0.963 | 0.963 |
| test_same | 0.908 ± 0.022 | **0.925 ± 0.011** | 5.3 | 5.7 | 0.927 | 0.922 |
| test_other | 0.700 ± 0.026 | **0.744 ± 0.028** | 30.7 | **27.0** | 0.580 | **0.630** |

## Findings

1. **Doubling the fine-tuning learning rate (1e-4 → 2e-4) is the win.** It lifts
   validation AUC 0.947 → 0.957 and, as a held-out effect, improves in-distribution
   test AUC (0.908 → 0.925) and cross-method test AUC (0.700 → 0.744).
2. **False negatives drop where it matters.** On the hard cross-method split,
   false negatives fall 30.7 → 27.0 and recall rises 0.58 → 0.63 at the same
   high-recall operating point — a real reduction in missed manipulations.
3. **Recommendation:** adopt `lr 2e-4` for CT removal. Lowering the LR (5e-5) is
   clearly worse. Before making it the global default, confirm it does not hurt
   injection (already near-ceiling) with a quick check.

## Caveats

Single task/architecture; selection used validation AUC only (no test peeking).
The cross-method gains are consistent but modest and within partially overlapping
seed variance — tuning narrows the generalization gap, it does not close it.
Method-diverse training data remains the larger lever (see `ARCH_COMPARISON.md`).
