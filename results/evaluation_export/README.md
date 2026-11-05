# Evaluation export — everything needed to write the report

This folder is a **complete, self-contained copy of every evaluation artifact**
produced by Team B, *except the model weights* (`best.pt`, 971 MB, deliberately
excluded — see "What is not here"). Everything required to write the report and
reproduce every table and figure is here, in plain CSV/JSON.

All numbers were produced with the unchanged shared pipeline (`src/train.py`,
`src/models.py`) on the same BTD archive and the same machine (Apple MPS). Split
seed is 42; thresholds are chosen on **validation only** and reused unchanged on
the test splits (no test leakage).

---

## Start here

| File | What it gives you |
|---|---|
| `RUN_INDEX.csv` | One row per training run (28 runs): arch, modality, task, method, seed, best epoch, and per-split n / AUC / false-negatives / recall at the 95%-recall operating point. The at-a-glance table. |
| `../arch_comparison_{default_0.5,youden_val,recall0.95_val}.csv` | ResNet-18 vs EfficientNet-B0, aggregated mean ± sd over seeds, at each of the three thresholds. The headline comparison tables. |
| `../efficientnet_matched_{...}.csv` | EfficientNet-only aggregated summaries at each threshold. |
| `../tuning_efficientnet_removal.json` | Full hyperparameter-tuning record (grid + winner at 3 seeds). |
| `../../docs/ARCH_COMPARISON.md` | Written-up findings (CT + MRI), already prose. |
| `../../docs/TUNING_EFFICIENTNET.md` | Written-up tuning story. |
| `../../docs/EFFICIENTNET_MATCHED_PROTOCOL.md`, `docs/RESNET_MATCHED_PROTOCOL.md` | Exact protocol / training defaults for the methods section. |

The `../` files above are already committed in `results/` and `docs/`. This
folder adds the **per-run, per-sample raw data** behind them.

---

## Directory layout

```
runs/
  <run_name>/                         # 28 main runs (see naming below)
    metrics.json                      # all metrics, all splits, all thresholds
    history.csv                       # per-epoch training curve
    preds_val.csv                     # per-sample predictions (validation)
    preds_test_same.csv               # per-sample predictions (same-method test)
    preds_test_other.csv              # per-sample predictions (cross-method test)
  matched_protocol_efficientnet/      # provenance for the CT EfficientNet runs
  matched_protocol_resnet/            # provenance for the CT ResNet runs
  matched_protocol_mri/               # provenance for the MRI runs
  tune_efficientnet_removal/          # the 8-point tuning grid (per-config runs)
```

MRI runs have `preds_val.csv` + `preds_test.csv` only (no `test_other` — MRI has
a single generator; see below).

---

## Run naming

`{arch}_{modality}_{task}_{method}_pre_seed{seed}`, e.g.
`efficientnet_b0_CT_removal_CTGAN_pre_seed2`.

- **arch**: `efficientnet_b0` or `resnet18`
- **modality**: `CT` or `MRI`
- **task**: `injection` (adding a fake lesion) or `removal` (erasing a real one)
- **method**: training generator — `CTGAN` or `SD` (Stable Diffusion)
- **seed**: training seed (0/1/2; SD-trained CT runs are seed 0 only)

### The two generalization axes
- **Generator (CT only):** train on one generator, test on the other.
  `test_same` = held-out slices of the *training* method; `test_other` = the
  *other* method (CTGAN→SD or SD→CTGAN). `test_other` is the generalization test.
- **Modality (MRI):** MRI in BTD is **Stable-Diffusion only**, so there is no
  within-MRI cross-generator test — MRI runs have `val` and `test` only and
  measure whether the CT-validated approach transfers to a second modality.

---

## File formats

### `metrics.json`
```
run            run name
args           every training hyperparameter (arch, lr, dropout, weight_decay,
               batch_size, epochs, seed, recall_target, device, ...)
best_epoch     epoch of the selected checkpoint (chosen by validation AUC)
thresholds     the three operating points, as probability cutoffs:
                 default_0.5, youden_val, recall0.95_val
splits         per split (val / test_same / test_other, or val / test for MRI):
  n            number of samples
  auc          ROC-AUC (threshold-independent)
  <threshold>  one block per threshold, each with:
               threshold, accuracy, balanced_accuracy, precision, recall,
               specificity, f1, tp, tn, fp, fn
```
`fn` (false negatives = missed manipulations) is the clinically critical count.

### `preds_*.csv` — per-sample predictions (for ROC curves, confusion matrices, error analysis)
```
label        ground truth (1 = manipulated, 0 = authentic)
prob         model's predicted probability of manipulation
path         source slice path within the BTD archive
scan_id      DICOM series UID (group by this for scan-level analysis)
scanner      scanner model string
cur_slice    slice index
x, y, source_x, source_y   lesion / crop coordinates
pred_youden  binary prediction at the validation-Youden threshold
```
These let you recompute **any** metric or curve at **any** threshold yourself —
e.g. `sklearn.metrics.roc_curve(label, prob)` per file.

### `history.csv` — training curves
`epoch, phase (head|full), train_loss, val_loss, val_auc, lr, seconds`.
`phase=head` = the 3 head-only warmup epochs; `phase=full` = full fine-tuning.

### Provenance (`matched_protocol_*/`)
- `plan.json` — shared commit, SHA256 of every source file, split seed, device,
  exact package versions (torch/torchvision/numpy/pandas/scikit-learn), job list.
- `<run>_command.json` — the exact command + environment for that run.
- `<run>_completion.json` — wall-clock duration and the **checkpoint SHA256**
  (so a regenerated `best.pt` can be verified bit-for-bit).
- `<run>.log` — full stdout/stderr of the run.
- `complete.json` — confirms all runs finished and sources were unchanged.

Use these for the methods / reproducibility section.

---

## One caveat to avoid confusion

In the **MRI** runs, `args.train_method` reads `CTGAN` — that is just the CLI
default; it is **ignored** for MRI (the MRI loader always uses the only available
generator, Stable Diffusion). The run **name** is authoritative: MRI = SD. The CT
runs' `train_method` is correct and meaningful.

---

## What is NOT here (and how to get it)

The **38 `best.pt` checkpoints (971 MB of trained weights)** are intentionally
excluded — too large for git, and not needed to *write* the report (every number
and per-sample prediction is already here). If the actual weights are ever needed
for new inference, they regenerate deterministically:

```bash
export BTD_DATA_ROOT=/path/to/archive      # BTD archive with CT/ and MRI/
python scripts/run_matched_efficientnet.py --root "$BTD_DATA_ROOT"
python scripts/run_matched_resnet_local.py --root "$BTD_DATA_ROOT"
python scripts/run_matched_mri.py          --root "$BTD_DATA_ROOT"
python scripts/tune_efficientnet.py        # needs BTD_DATA_ROOT set
```

Each run's `completion.json` holds the original checkpoint SHA256 to verify a
regenerated weight matches.

---

## Important framing for the report

These BTD test populations were inspected during Team A's earlier preprocessing
work, so this is a **fixed-protocol comparison, not a fresh independent
evaluation** — do not tune any setting from the test numbers. The CT SD-trained
rows are single-seed (no variance). See `docs/ARCH_COMPARISON.md` for the full
findings and caveats.
