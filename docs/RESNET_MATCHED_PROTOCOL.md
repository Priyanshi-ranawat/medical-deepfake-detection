# ResNet-18 training

Uses the unchanged shared pipeline from `feature/efficientnet` commit
`10973599e4ddec4a360e674d5503a9ab9fbafa73`, adding only `--arch resnet18`
to the EfficientNet teammate’s eight commands.

```bash
python -m src.train --arch resnet18 --task removal --train-method CTGAN --seed 0
python -m src.train --arch resnet18 --task removal --train-method CTGAN --seed 1
python -m src.train --arch resnet18 --task removal --train-method CTGAN --seed 2
python -m src.train --arch resnet18 --task injection --train-method CTGAN --seed 0
python -m src.train --arch resnet18 --task injection --train-method CTGAN --seed 1
python -m src.train --arch resnet18 --task injection --train-method CTGAN --seed 2
python -m src.train --arch resnet18 --task removal --train-method SD --seed 0
python -m src.train --arch resnet18 --task injection --train-method SD --seed 0
```

Set `BTD_DATA_ROOT` to the BTD v7 archive containing CT/ and MRI/.
Alternatively, `python scripts/run_matched_resnet.py --root /path/to/archive`
runs all eight commands and exports aggregate summaries.

Defaults: ImageNet weights, batch 32, maximum 30 epochs, three head-only warmup
epochs, head LR 0.001, full-network LR 0.0001, weight decay 0.01, dropout 0.3,
cosine scheduling, class weighting, and early stopping with patience 7.
Checkpoints are selected by validation AUC. Split seed is 42; training seeds
are listed above. Runs used Apple MPS.

`results/resnet18_metrics.csv` contains all eight runs, including validation,
same-method tests and cross-method tests at 0.5, validation Youden, and
validation 95%-recall thresholds. `resnet18_seed_summary.csv` reports means
and sample standard deviations; SD-trained models have only one seed.
Thresholds remain fixed on test data; achieving 95% validation recall does
not guarantee 95% test recall.

These test populations were inspected during earlier experiments; this is a
fixed protocol comparison, not an independent confirmation. Keep task-specific
splits separate, and account for shared authentic samples across test methods.
Matching seeds/settings does not ensure identical random streams across
architectures or hardware. EfficientNet results are needed for the architecture
comparison. Earlier custom ResNet experiments remain separate and local.
