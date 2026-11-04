# EfficientNet-B0 training

Uses the unchanged shared pipeline from `src/train.py` and `src/models.py`,
running the eight matched-protocol commands with the default architecture
`efficientnet_b0`. This is the exact protocol used for ResNet-18
(`docs/RESNET_MATCHED_PROTOCOL.md`), so the two architectures are directly
comparable.

```bash
python -m src.train --arch efficientnet_b0 --task removal   --train-method CTGAN --seed 0
python -m src.train --arch efficientnet_b0 --task removal   --train-method CTGAN --seed 1
python -m src.train --arch efficientnet_b0 --task removal   --train-method CTGAN --seed 2
python -m src.train --arch efficientnet_b0 --task injection --train-method CTGAN --seed 0
python -m src.train --arch efficientnet_b0 --task injection --train-method CTGAN --seed 1
python -m src.train --arch efficientnet_b0 --task injection --train-method CTGAN --seed 2
python -m src.train --arch efficientnet_b0 --task removal   --train-method SD    --seed 0
python -m src.train --arch efficientnet_b0 --task injection --train-method SD    --seed 0
```

Set `BTD_DATA_ROOT` to the BTD archive containing `CT/` and `MRI/`.
Alternatively, `python scripts/run_matched_efficientnet.py --root /path/to/archive`
runs all eight commands with provenance (source hashes, per-run command/log,
checkpoint hashes) and then writes the aggregate summaries.

Defaults (identical to ResNet): ImageNet weights, batch 32, maximum 30 epochs,
three head-only warmup epochs, head LR 0.001, full-network LR 0.0001, weight
decay 0.01, dropout 0.3, cosine scheduling, class weighting, and early stopping
with patience 7. Checkpoints are selected by validation AUC. Split seed is 42;
training seeds are listed above. Runs used Apple MPS.

`results/efficientnet_matched_{default_0.5,youden_val,recall0.95_val}.csv` report
mean ± sample standard deviation across seeds for every split (validation,
same-method test, cross-method test). SD-trained models have only one seed.
Thresholds are chosen on validation and reused unchanged on test; reaching 95%
validation recall does not guarantee 95% test recall.

These BTD test populations were inspected during Team A's earlier work; this is a
fixed-protocol comparison, not an independent confirmation. Keep the two CT tasks
separate, and note that authentic samples are shared across the two test methods.
Matching seeds/settings does not guarantee identical random streams across
architectures or hardware.
