"""Hyperparameter search for EfficientNet-B0 on CT removal/CTGAN.

Model selection uses VALIDATION ONLY (mean val AUC); test splits are never used
to choose hyperparameters. The goal is to reduce false negatives: higher
validation AUC means a better ROC, hence higher recall (fewer missed
manipulations) at any operating point. The chosen config is then re-run at three
seeds and compared to the shipped baseline, reporting test false negatives at the
validation-chosen 95%-recall threshold.

Phase 1: 8-point grid at seed 0 (lr x dropout x weight_decay around the baseline).
Phase 2: re-run the best config at seeds 1 and 2 (seed 0 reused from phase 1).
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCH, TASK, METHOD = 'efficientnet_b0', 'removal', 'CTGAN'
TUNE_DIR = ROOT / 'runs' / 'tune_efficientnet_removal'

# lr, dropout, weight_decay. The baseline is (1e-4, 0.3, 1e-2).
GRID = [
    dict(lr=1e-4, dropout=0.3, weight_decay=1e-2),   # baseline config
    dict(lr=5e-5, dropout=0.3, weight_decay=1e-2),
    dict(lr=2e-4, dropout=0.3, weight_decay=1e-2),
    dict(lr=1e-4, dropout=0.5, weight_decay=1e-2),
    dict(lr=5e-5, dropout=0.5, weight_decay=1e-2),
    dict(lr=2e-4, dropout=0.5, weight_decay=1e-2),
    dict(lr=1e-4, dropout=0.3, weight_decay=1e-1),
    dict(lr=1e-4, dropout=0.5, weight_decay=1e-1),
]


def tag(cfg):
    return f"lr{cfg['lr']:g}_do{cfg['dropout']:g}_wd{cfg['weight_decay']:g}"


def env():
    return {**os.environ, 'BTD_DATA_ROOT': str(Path(os.environ['BTD_DATA_ROOT']).resolve()),
            'TORCH_HOME': str(ROOT / 'weights'), 'OMP_NUM_THREADS': '4',
            'MKL_NUM_THREADS': '4', 'PYTHONUNBUFFERED': '1', 'PYTHONPATH': str(ROOT)}


def run(cfg, seed):
    out = TUNE_DIR / tag(cfg) / f'seed{seed}'
    name = f'{ARCH}_CT_{TASK}_{METHOD}_pre_seed{seed}'
    metrics = out / name / 'metrics.json'
    if metrics.exists():
        print(f'REUSE {tag(cfg)} seed{seed}', flush=True)
        return json.loads(metrics.read_text())
    cmd = [sys.executable, '-u', '-m', 'src.train', '--arch', ARCH, '--task', TASK,
           '--train-method', METHOD, '--seed', str(seed), '--out', str(out),
           '--lr', str(cfg['lr']), '--dropout', str(cfg['dropout']),
           '--weight-decay', str(cfg['weight_decay'])]
    print(f'RUN {tag(cfg)} seed{seed}', flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env(), check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return json.loads(metrics.read_text())


def summarize(m):
    s = m['splits']
    return {sp: {'auc': s[sp]['auc'],
                 'fn_recall95': s[sp]['recall0.95_val']['fn'],
                 'recall_recall95': s[sp]['recall0.95_val']['recall']}
            for sp in s}


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    TUNE_DIR.mkdir(parents=True, exist_ok=True)
    # Phase 1: grid at seed 0, selected on validation AUC only.
    phase1 = []
    for cfg in GRID:
        r = summarize(run(cfg, 0))
        phase1.append((cfg, r))
        print(f"  {tag(cfg):26s} val_auc={r['val']['auc']:.4f} "
              f"test_same_auc={r['test_same']['auc']:.4f} "
              f"test_other_auc={r['test_other']['auc']:.4f}", flush=True)
    best_cfg, best_r = max(phase1, key=lambda cr: cr[1]['val']['auc'])
    print(f"\nBEST by val AUC: {tag(best_cfg)} (val_auc={best_r['val']['auc']:.4f})", flush=True)
    # Phase 2: confirm best config at seeds 1 and 2.
    seeds = [summarize(run(best_cfg, 0))]
    for seed in (1, 2):
        seeds.append(summarize(run(best_cfg, seed)))

    def mean(split, key):
        import statistics
        vals = [s[split][key] for s in seeds]
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return sum(vals) / len(vals), sd

    report = {'arch': ARCH, 'task': TASK, 'train_method': METHOD,
              'selection': 'validation AUC (test never used for selection)',
              'grid': [{'config': tag(c), 'val_auc': r['val']['auc'],
                        'test_same_auc': r['test_same']['auc'],
                        'test_other_auc': r['test_other']['auc']} for c, r in phase1],
              'baseline_config': tag(GRID[0]), 'best_config': tag(best_cfg),
              'best_3seed': {sp: {'auc_mean': mean(sp, 'auc')[0], 'auc_sd': mean(sp, 'auc')[1],
                                  'fn_recall95_mean': mean(sp, 'fn_recall95')[0],
                                  'recall_recall95_mean': mean(sp, 'recall_recall95')[0]}
                             for sp in ('val', 'test_same', 'test_other')}}
    (ROOT / 'results').mkdir(exist_ok=True)
    (ROOT / 'results' / 'tuning_efficientnet_removal.json').write_text(json.dumps(report, indent=2) + '\n')
    print('\nwrote results/tuning_efficientnet_removal.json', flush=True)
    print('TUNING COMPLETE', flush=True)


if __name__ == '__main__':
    main()
