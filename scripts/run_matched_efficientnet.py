"""Run the eight matched-protocol commands with --arch efficientnet_b0.

Mirrors scripts/run_matched_resnet.py exactly (same jobs, seeds, split seed and
provenance), changing only the architecture, so EfficientNet-B0 and ResNet-18
results are produced under one identical protocol and are directly comparable.
The shared src/train.py and src/models.py are used without modification.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ARCH = 'efficientnet_b0'
ROOT = Path(__file__).resolve().parents[1]
JOBS = [('removal', 'CTGAN', seed) for seed in (0, 1, 2)]
JOBS += [('injection', 'CTGAN', seed) for seed in (0, 1, 2)]
JOBS += [('removal', 'SD', 0), ('injection', 'SD', 0)]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    import torch
    protocol = ROOT / 'runs/matched_protocol_efficientnet'
    protocol.mkdir(parents=True, exist_ok=True)
    paths = ['src/train.py', 'src/models.py', 'src/config.py', 'src/data_loader.py',
             'src/mri_data_loader.py', 'src/splits.py']
    source = {p: digest(ROOT / p) for p in paths}
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    # Ensure the actual shared training/model source has not been modified.
    for p in paths:
        committed = subprocess.check_output(['git', 'show', f'HEAD:{p}'], cwd=ROOT)
        if hashlib.sha256(committed).hexdigest() != source[p]:
            raise ValueError(f'Shared source modified: {p}')
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    env = {**os.environ, 'BTD_DATA_ROOT': str(args.root.resolve()),
           'TORCH_HOME': str(ROOT / 'weights'), 'OMP_NUM_THREADS': '4',
           'MKL_NUM_THREADS': '4', 'PYTHONUNBUFFERED': '1'}
    plan = dict(shared_commit=commit, source_sha256=source, split_seed=42,
                training_seeds_CTGAN=[0, 1, 2], training_seeds_SD=[0],
                architecture=ARCH, auto_selected_device=device,
                python=platform.python_version(),
                packages={p: importlib.metadata.version(p) for p in
                          ('torch', 'torchvision', 'numpy', 'pandas', 'scikit-learn')},
                jobs=[dict(task=t, train_method=m, seed=s) for t, m, s in JOBS],
                rule='Matched protocol identical to run_matched_resnet.py; only --arch efficientnet_b0. All shared training defaults unchanged.',
                qualification='Fixed matched-protocol comparison on previously inspected BTD test populations; not a fresh independent evaluation. Do not tune settings from these test results.')
    save(protocol / 'plan.json', plan)
    for index, (task, method, seed) in enumerate(JOBS, 1):
        name = f'{ARCH}_CT_{task}_{method}_pre_seed{seed}'
        out = ROOT / 'runs' / name
        if (out / 'metrics.json').exists():
            print(f'REUSE completed {name}', flush=True)
            continue
        if out.exists() and any(out.iterdir()):
            raise ValueError(f'Partial output already exists; preserve and diagnose before retrying: {out}')
        command = [sys.executable, '-u', '-m', 'src.train', '--arch', ARCH,
                   '--task', task, '--train-method', method, '--seed', str(seed)]
        print(f'\nMATCHED RUN {index}/{len(JOBS)}: {name}', flush=True)
        save(protocol / f'{name}_command.json', dict(command=command, environment={k: env[k] for k in
             ('BTD_DATA_ROOT', 'TORCH_HOME', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}))
        started = time.monotonic()
        with (protocol / f'{name}.log').open('w') as log:
            with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1) as child:
                for line in child.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end='', flush=True)
                code = child.wait()
            if code:
                raise RuntimeError(f'Run failed ({code}); see {protocol / name}.log')
        if not (out / 'metrics.json').exists():
            raise RuntimeError('Training exited without final metrics')
        if any(digest(ROOT / p) != h for p, h in source.items()):
            raise ValueError('Shared source changed during the experiment')
        save(protocol / f'{name}_completion.json', dict(completed=True,
             duration_seconds=time.monotonic() - started, checkpoint_sha256=digest(out / 'best.pt')))
    for threshold in ('default_0.5', 'youden_val', 'recall0.95_val'):
        subprocess.run([sys.executable, '-m', 'src.summarize_runs', '--threshold', threshold,
                        '--out', str(ROOT / 'results' / f'efficientnet_matched_{threshold}.csv')],
                       cwd=ROOT, env=env, check=True)
    save(protocol / 'complete.json', dict(completed=True, runs=len(JOBS), shared_sources_unchanged=True))


if __name__ == '__main__':
    main()
