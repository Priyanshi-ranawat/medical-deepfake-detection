"""Run Team C Grad-CAM analysis for every checkpoint under one runs directory."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-overlays", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_root = Path(args.checkpoint_root)
    checkpoints = sorted(checkpoint_root.glob("runs/*/best.pt"))
    if not checkpoints:
        raise SystemExit(f"no runs/*/best.pt checkpoints found under {checkpoint_root}")
    output_root = Path(args.output_root)
    environment = os.environ.copy()
    environment.setdefault("MPLCONFIGDIR", "/private/tmp/codex-matplotlib-cache")

    completed = skipped = 0
    for index, checkpoint in enumerate(checkpoints, 1):
        run_name = checkpoint.parent.name
        output = output_root / run_name / "gradcam"
        alignment = output / "alignment.csv"
        if alignment.exists() and not args.force:
            skipped += 1
            print(f"[{index}/{len(checkpoints)}] SKIP {run_name}", flush=True)
            continue
        print(f"[{index}/{len(checkpoints)}] RUN  {run_name}", flush=True)
        command = [
            sys.executable, "-m", "src.interpretability",
            "--checkpoint", str(checkpoint),
            "--root", args.data_root,
            "--output", str(output),
            "--device", args.device,
            "--batch-size", str(args.batch_size),
            "--save-overlays",
            "--max-overlays", str(args.max_overlays),
            "--visual-selection", "true-positive",
        ]
        subprocess.run(command, check=True, env=environment)
        completed += 1
    print(f"complete: analyzed={completed} skipped={skipped} total={len(checkpoints)}", flush=True)


if __name__ == "__main__":
    main()
