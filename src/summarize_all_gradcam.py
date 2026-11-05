"""Summarize every completed Grad-CAM run across seeds and checkpoint variants."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

GROUP_KEYS = ["arch", "modality", "task", "train_method", "pretrained", "split"]


def load_alignment_runs(output_root: Path) -> pd.DataFrame:
    frames = []
    for alignment_path in sorted(output_root.glob("*/gradcam/alignment.csv")):
        run = alignment_path.parents[1].name
        manifest = json.loads((alignment_path.parent / "manifest.json").read_text())
        args = manifest["run_args"]
        frame = pd.read_csv(alignment_path)
        frame["run"] = run
        frame["arch"] = args["arch"]
        frame["modality"] = args["modality"]
        frame["task"] = args["task"]
        frame["train_method"] = args.get("train_method", "CTGAN") if args["modality"] == "CT" else "SD"
        frame["pretrained"] = bool(args.get("pretrained", True))
        frame["seed"] = int(args.get("seed", 0))
        frames.append(frame)
    if not frames:
        raise ValueError(f"no */gradcam/alignment.csv files found under {output_root}")
    return pd.concat(frames, ignore_index=True)


def summarize_localization(alignment: pd.DataFrame):
    per_run = alignment.groupby(["run", "seed", *GROUP_KEYS], dropna=False).agg(
        gradcam_rows=("path", "size"),
        gradcam_scans=("scan_id", "nunique"),
        valid_cam_rate=("cam_valid", "mean"),
        median_peak_distance_px=("peak_distance_px", "median"),
        mean_annotation_percentile=("annotation_percentile", "mean"),
        hit_32px_rate=("hit_32px", "mean"),
        top_10pct_hit_rate=("top_10pct_hit", "mean"),
    ).reset_index()
    summary = per_run.groupby(GROUP_KEYS, dropna=False).agg(
        n_runs=("run", "nunique"),
        gradcam_rows=("gradcam_rows", "sum"),
        median_peak_distance_mean=("median_peak_distance_px", "mean"),
        median_peak_distance_sd=("median_peak_distance_px", "std"),
        hit_32px_mean=("hit_32px_rate", "mean"),
        hit_32px_sd=("hit_32px_rate", "std"),
        top_10pct_hit_mean=("top_10pct_hit_rate", "mean"),
        top_10pct_hit_sd=("top_10pct_hit_rate", "std"),
        valid_cam_rate_mean=("valid_cam_rate", "mean"),
    ).reset_index()
    return per_run, summary


def load_classifier_runs(alignment: pd.DataFrame, checkpoint_root: Path, export_root: Path):
    rows = []
    for run in sorted(alignment["run"].unique()):
        candidates = [checkpoint_root / "runs" / run / "metrics.json", export_root / run / "metrics.json"]
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            raise FileNotFoundError(f"metrics.json not found for {run}")
        data = json.loads(path.read_text())
        args = data["args"]
        method = args.get("train_method", "CTGAN") if args["modality"] == "CT" else "SD"
        for split, split_metrics in data["splits"].items():
            point = split_metrics["youden_val"]
            rows.append({
                "run": run, "seed": int(args.get("seed", 0)), "arch": args["arch"],
                "modality": args["modality"], "task": args["task"], "train_method": method,
                "pretrained": bool(args.get("pretrained", True)), "split": split,
                "auc": split_metrics["auc"], "recall": point["recall"], "fn": point["fn"],
                "f1": point["f1"], "balanced_accuracy": point["balanced_accuracy"],
            })
    per_run = pd.DataFrame(rows)
    summary = per_run.groupby(GROUP_KEYS, dropna=False).agg(
        classifier_auc_mean=("auc", "mean"), classifier_auc_sd=("auc", "std"),
        classifier_recall_mean=("recall", "mean"), classifier_recall_sd=("recall", "std"),
        classifier_fn_mean=("fn", "mean"), classifier_fn_sd=("fn", "std"),
        classifier_f1_mean=("f1", "mean"), classifier_balanced_accuracy_mean=("balanced_accuracy", "mean"),
    ).reset_index()
    return per_run, summary


def write_markdown(frame: pd.DataFrame, path: Path):
    columns = ["arch", "modality", "task", "train_method", "pretrained", "split", "n_runs",
               "gradcam_rows", "median_peak_distance_mean", "hit_32px_mean", "top_10pct_hit_mean",
               "classifier_auc_mean", "classifier_recall_mean", "classifier_fn_mean"]
    def render(value):
        if pd.isna(value): return ""
        if isinstance(value, float): return f"{value:.3f}"
        return str(value)
    table = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    table += ["| " + " | ".join(render(value) for value in row) + " |"
              for row in frame[columns].itertuples(index=False, name=None)]
    path.write_text("\n".join(["# All-checkpoint Grad-CAM summary", "", *table, ""]))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--evaluation-export", required=True)
    parser.add_argument("--report-output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.report_output); output.mkdir(parents=True, exist_ok=True)
    alignment = load_alignment_runs(Path(args.output_root))
    alignment.to_csv(output / "all_alignment.csv", index=False)
    localization_runs, localization_summary = summarize_localization(alignment)
    classifier_runs, classifier_summary = load_classifier_runs(
        alignment, Path(args.checkpoint_root), Path(args.evaluation_export)
    )
    report = localization_summary.merge(classifier_summary, on=GROUP_KEYS, how="left", validate="one_to_one")
    localization_runs.to_csv(output / "localization_per_run.csv", index=False)
    classifier_runs.to_csv(output / "classifier_per_run.csv", index=False)
    report.to_csv(output / "all_cross_reference.csv", index=False)
    write_markdown(report, output / "all_cross_reference.md")
    print(f"summarized {alignment['run'].nunique()} checkpoints, {len(alignment)} alignment rows, {len(report)} groups")


if __name__ == "__main__":
    main()
