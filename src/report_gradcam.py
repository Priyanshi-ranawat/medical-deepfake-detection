"""Cross-reference Grad-CAM localization with committed classifier results."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

JOIN_KEYS = ["arch", "modality", "task", "train_method", "split"]


def build_cross_reference(alignment: pd.DataFrame, model_results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate localization across seeds and join matching evaluation metrics."""
    required = {
        "architecture", "modality", "task", "train_method", "split", "path", "scan_id",
        "cam_valid", "peak_distance_px", "annotation_percentile", "hit_32px", "top_10pct_hit",
    }
    missing = required - set(alignment.columns)
    if missing:
        raise ValueError(f"alignment data missing columns: {sorted(missing)}")
    missing_results = set(JOIN_KEYS) - set(model_results.columns)
    if missing_results:
        raise ValueError(f"model results missing columns: {sorted(missing_results)}")

    data = alignment.rename(columns={"architecture": "arch"}).copy()
    location = data.groupby(JOIN_KEYS, dropna=False).agg(
        gradcam_rows=("path", "size"),
        gradcam_scans=("scan_id", "nunique"),
        valid_cam_rate=("cam_valid", "mean"),
        median_peak_distance_px=("peak_distance_px", "median"),
        mean_annotation_percentile=("annotation_percentile", "mean"),
        hit_32px_rate=("hit_32px", "mean"),
        top_10pct_hit_rate=("top_10pct_hit", "mean"),
    ).reset_index()

    classifier = model_results.copy()
    metric_columns = [column for column in classifier.columns if column not in JOIN_KEYS]
    classifier = classifier.rename(columns={column: f"classifier_{column}" for column in metric_columns})
    return location.merge(classifier, on=JOIN_KEYS, how="left", validate="many_to_one")


def write_markdown(frame: pd.DataFrame, path: Path):
    columns = [
        "arch", "modality", "task", "train_method", "split", "gradcam_rows",
        "median_peak_distance_px", "hit_32px_rate", "top_10pct_hit_rate", "classifier_auc",
        "classifier_recall", "classifier_fn",
    ]
    available = [column for column in columns if column in frame.columns]
    def render(value):
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value).replace("|", "\\|")

    table = ["| " + " | ".join(available) + " |",
             "| " + " | ".join("---" for _ in available) + " |"]
    table.extend("| " + " | ".join(render(value) for value in row) + " |"
                 for row in frame[available].itertuples(index=False, name=None))
    lines = [
        "# Grad-CAM and classifier cross-reference",
        "",
        "Localization is computed only for manipulated evaluation rows. CSV annotations are points, not masks.",
        "",
        *table,
        "",
    ]
    path.write_text("\n".join(lines))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment", nargs="+", required=True,
                        help="one or more alignment.csv files from src.explain")
    parser.add_argument("--model-results", default="results/arch_comparison_youden_val.csv")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    alignment = pd.concat([pd.read_csv(path) for path in args.alignment], ignore_index=True)
    model_results = pd.read_csv(args.model_results)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = build_cross_reference(alignment, model_results)
    report.to_csv(output / "cross_reference.csv", index=False)
    write_markdown(report, output / "cross_reference.md")
    print(f"wrote {len(report)} cross-reference rows to {output}")


if __name__ == "__main__":
    main()

