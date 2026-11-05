"""Run checkpoint-driven Grad-CAM and point-alignment analysis."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.gradcam import GradCAM
from src.gradcam_alignment import alignment_metrics, summarize_alignment
from src.models import build_model, gradcam_target_layer
from src.train import pick_device

IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406))[:, None, None]
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225))[:, None, None]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--root", default=None, help="dataset root; defaults to $BTD_DATA_ROOT")
    p.add_argument("--output", required=True)
    p.add_argument("--splits", nargs="+", default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--save-overlays", action="store_true")
    p.add_argument("--visual-selection", choices=["true-positive", "false-negative", "all-manipulated"],
                   default="true-positive",
                   help="which manipulated rows receive PNG heatmaps; metrics still cover every fake row")
    p.add_argument("--max-overlays", type=int, default=100, help="0 means no limit")
    p.add_argument("--limit-batches", type=int, default=0, help="smoke-test limit per split")
    return p.parse_args()


def _checkpoint_args(checkpoint):
    args = checkpoint.get("args")
    if not isinstance(args, dict):
        raise ValueError("checkpoint is missing its training args dictionary")
    required = {"arch", "modality", "task"}
    missing = required - set(args)
    if missing:
        raise ValueError(f"checkpoint args missing {sorted(missing)}")
    return args


def make_loaders(args, root, batch_size, num_workers):
    if args["modality"] == "CT":
        from src.data_loader import get_dataloaders
        return get_dataloaders(args["task"], train_method=args.get("train_method", "CTGAN"),
                               root=root, batch_size=batch_size, num_workers=num_workers)
    if args["modality"] == "MRI":
        from src.mri_data_loader import get_mri_dataloaders
        return get_mri_dataloaders(args["task"], root=root, batch_size=batch_size,
                                   num_workers=num_workers)
    raise ValueError(f"unsupported modality {args['modality']!r}")


def _meta_value(meta, key, index):
    value = meta[key]
    item = value[index]
    return item.item() if torch.is_tensor(item) else item


def _validate_coordinate_contract(meta, split):
    valid = meta.get("coordinates_valid")
    if valid is not None and not bool(torch.as_tensor(valid).all()):
        raise ValueError(f"split {split!r} contains invalid/augmented coordinates")
    conventions = meta.get("coordinate_convention")
    if conventions is not None and any(value != "pixel_center" for value in conventions):
        raise ValueError(f"split {split!r} does not use the pixel_center coordinate contract")


def _display_image(image):
    image = (image.detach().cpu() * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)
    return image.permute(1, 2, 0).numpy()


def save_overlay(path, image, cam, point, peak, title):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    ax.imshow(_display_image(image), cmap="gray")
    ax.imshow(cam, cmap="turbo", alpha=0.42, vmin=0, vmax=1)
    ax.scatter([point[0]], [point[1]], marker="+", s=100, linewidths=2, c="lime", label="CSV point")
    ax.scatter([peak[0]], [peak[1]], marker="o", s=55, facecolors="none", edgecolors="cyan",
               linewidths=1.5, label="CAM peak")
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    ax.legend(loc="lower right", fontsize=7)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_heatmap(path, cam, point, peak, title):
    """Save the normalized CAM by itself with annotation and maximum markers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    image = ax.imshow(cam, cmap="turbo", vmin=0, vmax=1)
    ax.scatter([point[0]], [point[1]], marker="+", s=100, linewidths=2, c="lime", label="CSV point")
    ax.scatter([peak[0]], [peak[1]], marker="o", s=55, facecolors="none", edgecolors="cyan",
               linewidths=1.5, label="CAM peak")
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    ax.legend(loc="lower right", fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Normalized Grad-CAM")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _selected_for_visual(correct: bool, selection: str) -> bool:
    if selection == "true-positive":
        return correct
    if selection == "false-negative":
        return not correct
    if selection == "all-manipulated":
        return True
    raise ValueError(f"unknown visual selection {selection!r}")


def analyze(args):
    device = pick_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    run_args = _checkpoint_args(checkpoint)
    model = build_model(run_args["arch"], pretrained=False, dropout=run_args.get("dropout", 0.3))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()

    root = args.root or os.environ.get("BTD_DATA_ROOT")
    loaders = make_loaders(run_args, root, args.batch_size, args.num_workers)
    available = [name for name in loaders if name != "train"]
    splits = args.splits or available
    unknown = set(splits) - set(available)
    if unknown:
        raise ValueError(f"invalid evaluation splits {sorted(unknown)}; available: {available}")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    threshold = checkpoint.get("thresholds", {}).get("youden_val", 0.5)
    train_method = run_args.get("train_method", "CTGAN") if run_args["modality"] == "CT" else "SD"
    run_name = Path(args.checkpoint).parent.name
    visual_selection = getattr(args, "visual_selection", "true-positive")
    rows, visual_counts = [], {split: 0 for split in splits}
    with GradCAM(model, gradcam_target_layer(model, run_args["arch"])) as gradcam:
        for split in splits:
            for batch_index, (images, labels, meta) in enumerate(loaders[split]):
                if args.limit_batches and batch_index >= args.limit_batches:
                    break
                _validate_coordinate_contract(meta, split)
                fake_indices = torch.nonzero(labels == 1, as_tuple=False).flatten()
                if not len(fake_indices):
                    continue
                selected = images[fake_indices].to(device)
                logits, cams = gradcam(selected)
                probabilities = torch.sigmoid(logits).cpu().numpy()
                cams_np = cams.cpu().numpy()

                for local_i, source_i_tensor in enumerate(fake_indices):
                    source_i = int(source_i_tensor)
                    x = float(_meta_value(meta, "x", source_i))
                    y = float(_meta_value(meta, "y", source_i))
                    path = str(_meta_value(meta, "path", source_i))
                    metrics = alignment_metrics(cams_np[local_i], x, y)
                    probability = float(probabilities[local_i])
                    row = {
                        "architecture": run_args["arch"],
                        "run": run_name,
                        "modality": run_args["modality"],
                        "task": run_args["task"],
                        "train_method": train_method,
                        "seed": run_args.get("seed"),
                        "pretrained": bool(run_args.get("pretrained", True)),
                        "split": split,
                        "path": path,
                        "scan_id": str(_meta_value(meta, "scan_id", source_i)),
                        "label": 1,
                        "probability": probability,
                        "threshold": float(threshold),
                        "prediction": int(probability >= threshold),
                        "correct": bool(probability >= threshold),
                        **metrics,
                    }
                    for key in ("source_x", "source_y", "crop_left", "crop_top"):
                        if key in meta:
                            row[key] = _meta_value(meta, key, source_i)
                    if "coordinate_convention" in meta:
                        row["coordinate_convention"] = str(
                            _meta_value(meta, "coordinate_convention", source_i)
                        )
                    split_visual_index = visual_counts[split]
                    if args.save_overlays and _selected_for_visual(row["correct"], visual_selection) and (
                        args.max_overlays == 0 or split_visual_index < args.max_overlays
                    ):
                        name = f"{split}_{split_visual_index:05d}.png"
                        title = (f"{run_args['arch']} | {split} | P(manip)={probability:.3f} | "
                                 f"distance={metrics['peak_distance_px']:.1f}px")
                        save_overlay(output / "overlays" / name, images[source_i], cams_np[local_i],
                                     (x, y), (metrics["peak_x"], metrics["peak_y"]), title)
                        save_heatmap(output / "heatmaps" / name, cams_np[local_i], (x, y),
                                     (metrics["peak_x"], metrics["peak_y"]), title)
                        row["overlay_file"] = str(Path("overlays") / name)
                        row["heatmap_file"] = str(Path("heatmaps") / name)
                        visual_counts[split] += 1
                    rows.append(row)

    if not rows:
        raise RuntimeError("no manipulated evaluation samples were analyzed")
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "alignment.csv", index=False)
    summary = summarize_alignment(
        frame, ["architecture", "modality", "task", "train_method", "seed", "split", "correct"]
    )
    summary.to_csv(output / "summary.csv", index=False)
    manifest = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_val_auc": checkpoint.get("val_auc"),
        "run_args": run_args,
        "splits": splits,
        "n_manipulated_rows": len(frame),
        "n_visuals_by_split": visual_counts,
        "n_overlays": sum(visual_counts.values()),
        "n_heatmaps": sum(visual_counts.values()),
        "visual_selection": visual_selection,
        "alignment_note": "CSV annotations are points, not manipulation masks.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"wrote {len(frame)} rows and {sum(visual_counts.values())} visual pairs to {output}")


def main():
    analyze(parse_args())


if __name__ == "__main__":
    main()

