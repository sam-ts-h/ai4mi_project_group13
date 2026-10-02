#!/usr/bin/env python3
"""
Per-organ ablation figure: one group of bars per experiment, one bar per
organ within each group -- background excluded (not really "an organ"),
consistent with plot_ablation_summary.py's framing.

Per-organ values are taken at EACH RUN'S OWN best epoch (same selection
rule as plot_ablation_summary.py / main.py's own checkpoint selection: mean
over the 4 organs, across val samples) -- i.e. this shows the actual
deployed best_epoch checkpoint's organ-wise breakdown, not a separately
cherry-picked best epoch per organ (which wouldn't correspond to any single
real saved model).

Groups are sorted by overall (4-organ mean) performance, left to right,
matching plot_ablation_summary.py's ordering so the two figures tell a
consistent story side by side.

Usage: python plot_ablation_per_organ.py \
        --run original_baseline:results/original_baseline \
        --run none:results/none --run clip:results/clip --run resample:results/resample \
        --run normalize:results/normalize --run clip_resample:results/clip_resample \
        --run clip_normalize:results/clip_normalize --run resample_normalize:results/resample_normalize \
        --run full:results/full \
        --dest ablation_per_organ.png
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

CLASS_NAMES = ["background", "esophagus", "heart", "trachea", "aorta"]
ORGAN_NAMES = CLASS_NAMES[1:]


def best_epoch_per_class(dice_val: np.ndarray) -> np.ndarray:
    """
    dice_val: shape (epochs, val_samples, K). Picks the best epoch using the
    same 4-organ-mean rule as main.py's own checkpoint selection, then
    returns that epoch's mean-over-val-samples Dice for EVERY class
    (including background, at index 0), shape (K,).
    """
    per_epoch_per_sample = dice_val[:, :, 1:].mean(axis=2)
    per_epoch_mean = per_epoch_per_sample.mean(axis=1)
    best_e = int(np.argmax(per_epoch_mean))
    return dice_val[best_e].mean(axis=0)  # (K,)


def main(args: argparse.Namespace) -> None:
    runs = []  # (label, per_class_array)
    for spec in args.run:
        label, path_str = spec.split(":", 1)
        dice_val = np.load(Path(path_str) / "dice_val.npy")
        per_class = best_epoch_per_class(dice_val)
        runs.append((label, per_class))
        print(f"{label:25s}: " + "  ".join(f"{n}={v:.3f}" for n, v in zip(CLASS_NAMES, per_class)))

    # Sort groups by overall (4-organ mean, background excluded) performance,
    # matching plot_ablation_summary.py's ordering -- selection/ranking stays
    # organ-only even though background is now ALSO plotted, so the x-axis
    # order is consistent across figures.
    runs.sort(key=lambda r: r[1][1:].mean(), reverse=True)
    labels = [r[0] for r in runs]
    n_runs = len(runs)
    n_classes = len(CLASS_NAMES)  # now plotting background too, not just the 4 organs

    x = np.arange(n_runs)
    width = 0.8 / n_classes

    fig, ax = plt.subplots(figsize=(max(10, 1.5 * n_runs), 6))
    for j, class_name in enumerate(CLASS_NAMES):
        heights = [r[1][j] for r in runs]
        offset = (j - (n_classes - 1) / 2) * width
        color = "lightgray" if j == 0 else None  # visually de-emphasize background vs. the 4 organs
        ax.bar(x + offset, heights, width=width, label=class_name, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Best-epoch validation Dice")
    ax.set_ylim(0, 1)
    ax.set_title("Preprocessing ablation: per-class Dice at each run's best epoch\n"
                "(background included for reference)")
    ax.legend(title="class", loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.dest, bbox_inches="tight")
    print(f"\nSaved per-organ comparison to {args.dest}")

    if not args.headless:
        plt.show()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-organ ablation comparison across multiple training runs")
    parser.add_argument("--run", action="append", required=True,
                        help="label:path_to_results_folder -- repeat once per run to compare")
    parser.add_argument("--dest", type=str, default="ablation_per_organ.png")
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())