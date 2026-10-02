#!/usr/bin/env python3
"""
Headline ablation figure: one bar per experiment, sorted by best-epoch
validation Dice averaged across the 4 organs (background excluded) -- the
exact same quantity main.py itself tracks as "current_dice" for picking the
best checkpoint, so this reports the metric the training code already
considers definitive, not a redefinition.

Error bars show the STANDARD DEVIATION ACROSS VALIDATION PATIENTS at that
best epoch -- this reflects patient-to-patient variability within one run,
NOT run-to-run training variance (that would need repeated seeds per
experiment, which this script does not assume you have). Labelled as such
on the figure to avoid overclaiming what it shows.

Usage:
    $ python plot_ablation_summary.py \
        --run original_baseline:results/original_baseline \
        --run none:results/none --run clip:results/clip --run resample:results/resample \
        --run normalize:results/normalize --run clip_resample:results/clip_resample \
        --run clip_normalize:results/clip_normalize --run resample_normalize:results/resample_normalize \
        --run full:results/full \
        --dest ablation_summary.png
"""

import argparse
from pathlib import Path
 
import numpy as np
import matplotlib.pyplot as plt
 
 
def best_epoch_mean_and_std(dice_val: np.ndarray) -> tuple[float, float, int]:
    """
    dice_val: shape (epochs, val_samples, K). Returns (best_mean, std_at_best,
    best_epoch_index) using the SAME definition main.py uses to pick its best
    checkpoint: mean over samples AND over classes 1..K-1 (background
    excluded), per epoch; then the epoch with the highest such value.
    """
    per_epoch_per_sample = dice_val[:, :, 1:].mean(axis=2)  # (epochs, val_samples)
    per_epoch_mean = per_epoch_per_sample.mean(axis=1)      # (epochs,)
    best_e = int(np.argmax(per_epoch_mean))
    best_mean = float(per_epoch_mean[best_e])
    std_at_best = float(per_epoch_per_sample[best_e].std())
    return best_mean, std_at_best, best_e
 
 
def main(args: argparse.Namespace) -> None:
    labels, means, stds, best_epochs = [], [], [], []
 
    for spec in args.run:
        label, path_str = spec.split(":", 1)
        dice_val = np.load(Path(path_str) / "dice_val.npy")
        mean, std, best_e = best_epoch_mean_and_std(dice_val)
        labels.append(label)
        means.append(mean)
        stds.append(std)
        best_epochs.append(best_e)
        print(f"{label:25s}: best epoch {best_e:3d}, mean Dice (4 organs) = {mean:.4f} (+/- {std:.4f} across val patients)")
 
    # Sort descending by mean Dice for the classic ablation-ranking look.
    order = np.argsort(means)[::-1]
    labels = [labels[i] for i in order]
    means = [means[i] for i in order]
    stds = [stds[i] for i in order]
 
    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(labels)), 5.5))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=4, color="tab:blue", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Best-epoch mean validation Dice\n(4 organs, background excluded)")
    ax.set_ylim(0, 1)
    # The error-bar explanation lives directly in the title (not a separate
    # floating caption) so it can't be missed or accidentally clipped --
    # titles are always included in the saved figure's layout.
    ax.set_title("Preprocessing ablation: best-epoch mean Dice per experiment\n"
                "Error bars: std across validation patients at that epoch ",
                fontsize=11)
    ax.grid(axis="y", alpha=0.3)
 
    fig.tight_layout()
    fig.savefig(args.dest, bbox_inches="tight")
    print(f"\nSaved summary bar chart to {args.dest}")
 
    if not args.headless:
        plt.show()
 
 
def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablation summary bar chart across multiple training runs")
    parser.add_argument("--run", action="append", required=True,
                        help="label:path_to_results_folder -- repeat once per run to compare")
    parser.add_argument("--dest", type=str, default="ablation_summary.png")
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()
 
 
if __name__ == "__main__":
    main(get_args())
 