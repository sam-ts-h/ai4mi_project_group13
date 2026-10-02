#!/usr/bin/env python3
"""
Compare validation Dice across multiple training runs (e.g. your 8 ablation
combos), one subplot per class INCLUDING background -- unlike plot.py, which
only ever plots one run at a time and skips background.

Usage:
    $ python compare_dice.py \
        --run original_baseline:results/original_baseline \
        --run none:results/none --run clip:results/clip --run resample:results/resample \
        --run normalize:results/normalize --run clip_resample:results/clip_resample \
        --run clip_normalize:results/clip_normalize --run resample_normalize:results/resample_normalize \
        --run full:results/full \
        --dest compare_dice.png

Each --run is "label:path_to_results_folder" (the folder passed as --dest to
main.py -- this script looks for dice_val.npy inside it).
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

CLASS_NAMES = ["background", "esophagus", "heart", "trachea", "aorta"]


def load_run(path: Path) -> np.ndarray:
    """Returns dice_val.npy's array, shape (epochs, val_samples, K)."""
    return np.load(path / "dice_val.npy")


def main(args: argparse.Namespace) -> None:
    runs = []  # (label, array)
    for spec in args.run:
        label, path_str = spec.split(":", 1)
        arr = load_run(Path(path_str))
        runs.append((label, arr))
        print(f"Loaded '{label}' from {path_str}: shape {arr.shape}")

    K = runs[0][1].shape[2]
    if K != len(CLASS_NAMES):
        print(f"NOTE: K={K} in the data doesn't match the {len(CLASS_NAMES)} "
              f"hardcoded CLASS_NAMES -- using generic 'class {{k}}' labels instead.")
        class_labels = [f"class {k}" for k in range(K)]
    else:
        class_labels = CLASS_NAMES

    fig, axes = plt.subplots(1, K, figsize=(4 * K, 5), sharey=True)
    if K == 1:
        axes = [axes]

    for k, ax in enumerate(axes):
        for label, arr in runs:
            # Mean Dice across validation samples, per epoch, for this class.
            per_epoch_mean = arr[:, :, k].mean(axis=1)
            ax.plot(per_epoch_mean, label=label, linewidth=1.5)
        ax.set_title(class_labels[k])
        ax.set_xlabel("epoch")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        # sharey=True hides tick LABELS on all but the leftmost subplot by
        # default -- force them visible on every subplot so each one is
        # readable on its own, not just the first.
        ax.tick_params(labelleft=True)

    axes[0].set_ylabel("mean validation Dice")
    # A shared legend below the whole figure, rather than crammed inside one
    # subplot -- scales much better once there are many runs to compare
    # (e.g. 9 ablation combos) than a per-axes corner legend would. Reserve
    # explicit vertical space for it via tight_layout's rect BEFORE placing
    # it, so it can't overlap the "epoch" x-axis labels above it.
    fig.tight_layout(rect=[0, 0.14, 1, 0.94])
    handles, run_labels = axes[0].get_legend_handles_labels()
    n_cols = min(len(run_labels), 5)
    fig.legend(handles, run_labels, loc="lower center", ncol=n_cols,
              bbox_to_anchor=(0.5, 0.0), fontsize=8)
    fig.suptitle("Validation Dice per class, across preprocessing ablations")
    fig.savefig(args.dest, bbox_inches="tight")
    print(f"Saved comparison figure to {args.dest}")

    if not args.headless:
        plt.show()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare per-class Dice across multiple training runs")
    parser.add_argument("--run", action="append", required=True,
                        help="label:path_to_results_folder -- repeat once per run to compare")
    parser.add_argument("--dest", type=str, default="compare_dice.png",
                        help="Output PNG path for the comparison figure")
    parser.add_argument("--headless", action="store_true",
                        help="Don't open an interactive display window, just save the PNG")
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())