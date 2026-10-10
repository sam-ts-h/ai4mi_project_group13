#!/usr/bin/env python3
"""
Generic dot plot: one metric on x, one row per experiment variant on y,
one coloured dot per seed plus a black diamond for the mean and a +/- 1 std bar.

Reads either a .npy log written by main.py (dice_val.npy, shape (epochs, N, K)) --
best epoch, mean over foreground classes -- or a .csv column.

The file location is a format string, so this works for any experiment layout.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_value(path: Path, metric: str) -> float:
    if path.suffix == ".npy":
        d = np.load(path)                      # (epochs, N, K)
        per_epoch = d.mean(axis=1)[:, 1:]      # drop background
        return float(per_epoch.mean(axis=1).max())
    return float(pd.read_csv(path)[metric].mean())


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=Path, required=True)
    p.add_argument("--pattern", default="{variant}/split{seed}_train{seed}/dice_val.npy",
                   help="Path under --data_dir, with {variant} and {seed} placeholders.")
    p.add_argument("--variants", nargs="+", required=True, help="Top to bottom.")
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--metric", default="DSC", help="Axis label, and CSV column if reading a csv.")
    p.add_argument("--out", type=Path, default=Path("dotplot.png"))
    return p.parse_args()


def main():
    args = get_args()
    labels = args.labels or args.variants

    values = []
    for variant in args.variants:
        row = []
        for seed in args.seeds:
            f = args.data_dir / args.pattern.format(variant=variant, seed=seed)
            if not f.exists():
                raise SystemExit(f"missing: {f}")
            row.append(load_value(f, args.metric))
        values.append(row)

    fig, ax = plt.subplots(figsize=(6.5, 1.0 + 0.9 * len(values)))

    for i, row in enumerate(values):
        y = len(values) - 1 - i                       # first variant on top
        for j, v in enumerate(row):
            ax.scatter(v, y + 0.18, color=f"C{j}", s=45, zorder=3,
                       label=f"seed {args.seeds[j]}" if i == 0 else None)
        ax.errorbar(np.mean(row), y, xerr=np.std(row), fmt="D",
                    color="black", capsize=4, markersize=6, zorder=4)

    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(labels[::-1])
    ax.set_ylim(-0.5, len(values) - 0.3)
    ax.set_xlabel(args.metric)
    ax.set_title(args.metric)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, frameon=False)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}\n")

    for label, row in zip(labels, values):
        print(f"{label:24s} {np.mean(row):.4f} +- {np.std(row):.4f}   "
              + "  ".join(f"{v:.4f}" for v in row))


if __name__ == "__main__":
    main()