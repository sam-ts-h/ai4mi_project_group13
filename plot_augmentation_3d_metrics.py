#!/usr/bin/env python3
"""Plot the patient-volume metrics for the 18 paired augmentation runs.

Run on the VU server after evaluate_augmentation_3d.py:
    python plot_augmentation_3d_metrics.py

The four metrics are computed at the epoch selected by the existing 2D
validation slice-Dice criterion. This script never selects an epoch itself.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


EXPERIMENTS = {
    "A00_baseline": "Baseline",
    "A01_rotation": "Rotation",
    "A02_translation": "Translation",
    "A03_scaling": "Scaling",
    "A04_noise": "Noise",
    "A05_combination": "Combination",
}
ORDER = list(EXPERIMENTS.values())
ORGANS = ("esophagus", "heart", "trachea", "aorta")
METRICS = ("dsc", "hd95", "assd", "fpSliceRate")
SPLITS = ((42, 0), (43, 1), (44, 2))


def load_results(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run evaluate_augmentation_3d.py first")
    data = pd.read_csv(path)
    required = {"split_seed", "train_seed", "experiment", "metric", "combined", *ORGANS}
    if missing := required - set(data.columns):
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")
    expected = {(split, seed, exp, metric)
                for split, seed in SPLITS for exp in EXPERIMENTS for metric in METRICS}
    observed = set(zip(data.split_seed, data.train_seed, data.experiment, data.metric))
    if observed != expected or len(data) != len(expected):
        raise ValueError(f"Expected exactly 18 runs x 4 metrics. Missing: {sorted(expected - observed)}; "
                         f"unexpected: {sorted(observed - expected)}; rows: {len(data)}")
    data["method"] = data.experiment.map(EXPERIMENTS)
    data["split"] = data.apply(lambda row: f"split{row.split_seed}_train{row.train_seed}", axis=1)
    for col in ("combined", *ORGANS):
        data[col] = pd.to_numeric(data[col], errors="raise")
    if data["combined"].isna().any():
        raise ValueError("A combined score is missing; inspect the per-run CSV")
    return data


def paired_differences(data):
    baseline = (data[data.method == "Baseline"]
                .set_index(["split_seed", "train_seed", "metric"]))
    rows = []
    for _, row in data.iterrows():
        original = baseline.loc[(row.split_seed, row.train_seed, row.metric)]
        record = {key: row[key] for key in
                  ("split_seed", "train_seed", "split", "experiment", "method", "metric")}
        for col in ("combined", *ORGANS):
            record[col] = row[col] - original[col]
        rows.append(record)
    return pd.DataFrame(rows)


def finish(fig, path):
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def dot_summary(ax, data, column, *, paired=False, xlabel="", title=""):
    """Three paired split points and their mean ± sample SD per method."""
    seed_colors = sns.color_palette("colorblind", n_colors=3)
    for j, method in enumerate(ORDER[1:] if paired else ORDER):
        part = data[data.method == method].set_index("split_seed").loc[[42, 43, 44]]
        values = part[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Missing {column} for {method}; inspect CSV")
        ax.errorbar(values.mean(), j, xerr=values.std(ddof=1), fmt="D", ms=7,
                    capsize=5, color="#202020", lw=1.7, zorder=4)
        for k, value in enumerate(values):
            ax.scatter(value, j + (-0.17, 0, 0.17)[k], s=51,
                       color=seed_colors[k], edgecolor="white", lw=0.7, zorder=3,
                       label=f"Split {42 + k} / train {k}" if j == 0 else None)
    labels = ORDER[1:] if paired else ORDER
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set(title=title, xlabel=xlabel)
    ax.grid(axis="x", alpha=0.22)
    if paired:
        ax.axvline(0, color="#555555", lw=1, ls="--")


def make_plots(data, out):
    out.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.06)
    delta = paired_differences(data)
    delta.to_csv(out / "paired_baseline_differences_3d.csv", index=False)

    dice = data[data.metric == "dsc"]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    dot_summary(ax, dice, "combined", xlabel="3D Dice on 256×256 volumes (higher is better)",
                title="Four-organ 3D Dice at each run's selected epoch | mean ± SD")
    ax.set_xlim(0, 1)
    ax.legend(title="Paired split / training seed", bbox_to_anchor=(1.01, 1),
              loc="upper left", frameon=False)
    fig.tight_layout()
    finish(fig, out / "01_3d_dice_overview.png")

    changes = delta[(delta.metric == "dsc") & (delta.method != "Baseline")]
    matrix = changes.groupby("method")[list(ORGANS)].mean().reindex(ORDER[1:])
    values = matrix.to_numpy()
    scale = max(0.01, np.nanmax(np.abs(values)))
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    sns.heatmap(matrix, ax=ax, cmap="RdBu", center=0, vmin=-scale, vmax=scale,
                annot=True, fmt="+.3f", linewidths=1.5, linecolor="white",
                cbar_kws={"label": "3D Dice difference vs paired baseline"})
    ax.set(title="Mean organ Dice change vs baseline on the same split (n = 3)",
           xlabel="", ylabel="")
    ax.set_xticklabels([o.capitalize() for o in ORGANS], rotation=0)
    ax.tick_params(axis="y", labelrotation=0)
    fig.tight_layout()
    finish(fig, out / "02_organ_dice_paired_differences.png")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for ax, metric, name in zip(axes, ("hd95", "assd"), ("HD95", "ASSD")):
        dot_summary(ax, delta[(delta.metric == metric) & (delta.method != "Baseline")],
                    "combined", paired=True, xlabel="Difference vs baseline (mm; lower is better)",
                    title=f"{name}: paired change | mean ± SD")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    finish(fig, out / "03_surface_distances_paired.png")

    fig, ax = plt.subplots(figsize=(10, 5.2))
    dot_summary(ax, delta[(delta.metric == "fpSliceRate") & (delta.method != "Baseline")],
                "combined", paired=True,
                xlabel="False-positive slice rate difference vs baseline (lower is better)",
                title="False-positive slice rate: paired change | mean ± SD")
    ax.legend(title="Paired split / training seed", bbox_to_anchor=(1.01, 1),
              loc="upper left", frameon=False)
    fig.tight_layout()
    finish(fig, out / "04_false_positive_slice_rate_paired.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path,
                        default=Path("/local/data/frk340/ai4mi_results/augmentation_full_training/metrics_3d_256_all.csv"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.summary.parent / "plots" / "3d_metrics"
    data = load_results(args.summary)
    make_plots(data, output)
    print("Each point is one paired split/training seed; SD describes three pairs.")
    print("Scores are computed on 256×256 patient volumes at the selected 2D Dice epoch.")


if __name__ == "__main__":
    main()
