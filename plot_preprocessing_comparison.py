"""Compare SEGTHOR preprocessing methods across split seeds 42, 43 and 44.

Compares Baseline, Windowed HU [-1000, 1000] and Percentile preprocessing.
Expected folders are results/SEGTHOR_<method>_seed<seed> for seeds 42, 43, 44.
Each run should contain dice_val.npy, dice_tra.npy, loss_val.npy, loss_tra.npy
and metrics/summary.csv. Training seed is assumed fixed at 0.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SEEDS = (42, 43, 44)
METHODS = {"Baseline": "baseline", "Windowed": "windowed", "Percentile": "percentile"}
ORDER = list(METHODS.keys())
ORGANS = ("Aorta", "Heart", "Trachea", "Esophagus")
SUMMARY_ORGANS = ("aorta", "heart", "trachea", "esophagus")
COLORS = dict(zip(ORDER, sns.color_palette("colorblind", n_colors=len(ORDER))))


def result_folder(root: Path, seed: int, method: str) -> Path:
    return root / f"SEGTHOR_{METHODS[method]}_seed{seed}"


def read_history(path: Path, kind: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Make sure this training run has finished.")

    arr = np.asarray(np.load(path, allow_pickle=False), dtype=float)

    if kind == "loss":
        if arr.ndim == 2 and arr.shape[1] > 1:
            arr = arr.mean(axis=1)
        else:
            arr = np.squeeze(arr)
        if arr.ndim != 1:
            raise ValueError(f"Expected [epochs] or [epochs, batches] loss in {path}; shape={arr.shape}")

    elif kind == "dice":
        if arr.ndim == 1:
            raise ValueError(f"{path} has shape {arr.shape}: per-organ Dice is unavailable.")
        if arr.ndim == 3:
            if arr.shape[-1] not in (4, 5):
                raise ValueError(f"Expected classes on the last axis in {path}; shape={arr.shape}")
            arr = arr.mean(axis=1)
        if arr.ndim != 2:
            raise ValueError(f"Expected [epochs, classes] or [epochs, slices, classes] Dice in {path}; shape={arr.shape}")
        if arr.shape[1] not in (4, 5) and arr.shape[0] in (4, 5):
            arr = arr.T
        if arr.shape[1] == 5:
            arr = arr[:, 1:]
        elif arr.shape[1] != 4:
            raise ValueError(f"Expected 4 organs or background + 4 organs in {path}; shape={arr.shape}")
    else:
        raise ValueError(f"Unknown history kind: {kind}")

    if arr.shape[0] == 0:
        raise ValueError(f"Empty history: {path}")
    if not np.isfinite(arr).all():
        raise ValueError(f"Non-finite values in {path}; inspect the training run.")
    if kind == "dice" and (np.any(arr < -1e-6) or np.any(arr > 1 + 1e-6)):
        raise ValueError(f"Dice values outside [0, 1] in {path}; inspect class order/shape.")

    return arr


def read_summary(folder: Path) -> pd.DataFrame:
    path = folder / "metrics" / "summary.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run evalData first for this result folder.")

    df = pd.read_csv(path, index_col=0)
    df.index = [str(v).strip().lower() for v in df.index]
    df.columns = [str(c).strip().lower() for c in df.columns]

    expected_metrics = {"dsc", "hd95", "assd", "fpslicerate"}
    expected_columns = {"background", "aorta", "heart", "trachea", "esophagus", "combined"}

    missing_metrics = expected_metrics - set(df.index)
    missing_columns = expected_columns - set(df.columns)

    if missing_metrics:
        raise ValueError(f"{path} is missing metrics: {sorted(missing_metrics)}; found {list(df.index)}")
    if missing_columns:
        raise ValueError(f"{path} is missing columns: {sorted(missing_columns)}; found {list(df.columns)}")

    return df


def load_results(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    histories = []
    best_rows = []

    for seed in SEEDS:
        for method in ORDER:
            folder = result_folder(root, seed, method)
            if not folder.is_dir():
                raise FileNotFoundError(f"Missing result folder: {folder}")

            dice_val = read_history(folder / "dice_val.npy", "dice")
            dice_tra = read_history(folder / "dice_tra.npy", "dice")
            loss_val = read_history(folder / "loss_val.npy", "loss")
            loss_tra = read_history(folder / "loss_tra.npy", "loss")

            lengths = {len(dice_val), len(dice_tra), len(loss_val), len(loss_tra)}
            if len(lengths) != 1:
                raise ValueError(f"Unequal history lengths in {folder}: {sorted(lengths)}")

            overall = dice_val.mean(axis=1)
            best_epoch = int(np.argmax(overall))
            summary = read_summary(folder)

            best_rows.append({
                "split_seed": seed,
                "training_seed": 0,
                "method": method,
                "best_epoch": best_epoch + 1,
                "overall_slice_dice": overall[best_epoch],
                **{organ: dice_val[best_epoch, i] for i, organ in enumerate(ORGANS)},
                "volume_dsc": float(summary.loc["dsc", "combined"]),
                "hd95": float(summary.loc["hd95", "combined"]),
                "assd": float(summary.loc["assd", "combined"]),
                "fpSliceRate": float(summary.loc["fpslicerate", "combined"]),
                **{
                    f"volume_dsc_{organ}": float(summary.loc["dsc", organ])
                    for organ in SUMMARY_ORGANS
                },
            })

            for epoch in range(len(dice_val)):
                row = {
                    "split_seed": seed,
                    "training_seed": 0,
                    "method": method,
                    "epoch": epoch + 1,
                    "overall_val_dice": overall[epoch],
                    "overall_train_dice": dice_tra[epoch].mean(),
                    "val_loss": loss_val[epoch],
                    "train_loss": loss_tra[epoch],
                }
                row.update({f"{organ}_val_dice": dice_val[epoch, i] for i, organ in enumerate(ORGANS)})
                histories.append(row)

            print(
                f"seed {seed:<2}  {method:<10}  {len(dice_val):>3} epochs  "
                f"best={best_epoch + 1:>3}  slice Dice={overall[best_epoch]:.4f}  "
                f"volume DSC={float(summary.loc['dsc', 'combined']):.4f}"
            )

    return pd.DataFrame(histories), pd.DataFrame(best_rows)


def curve(ax, data: pd.DataFrame, metric: str, ylabel: str, title: str) -> None:
    for method in ORDER:
        subset = data[data.method == method]
        stats = subset.groupby("epoch")[metric].agg(["mean", "std", "count"])
        x = stats.index.to_numpy(dtype=float)
        mean = stats["mean"].to_numpy(dtype=float)
        sd = stats["std"].fillna(0).to_numpy(dtype=float)

        ax.plot(x, mean, color=COLORS[method], lw=2.5, label=method)
        ax.fill_between(
            x,
            mean - sd,
            mean + sd,
            color=COLORS[method],
            alpha=0.14,
            where=stats["count"].to_numpy() >= 2,
        )

    ax.set(title=title, xlabel="Epoch", ylabel=ylabel)
    ax.grid(alpha=0.2)


def save_figure(fig, path: Path) -> None:
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def add_bar_labels(ax, bars, values, decimals: int) -> None:
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.{decimals}f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def make_plots(histories: pd.DataFrame, best: pd.DataFrame, out: Path) -> None:
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)
    out.mkdir(parents=True, exist_ok=True)

    histories.to_csv(out / "epoch_metrics.csv", index=False)
    best.to_csv(out / "best_epoch_per_run.csv", index=False)

    dice_metrics = ["overall_slice_dice", *ORGANS]
    baseline = best[best.method == "Baseline"].set_index("split_seed")
    deltas = best.copy()

    for metric in dice_metrics:
        deltas[f"delta_{metric}"] = deltas.apply(
            lambda row: row[metric] - baseline.loc[row.split_seed, metric],
            axis=1,
        )

    deltas.to_csv(out / "paired_baseline_differences.csv", index=False)

    # 01: Best slice Dice overview
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    seed_colors = sns.color_palette("colorblind", n_colors=len(SEEDS))
    offsets = (-0.14, 0.0, 0.14)

    for j, method in enumerate(ORDER):
        subset = best[best.method == method].set_index("split_seed").loc[list(SEEDS)]
        values = subset["overall_slice_dice"].to_numpy()
        mean = values.mean()
        sd = values.std(ddof=1)

        ax.errorbar(mean, j, xerr=sd, fmt="D", ms=7, color="#202020", capsize=5, lw=1.7, zorder=4)
        for k, value in enumerate(values):
            ax.scatter(
                value,
                j + offsets[k],
                s=55,
                color=seed_colors[k],
                edgecolor="white",
                linewidth=0.7,
                zorder=3,
                label=f"Seed {SEEDS[k]}" if j == 0 else None,
            )

    ax.set_yticks(range(len(ORDER)), ORDER)
    ax.invert_yaxis()
    all_scores = best["overall_slice_dice"].to_numpy()
    ax.set_xlim(max(0, all_scores.min() - 0.025), min(1, all_scores.max() + 0.025))
    ax.set(
        xlabel="Mean validation slice Dice (4 organs)",
        title="Best epoch per preprocessing method | diamond = mean ± SD",
    )
    ax.legend(title="Split seed", bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
    fig.tight_layout()
    save_figure(fig, out / "01_best_slice_dice_overview.png")

    # 02: Paired Dice differences vs Baseline
    difference = (
        deltas[deltas.method != "Baseline"]
        .groupby("method")[[f"delta_{metric}" for metric in dice_metrics]]
        .mean()
        .reindex(ORDER[1:])
    )
    matrix = difference.to_numpy()
    scale = max(0.01, np.abs(matrix).max())

    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="RdBu",
        center=0,
        vmin=-scale,
        vmax=scale,
        annot=True,
        fmt="+.3f",
        linewidths=1.5,
        linecolor="white",
        xticklabels=["Overall", *ORGANS],
        yticklabels=ORDER[1:],
        cbar_kws={"label": "Dice difference vs paired baseline"},
    )
    ax.set(
        title="Average Dice change versus Baseline on the same split seed (n = 3)",
        xlabel="",
        ylabel="",
    )
    ax.tick_params(axis="x", labelrotation=0)
    ax.tick_params(axis="y", labelrotation=0)
    fig.tight_layout()
    save_figure(fig, out / "02_paired_dice_differences.png")

    # 03: Dice learning curves
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), sharex=True)
    metrics = ["overall_val_dice", *[f"{organ}_val_dice" for organ in ORGANS]]
    titles = ["Overall", *ORGANS]

    for ax, metric, title in zip(axes.flat, metrics, titles):
        curve(ax, histories, metric, "Mean slice Dice", title)
        ax.set_ylim(0, 1.02)

    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Validation Dice learning curves | mean ± SD across split seeds 42, 43 and 44")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    save_figure(fig, out / "03_dice_learning_curves.png")

    # 04: Loss learning curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    later = histories[histories.epoch >= 3]

    for ax, metric, title in zip(
        axes,
        ("val_loss", "train_loss"),
        ("Validation loss", "Training loss"),
    ):
        curve(ax, later, metric, "Cross-entropy loss", title)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Loss from epoch 3 | mean ± SD across split seeds 42, 43 and 44")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    save_figure(fig, out / "04_loss_learning_curves.png")

    # 05-07: The four evaluation metrics next to each other, one figure per seed
    evaluation_metrics = (
        ("volume_dsc", "DSC", "Higher is better", 0, 1, 3),
        ("hd95", "HD95", "Lower is better", None, None, 2),
        ("assd", "ASSD", "Lower is better", None, None, 2),
        ("fpSliceRate", "False-positive slice rate", "Lower is better", 0, 1, 3),
    )

    for plot_number, seed in enumerate(SEEDS, start=5):
        seed_data = best[best.split_seed == seed].set_index("method").loc[ORDER]
        fig, axes = plt.subplots(1, 4, figsize=(19, 5))

        for ax, (column, title, direction, ymin, ymax, decimals) in zip(axes, evaluation_metrics):
            values = seed_data[column].to_numpy(dtype=float)
            bars = ax.bar(ORDER, values, color=[COLORS[method] for method in ORDER])
            ax.set_title(f"{title}\n{direction}")
            if ymin is not None:
                ax.set_ylim(bottom=ymin)
            if ymax is not None:
                ax.set_ylim(top=ymax)
            add_bar_labels(ax, bars, values, decimals)
            ax.tick_params(axis="x", labelrotation=20)

        fig.suptitle(f"Preprocessing comparison - split seed {seed} (training seed 0)")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        save_figure(fig, out / f"{plot_number:02d}_all_metrics_seed_{seed}.png")

    # 08: Volume DSC per organ for each seed
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    display_organs = ("Aorta", "Heart", "Trachea", "Esophagus")
    x = np.arange(len(display_organs))
    width = 0.24

    for ax, seed in zip(axes, SEEDS):
        seed_data = best[best.split_seed == seed].set_index("method").loc[ORDER]
        for i, method in enumerate(ORDER):
            values = [
                seed_data.loc[method, f"volume_dsc_{organ.lower()}"]
                for organ in display_organs
            ]
            ax.bar(
                x + (i - 1) * width,
                values,
                width,
                label=method,
                color=COLORS[method],
            )

        ax.set_xticks(x, display_organs, rotation=20)
        ax.set_ylim(0, 1)
        ax.set_title(f"Seed {seed}")
        ax.set_xlabel("Organ")

    axes[0].set_ylabel("Volume DSC")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Volume DSC per organ")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    save_figure(fig, out / "08_volume_dsc_per_organ.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Parent results folder containing SEGTHOR_baseline_seed42 etc. (default: results beside this script)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where PNG plots and CSV tables are saved (default: results/preprocessing_comparison/plots/overview)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    results_dir = args.results.expanduser() if args.results else project_root / "results"
    output_dir = (
        args.output.expanduser()
        if args.output
        else results_dir / "preprocessing_comparison" / "plots" / "overview"
    )

    histories, best = load_results(results_dir)
    make_plots(histories, best, output_dir)

    print("\nFinished.")
    print("SD is calculated across split seeds 42, 43 and 44.")
    print("Training seed is assumed to be fixed at 0 for every run.")
    print("Best epoch is selected using mean validation slice Dice over the four foreground organs.")
    print("DSC, HD95, ASSD and fpSliceRate are read from metrics/summary.csv after stitching.")


if __name__ == "__main__":
    main()
