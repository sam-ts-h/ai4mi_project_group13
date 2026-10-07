from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


SPLITS = (
    "split42_train42",
    "split43_train43",
    "split44_train44",
)

EXPERIMENTS = {
    "C01_context1": "Baseline",
    "C02_context5": "Context 5",
}
ORGANS = ("Esophagus", "Heart", "Trachea", "Aorta")
ORDER = list(EXPERIMENTS.values())
COLORS = dict(zip(ORDER, sns.color_palette("colorblind", n_colors=len(ORDER))))


def read_history(path: Path, kind: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Check whether rsync copied this file.")
    arr = np.asarray(np.load(path, allow_pickle=False), dtype=float)
    if kind == "loss":
        # Some training scripts store a value for every batch within each epoch.
        if arr.ndim == 2 and arr.shape[1] > 1:
            arr = arr.mean(axis=1)
        else:
            arr = np.squeeze(arr)
        if arr.ndim != 1:
            raise ValueError(f"Expected [epochs] or [epochs, batches] loss in {path}; shape={arr.shape}")
    else:
        if arr.ndim == 1:
            raise ValueError(
                f"{path} has shape {arr.shape}: per-organ Dice is unavailable. "
                "Please share the output of: python -c 'import numpy as np; "
                f"print(np.load(\"{path}\").shape)'"
            )
        if arr.ndim == 3:
            # Your files: [epochs, slices, classes]. Keep each epoch separate;
            # the plotted Dice is the arithmetic mean of the stored slice scores.
            if arr.shape[-1] not in (4, 5):
                raise ValueError(f"Expected classes on the last axis in {path}; shape={arr.shape}")
            arr = arr.mean(axis=1)
        if arr.ndim != 2:
            raise ValueError(f"Expected [epochs, classes] or [epochs, slices, classes] Dice in {path}; shape={arr.shape}")
        if arr.shape[1] not in (4, 5) and arr.shape[0] in (4, 5):
            arr = arr.T
        if arr.shape[1] == 5:
            arr = arr[:, 1:]  # SEGTHOR: background, esophagus, heart, trachea, aorta
        elif arr.shape[1] != 4:
            raise ValueError(f"Expected 4 organs or background + 4 organs in {path}; shape={arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError(f"Empty history: {path}")
    if not np.isfinite(arr).all():
        raise ValueError(f"Non-finite values in {path}; inspect the training run.")
    if kind == "dice" and (np.any(arr < -1e-6) or np.any(arr > 1 + 1e-6)):
        raise ValueError(f"Dice values outside [0, 1] in {path}; inspect class order/shape.")
    return arr


def load_results(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    histories = []
    best_rows = []
    for split in SPLITS:
        split_dir = root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing {split_dir}; use --results to point to its parent.")
        for prefix, experiment in EXPERIMENTS.items():
            matches = [split_dir / prefix]
            if not matches[0].is_dir():
                raise FileNotFoundError(f"Missing experiment folder {matches[0]}")
            if len(matches) != 1:
                raise ValueError(f"Expected one {prefix}_* folder in {split_dir}, found {len(matches)}")
            folder = matches[0]
            dice_val = read_history(folder / "dice_val.npy", "dice")
            dice_tra = read_history(folder / "dice_tra.npy", "dice")
            loss_val = read_history(folder / "loss_val.npy", "loss")
            loss_tra = read_history(folder / "loss_tra.npy", "loss")
            lengths = {len(dice_val), len(dice_tra), len(loss_val), len(loss_tra)}
            if len(lengths) != 1:
                raise ValueError(f"Unequal history lengths in {folder}: {sorted(lengths)}")

            overall = dice_val.mean(axis=1)  # macro average, excluding background
            best = int(np.argmax(overall))  # choose once, then report all organs at this epoch
            best_rows.append({
                "split_train_seed": split, "experiment": experiment,
                "best_epoch": best + 1, "overall_dice": overall[best],
                **{organ: dice_val[best, i] for i, organ in enumerate(ORGANS)},
            })
            for epoch in range(len(dice_val)):
                row = {
                    "split_train_seed": split, "experiment": experiment,
                    "epoch": epoch + 1, "overall_val_dice": overall[epoch],
                    "overall_train_dice": dice_tra[epoch].mean(),
                    "val_loss": loss_val[epoch], "train_loss": loss_tra[epoch],
                }
                row.update({f"{organ}_val_dice": dice_val[epoch, i] for i, organ in enumerate(ORGANS)})
                histories.append(row)
            print(f"{split:>15}  {experiment:<12}  {len(dice_val):>3} epochs  best={best + 1:>3}  Dice={overall[best]:.4f}")
    return pd.DataFrame(histories), pd.DataFrame(best_rows)


def curve(ax, data: pd.DataFrame, metric: str, ylabel: str, title: str,
          chosen: tuple[str, ...]) -> None:
    """Show only mean and run-to-run SD for the selected methods."""
    for experiment in chosen:
        subset = data[data.experiment == experiment]
        stats = subset.groupby("epoch")[metric].agg(["mean", "std", "count"])
        x = stats.index.to_numpy(dtype=float)
        mean = stats["mean"].to_numpy(dtype=float)
        sd = stats["std"].fillna(0).to_numpy(dtype=float)
        ax.plot(x, mean, color=COLORS[experiment], lw=2.8, label=experiment)
        ax.fill_between(x, mean - sd, mean + sd, color=COLORS[experiment], alpha=0.16,
                        where=stats["count"].to_numpy() >= 2)
    ax.set(title=title, xlabel="Epoch", ylabel=ylabel)
    ax.grid(alpha=0.2)


def save_figure(fig, path: Path) -> None:
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def make_plots(histories: pd.DataFrame, best: pd.DataFrame, out: Path) -> None:
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.08)
    out.mkdir(parents=True, exist_ok=True)
    histories.to_csv(out / "epoch_metrics.csv", index=False)
    best.to_csv(out / "best_epoch_per_run.csv", index=False)

    metrics = ["overall_dice", *ORGANS]
    base = best[best.experiment == "Baseline"].set_index("split_train_seed")
    deltas = best.copy()
    for metric in metrics:
        deltas[f"delta_{metric}"] = deltas.apply(
            lambda row: row[metric] - base.loc[row.split_train_seed, metric], axis=1)
    deltas.to_csv(out / "paired_baseline_differences.csv", index=False)

    # Overview: each seed pair gets a dot. Mean and sample SD are black.
    fig, ax = plt.subplots(figsize=(10, 5.5))
    seed_colors = sns.color_palette("colorblind", n_colors=3)
    offsets = (-0.17, 0, 0.17)
    for j, exp in enumerate(ORDER):
        subset = best[best.experiment == exp].set_index("split_train_seed").loc[list(SPLITS)]
        values = subset["overall_dice"].to_numpy()
        mean, sd = values.mean(), values.std(ddof=1)
        ax.errorbar(mean, j, xerr=sd, fmt="D", ms=7, color="#202020",
                    capsize=5, lw=1.7, zorder=4)
        for k, value in enumerate(values):
            ax.scatter(value, j + offsets[k], s=52, color=seed_colors[k],
                       edgecolor="white", linewidth=0.7, zorder=3,
                       label=SPLITS[k] if j == 0 else None)
    ax.set_yticks(range(len(ORDER)), ORDER)
    ax.invert_yaxis()
    all_scores = best.overall_dice.to_numpy()
    ax.set_xlim(max(0, all_scores.min() - 0.025), min(1, all_scores.max() + 0.025))
    ax.set(xlabel="Mean validation slice Dice (4 organs)",
           title="Best epoch of each run | diamonds = mean ± SD")
    ax.legend(title="Split / train seed", bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
    fig.tight_layout()
    save_figure(fig, out / "01_best_dice_overview.png")

    # Paired comparison removes the shared difficulty of each split.
    difference = (deltas[deltas.experiment != "Baseline"]
                  .groupby("experiment")[[f"delta_{m}" for m in metrics]]
                  .mean().reindex(ORDER[1:]))
    matrix = difference.to_numpy()
    scale = max(0.01, np.abs(matrix).max())
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(matrix, ax=ax, cmap="RdBu", center=0, vmin=-scale, vmax=scale,
                annot=True, fmt="+.3f", linewidths=1.5, linecolor="white",
                xticklabels=["Overall", *ORGANS], yticklabels=ORDER[1:],
                cbar_kws={"label": "Dice difference vs paired baseline (context=1)"})
    ax.set(title="Average change versus context_size=1 on the same split (n = 3)",
           xlabel="", ylabel="")
    ax.tick_params(axis="x", labelrotation=0)
    ax.tick_params(axis="y", labelrotation=0)
    fig.tight_layout()
    save_figure(fig, out / "02_paired_differences.png")

    # The complete 15-run histories remain in CSV; show two lines per chart.
    winner = best[best.experiment != "Baseline"].groupby("experiment")["overall_dice"].mean().idxmax()
    chosen = ("Baseline", winner)
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), sharex=True)
    for ax, metric, title in zip(axes.flat, ["overall_val_dice", *[f"{o}_val_dice" for o in ORGANS]],
                                 ["Overall", *ORGANS]):
        curve(ax, histories, metric, "Mean slice Dice", title, chosen)
        ax.set_ylim(0, 1.02)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle(f"Learning curves: context=1 (2D) versus {winner} | mean ± SD of 3 runs")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    save_figure(fig, out / "03_dice_learning_curves.png")

    # Epoch 1 has a large loss spike; show epochs 3+ so later changes are visible.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    later = histories[histories.epoch >= 3]
    for ax, metric, title in zip(axes, ("val_loss", "train_loss"),
                                 ("Validation loss", "Training loss")):
        curve(ax, later, metric, "Cross-entropy loss", title, chosen)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle(f"Loss from epoch 3: context=1 (2D) versus {winner} | mean ± SD")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    save_figure(fig, out / "04_loss_learning_curves.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=None,
                        help="Parent of the three split*_train* folders (default: results/context_size_experiment beside this script)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Where PNG plots and CSV tables are saved (default: results/context_size_experiment/plots/overview)")
    args = parser.parse_args()
    results_dir = args.results.expanduser() if args.results else Path(__file__).resolve().parent / "results" / "context_size_experiment"
    output_dir = args.output.expanduser() if args.output else results_dir / "plots" / "overview"
    histories, best = load_results(results_dir)
    make_plots(histories, best, output_dir)
    print("\nSD is over three paired split/training seed combinations, not independent training seeds.")
    print("Dice is averaged across stored slices, then across four organs; background is excluded.")


if __name__ == "__main__":
    main()