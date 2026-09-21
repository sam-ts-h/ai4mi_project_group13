#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


CLASS_NAMES = ["Esophagus", "Heart", "Trachea", "Aorta"]


def load_results(folder: Path) -> dict:
    """Load the validation results of one experiment."""
    dice_path = folder / "dice_val.npy"
    loss_path = folder / "loss_val.npy"

    if not dice_path.exists():
        raise FileNotFoundError(f"Niet gevonden: {dice_path}")

    if not loss_path.exists():
        raise FileNotFoundError(f"Niet gevonden: {loss_path}")

    dice = np.load(dice_path)
    loss = np.load(loss_path)

    if dice.ndim != 3:
        raise ValueError(
            f"{dice_path} heeft shape {dice.shape}; "
            "verwacht wordt (epochs, slices, classes)."
        )

    if dice.shape[2] != 5:
        raise ValueError(
            f"{dice_path} bevat {dice.shape[2]} classes; "
            "voor SEGTHOR worden 5 classes verwacht."
        )

    number_of_epochs = min(len(dice), len(loss))
    dice = dice[:number_of_epochs]
    loss = loss[:number_of_epochs]

    # Niet-uitgevoerde epochs bevatten alleen nullen.
    completed_dice = np.any(dice[:, :, 1:] != 0, axis=(1, 2))
    completed_loss = np.any(
        loss.reshape(number_of_epochs, -1) != 0,
        axis=1,
    )
    completed = completed_dice | completed_loss

    if not np.any(completed):
        raise ValueError(f"Geen uitgevoerde epochs gevonden in {folder}")

    last_completed_epoch = np.where(completed)[0][-1] + 1

    dice = dice[:last_completed_epoch]
    loss = loss[:last_completed_epoch]

    # Achtergrondclass 0 wordt niet meegenomen.
    overall_dice = np.nanmean(dice[:, :, 1:], axis=(1, 2))
    class_dice = np.nanmean(dice[:, :, 1:], axis=1)
    mean_loss = np.nanmean(
        loss.reshape(last_completed_epoch, -1),
        axis=1,
    )

    best_epoch = int(np.nanargmax(overall_dice))

    return {
        "name": folder.name,
        "folder": folder,
        "overall_dice": overall_dice,
        "class_dice": class_dice,
        "loss": mean_loss,
        "best_epoch": best_epoch,
    }


def find_experiment_folders(
    baseline_folder: Path,
    selected_folders: list[Path] | None,
) -> list[Path]:
    """
    Find all experiment folders.

    If --experiments is not supplied, every sibling folder containing
    dice_val.npy is automatically included.
    """
    if selected_folders:
        experiment_folders = selected_folders
    else:
        experiment_folders = sorted(
        folder
        for folder in baseline_folder.parent.iterdir()
        if (
        folder.is_dir()
        and folder != baseline_folder
        and "_debug" not in folder.name.lower()
        and (folder / "dice_val.npy").exists()
    )
)

    if not experiment_folders:
        raise ValueError(
            f"Geen experimenten gevonden naast {baseline_folder}."
        )

    return experiment_folders


def save_summary(
    baseline: dict,
    experiments: list[dict],
    output_path: Path,
) -> None:
    """Save one row per experiment, always compared with the baseline."""
    baseline_epoch = baseline["best_epoch"]
    baseline_score = baseline["overall_dice"][baseline_epoch]
    baseline_class_scores = baseline["class_dice"][baseline_epoch]

    fieldnames = [
        "experiment",
        "best_epoch",
        "best_overall_dice",
        "overall_difference_vs_baseline",
        "loss_at_best_epoch",
    ]

    for class_name in CLASS_NAMES:
        fieldnames.extend(
            [
                f"{class_name}_dice",
                f"{class_name}_difference_vs_baseline",
            ]
        )

    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        all_runs = [baseline] + experiments

        for run in all_runs:
            best_epoch = run["best_epoch"]
            best_score = run["overall_dice"][best_epoch]
            class_scores = run["class_dice"][best_epoch]

            row = {
                "experiment": run["name"],
                "best_epoch": best_epoch,
                "best_overall_dice": best_score,
                "overall_difference_vs_baseline": (
                    best_score - baseline_score
                ),
                "loss_at_best_epoch": run["loss"][best_epoch],
            }

            for class_index, class_name in enumerate(CLASS_NAMES):
                row[f"{class_name}_dice"] = class_scores[class_index]
                row[f"{class_name}_difference_vs_baseline"] = (
                    class_scores[class_index]
                    - baseline_class_scores[class_index]
                )

            writer.writerow(row)


def plot_overview(
    baseline: dict,
    experiments: list[dict],
    output_path: Path,
) -> None:
    """Create an overview figure of all experiments."""
    all_runs = [baseline] + experiments

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    # Overall validation Dice per epoch
    for run in all_runs:
        epochs = np.arange(len(run["overall_dice"]))
        axes[0, 0].plot(
            epochs,
            run["overall_dice"],
            label=run["name"],
            linewidth=2,
        )

    axes[0, 0].set_title("Overall validation Dice")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Dice")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    # Validation loss per epoch
    for run in all_runs:
        epochs = np.arange(len(run["loss"]))
        axes[0, 1].plot(
            epochs,
            run["loss"],
            label=run["name"],
            linewidth=2,
        )

    axes[0, 1].set_title("Validation loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()

    # Best overall Dice
    run_names = [run["name"] for run in all_runs]
    best_scores = [
        run["overall_dice"][run["best_epoch"]]
        for run in all_runs
    ]

    colors = [
        "grey" if run is baseline else "steelblue"
        for run in all_runs
    ]

    bars = axes[1, 0].bar(run_names, best_scores, color=colors)

    axes[1, 0].set_title("Best overall validation Dice")
    axes[1, 0].set_ylabel("Dice")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].tick_params(axis="x", rotation=25)
    axes[1, 0].grid(axis="y", alpha=0.3)

    for bar, score in zip(bars, best_scores):
        axes[1, 0].text(
            bar.get_x() + bar.get_width() / 2,
            score + 0.01,
            f"{score:.3f}",
            ha="center",
        )

    # Dice per class at the best epoch
    x = np.arange(len(CLASS_NAMES))
    width = 0.8 / len(all_runs)

    for run_index, run in enumerate(all_runs):
        best_epoch = run["best_epoch"]
        offset = (run_index - (len(all_runs) - 1) / 2) * width

        axes[1, 1].bar(
            x + offset,
            run["class_dice"][best_epoch],
            width,
            label=f"{run['name']} (epoch {best_epoch})",
        )

    axes[1, 1].set_title("Dice per class at best epoch")
    axes[1, 1].set_ylabel("Dice")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(CLASS_NAMES)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].grid(axis="y", alpha=0.3)
    axes[1, 1].legend(fontsize=8)

    fig.suptitle("Augmentation experiments compared with baseline")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_class_curves(
    baseline: dict,
    experiments: list[dict],
    output_path: Path,
) -> None:
    """Plot the validation Dice per class for every run."""
    all_runs = [baseline] + experiments

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    for class_index, (axis, class_name) in enumerate(
        zip(axes.flat, CLASS_NAMES)
    ):
        for run in all_runs:
            epochs = np.arange(len(run["overall_dice"]))

            axis.plot(
                epochs,
                run["class_dice"][:, class_index],
                label=run["name"],
                linewidth=2,
            )

        axis.set_title(class_name)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Validation Dice")
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)

    fig.suptitle("Validation Dice per SEGTHOR class")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def print_summary(baseline: dict, experiments: list[dict]) -> None:
    """Print every experiment's difference from the baseline."""
    baseline_epoch = baseline["best_epoch"]
    baseline_score = baseline["overall_dice"][baseline_epoch]
    baseline_classes = baseline["class_dice"][baseline_epoch]

    print("\nBaseline")
    print("-" * 70)
    print(
        f"{baseline['name']}: epoch {baseline_epoch}, "
        f"overall Dice = {baseline_score:.4f}"
    )

    for run in experiments:
        best_epoch = run["best_epoch"]
        best_score = run["overall_dice"][best_epoch]
        difference = best_score - baseline_score

        print("\n" + run["name"])
        print("-" * 70)
        print(f"Beste epoch: {best_epoch}")
        print(f"Overall Dice: {best_score:.4f}")
        print(f"Verschil versus baseline: {difference:+.4f}")

        for class_index, class_name in enumerate(CLASS_NAMES):
            score = run["class_dice"][best_epoch, class_index]
            class_difference = score - baseline_classes[class_index]

            print(
                f"{class_name:10s}: {score:.4f} "
                f"({class_difference:+.4f} versus baseline)"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Vergelijk augmentation-experimenten met één baseline."
        )
    )

    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("results/augmentation/A00_baseline"),
        help="Map met de baseline-resultaten.",
    )

    parser.add_argument(
        "--experiments",
        type=Path,
        nargs="*",
        default=None,
        help=(
            "Optionele lijst met experimentmappen. Als dit argument wordt "
            "weggelaten, worden alle experimenten automatisch gevonden."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/augmentation/comparison_with_baseline"
        ),
    )

    args = parser.parse_args()

    experiment_folders = find_experiment_folders(
        args.baseline,
        args.experiments,
    )

    baseline = load_results(args.baseline)
    experiments = [
        load_results(folder)
        for folder in experiment_folders
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print_summary(baseline, experiments)

    save_summary(
        baseline,
        experiments,
        args.output_dir / "comparison_summary.csv",
    )

    plot_overview(
        baseline,
        experiments,
        args.output_dir / "comparison_overview.png",
    )

    plot_class_curves(
        baseline,
        experiments,
        args.output_dir / "comparison_class_curves.png",
    )

    print(f"\nResultaten opgeslagen in: {args.output_dir}")


if __name__ == "__main__":
    main()