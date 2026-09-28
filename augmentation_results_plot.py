from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ORGAN_COLUMNS = {
    "Esophagus_dice": "Esophagus",
    "Heart_dice": "Heart",
    "Trachea_dice": "Trachea",
    "Aorta_dice": "Aorta",
}

EXPERIMENT_NAMES = {
    "B00_baseline": "Baseline",
    "B01_rotation_p050": "Rotation (p=0.5)",
    "B02_translation_p050": "Translation (p=0.5)",
    "B03_scaling_p050": "Scaling (p=0.5)",
    "B04_combination_p050": "Combination (p=0.5)",
    "B05_combination_p100": "Combination (p=1.0)",
}

EXPERIMENT_ORDER = [
    "Baseline",
    "Rotation (p=0.5)",
    "Translation (p=0.5)",
    "Scaling (p=0.5)",
    "Combination (p=0.5)",
    "Combination (p=1.0)",
]

EXPERIMENT_PALETTE = {
    "Baseline": "#9E9E9E",
    "Rotation (p=0.5)": "#4C78A8",
    "Translation (p=0.5)": "#72B7B2",
    "Scaling (p=0.5)": "#F2CF5B",
    "Combination (p=0.5)": "#E45756",
    "Combination (p=1.0)": "#B279A2",
}


def add_bar_labels(ax, decimals=3):
    """Add numeric values to all bars."""
    for container in ax.containers:
        ax.bar_label(
            container,
            fmt=f"%.{decimals}f",
            fontsize=9,
            padding=3,
        )


def safe_filename(name):
    """Create a filename-friendly version of an organ name."""
    return name.lower().replace(" ", "_")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create separate Seaborn plots for each organ."
        )
    )

    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "results/augmentation_corrected16/"
            "comparison_with_baseline/"
            "comparison_summary.csv"
        ),
        help="Path to comparison_summary.csv.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/augmentation_corrected16/"
            "comparison_plots_separate"
        ),
        help="Directory in which the plots are saved.",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(
        style="whitegrid",
        context="talk",
        font_scale=0.85,
    )

    summary = pd.read_csv(args.summary)

    required_columns = [
        "experiment",
        *ORGAN_COLUMNS.keys(),
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in summary.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in summary CSV: {missing_columns}"
        )

    summary["Experiment"] = summary["experiment"].map(
        EXPERIMENT_NAMES
    )

    if summary["Experiment"].isna().any():
        unknown = summary.loc[
            summary["Experiment"].isna(),
            "experiment",
        ].tolist()

        raise ValueError(
            f"Unknown experiment names: {unknown}"
        )

    summary["Experiment"] = pd.Categorical(
        summary["Experiment"],
        categories=EXPERIMENT_ORDER,
        ordered=True,
    )

    summary = (
        summary
        .sort_values("Experiment")
        .reset_index(drop=True)
    )

    # Mean Dice over all four real organ classes
    summary["mean_organ_dice"] = summary[
        list(ORGAN_COLUMNS.keys())
    ].mean(axis=1)

    baseline_mean = summary.loc[
        summary["Experiment"] == "Baseline",
        "mean_organ_dice",
    ].iloc[0]

    summary["difference_vs_baseline"] = (
        summary["mean_organ_dice"] - baseline_mean
    )

    summary.to_csv(
        args.output_dir / "comparison_summary_all_organs.csv",
        index=False,
    )

    # Create two separate plots for every organ:
    # 1. absolute Dice
    # 2. difference from baseline
    for organ_column, organ_name in ORGAN_COLUMNS.items():
        filename = safe_filename(organ_name)

        organ_data = summary[
            ["Experiment", organ_column]
        ].copy()

        organ_data = organ_data.rename(
            columns={organ_column: "Dice"}
        )

        baseline_score = organ_data.loc[
            organ_data["Experiment"] == "Baseline",
            "Dice",
        ].iloc[0]

        # -----------------------------------------------------
        # Absolute Dice plot
        # -----------------------------------------------------
        plt.figure(figsize=(11, 6))

        ax = sns.barplot(
            data=organ_data,
            x="Experiment",
            y="Dice",
            hue="Experiment",
            order=EXPERIMENT_ORDER,
            palette=EXPERIMENT_PALETTE,
            legend=False,
            errorbar=None,
        )

        add_bar_labels(ax)

        ax.axhline(
            baseline_score,
            color="#555555",
            linestyle="--",
            linewidth=1.5,
            label=f"Baseline: {baseline_score:.3f}",
        )

        ax.set(
            title=f"{organ_name}: validation Dice",
            xlabel="Augmentation method",
            ylabel="Validation Dice",
            ylim=(0, 1),
        )

        ax.tick_params(
            axis="x",
            rotation=25,
        )

        ax.legend(
            frameon=False,
            loc="upper left",
        )

        sns.despine()
        plt.tight_layout()

        plt.savefig(
            args.output_dir
            / f"{filename}_dice.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        # -----------------------------------------------------
        # Difference from baseline plot
        # -----------------------------------------------------
        difference_data = organ_data[
            organ_data["Experiment"] != "Baseline"
        ].copy()

        difference_data["Experiment"] = (
            difference_data["Experiment"]
            .cat.remove_unused_categories()
        )

        difference_data["Difference vs baseline"] = (
            difference_data["Dice"] - baseline_score
        )

        difference_order = EXPERIMENT_ORDER[1:]

        difference_palette = {
            experiment: EXPERIMENT_PALETTE[experiment]
            for experiment in difference_order
        }

        largest_difference = (
            difference_data["Difference vs baseline"]
            .abs()
            .max()
        )

        y_limit = max(
            0.02,
            largest_difference * 1.40,
        )

        plt.figure(figsize=(11, 6))

        ax = sns.barplot(
            data=difference_data,
            x="Experiment",
            y="Difference vs baseline",
            hue="Experiment",
            order=difference_order,
            palette=difference_palette,
            legend=False,
            errorbar=None,
        )

        add_bar_labels(ax)

        ax.axhline(
            0,
            color="black",
            linewidth=1.2,
        )

        ax.set(
            title=(
                f"{organ_name}: Dice difference "
                "relative to baseline"
            ),
            xlabel="Augmentation method",
            ylabel="Dice difference vs baseline",
            ylim=(-y_limit, y_limit),
        )

        ax.tick_params(
            axis="x",
            rotation=25,
        )

        sns.despine()
        plt.tight_layout()

        plt.savefig(
            args.output_dir
            / f"{filename}_difference_vs_baseline.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

    print("Plots saved to:", args.output_dir)
    print()

    for path in sorted(args.output_dir.glob("*.png")):
        print("-", path.name)

    print()
    print(
        summary[
            [
                "Experiment",
                "mean_organ_dice",
                "difference_vs_baseline",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()