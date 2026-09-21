from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ORGAN_COLUMNS = {
    "Esophagus_dice": "Esophagus",
    "Heart_dice": "Heart",
    "Trachea_dice": "Trachea",
}

EXPERIMENT_NAMES = {
    "A00_baseline": "Baseline",
    "A01_rotation": "Rotation",
    "A02_translation": "Translation",
    "A03_scaling": "Scaling",
    "A04_combination": "Combination",
}

EXPERIMENT_ORDER = [
    "Baseline",
    "Rotation",
    "Translation",
    "Scaling",
    "Combination",
]


def add_bar_labels(ax, decimals=3):
    """Add numeric values above all bars."""
    for container in ax.containers:
        ax.bar_label(
            container,
            fmt=f"%.{decimals}f",
            fontsize=9,
            padding=3,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Create Seaborn plots for augmentation experiments."
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "results/augmentation/comparison_with_baseline/comparison_summary.csv"
        ),
        help="Path to comparison_summary.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/augmentation/comparison_plots"
        ),
        help="Directory in which the plots are saved.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Presentation-friendly Seaborn theme
    sns.set_theme(
        style="whitegrid",
        context="talk",
        font_scale=0.85,
    )

    summary = pd.read_csv(args.summary)

    # Check whether the required columns are available
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
            summary["Experiment"].isna(), "experiment"
        ].tolist()

        raise ValueError(
            f"Unknown experiment names: {unknown}"
        )

    summary["Experiment"] = pd.Categorical(
        summary["Experiment"],
        categories=EXPERIMENT_ORDER,
        ordered=True,
    )

    summary = summary.sort_values("Experiment").reset_index(
        drop=True
    )

    # Mean Dice over the classes that are actually annotated
    summary["mean_annotated_dice"] = summary[
        list(ORGAN_COLUMNS.keys())
    ].mean(axis=1)

    baseline_score = summary.loc[
        summary["Experiment"] == "Baseline",
        "mean_annotated_dice",
    ].iloc[0]

    summary["annotated_difference_vs_baseline"] = (
        summary["mean_annotated_dice"] - baseline_score
    )

    # Save an extended results table
    summary.to_csv(
        args.output_dir / "comparison_summary_annotated.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Plot 1: mean Dice over annotated organs
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))

    ax = sns.barplot(
        data=summary,
        x="Experiment",
        y="mean_annotated_dice",
        hue="Experiment",
        order=EXPERIMENT_ORDER,
        palette={
            "Baseline": "#9E9E9E",
            "Rotation": "#4C78A8",
            "Translation": "#72B7B2",
            "Scaling": "#F2CF5B",
            "Combination": "#E45756",
        },
        legend=False,
        errorbar=None,
    )

    add_bar_labels(ax)

    ax.set(
        title="Effect of data augmentation on segmentation performance",
        xlabel="Augmentation method",
        ylabel="Mean validation Dice",
        ylim=(0, 1),
    )

    ax.axhline(
        baseline_score,
        color="#555555",
        linestyle="--",
        linewidth=1.5,
        label=f"Baseline: {baseline_score:.3f}",
    )

    ax.legend(frameon=False, loc="upper left")
    sns.despine()

    plt.tight_layout()
    plt.savefig(
        args.output_dir / "augmentation_mean_dice.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # ---------------------------------------------------------
    # Plot 2: Dice per organ
    # ---------------------------------------------------------
    organ_data = summary.melt(
        id_vars="Experiment",
        value_vars=list(ORGAN_COLUMNS.keys()),
        var_name="Organ_column",
        value_name="Dice",
    )

    organ_data["Organ"] = organ_data["Organ_column"].map(
        ORGAN_COLUMNS
    )

    plt.figure(figsize=(12, 7))

    ax = sns.barplot(
        data=organ_data,
        x="Organ",
        y="Dice",
        hue="Experiment",
        hue_order=EXPERIMENT_ORDER,
        palette="colorblind",
        errorbar=None,
    )

    ax.set(
        title="Validation Dice by organ and augmentation method",
        xlabel="Organ",
        ylabel="Validation Dice",
        ylim=(0, 1),
    )

    ax.legend(
        title="Experiment",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        frameon=False,
    )

    sns.despine()
    plt.tight_layout()
    plt.savefig(
        args.output_dir / "augmentation_dice_per_organ.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # ---------------------------------------------------------
    # Plot 3: difference from baseline
    # ---------------------------------------------------------
    baseline_values = (
        organ_data[
            organ_data["Experiment"] == "Baseline"
        ]
        .set_index("Organ")["Dice"]
    )

    organ_data["Difference vs baseline"] = organ_data.apply(
        lambda row: (
            row["Dice"] - baseline_values[row["Organ"]]
        ),
        axis=1,
    )

    differences = (
        organ_data[
            organ_data["Experiment"] != "Baseline"
        ]
        .pivot(
            index="Experiment",
            columns="Organ",
            values="Difference vs baseline",
        )
        .reindex(EXPERIMENT_ORDER[1:])
    )

    plt.figure(figsize=(8, 5))

    ax = sns.heatmap(
        differences,
        annot=True,
        fmt="+.3f",
        cmap="RdYlGn",
        center=0,
        linewidths=0.5,
        cbar_kws={
            "label": "Dice difference vs baseline"
        },
    )

    ax.set(
        title="Change in validation Dice relative to baseline",
        xlabel="Organ",
        ylabel="Augmentation method",
    )

    plt.tight_layout()
    plt.savefig(
        args.output_dir
        / "augmentation_difference_heatmap.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print("Plots saved to:", args.output_dir)
    print()
    print(
        summary[
            [
                "Experiment",
                "mean_annotated_dice",
                "annotated_difference_vs_baseline",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()