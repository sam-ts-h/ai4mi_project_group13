import csv
from pathlib import Path

import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path("experiments/split42_seed0/final")

METHODS = {
    "No post-processing": BASE_DIR / "metrics_before_lcc" / "summary.csv",
    "LCC": BASE_DIR / "metrics_after_lcc" / "summary.csv",
    "Small components": BASE_DIR / "metrics_after_sc" / "summary.csv",
}

METRICS = {
    "dsc": "3D Dice\nhigher is better",
    "hd95": "HD95 (mm)\nlower is better",
    "assd": "ASSD (mm)\nlower is better",
    "fpSliceRate": "False-positive slice rate\nlower is better",
}

OUTPUT_DIR = BASE_DIR / "plots_postprocessing"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "postprocessing_comparison.png"


# ============================================================
# READ SUMMARY.CSV
# ============================================================

def read_summary(path: Path):

    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}"
        )

    result = {}

    with open(path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            metric = row["metric"]

            if metric in METRICS:
                result[metric] = float(
                    row["combined"]
                )

    return result


# ============================================================
# LOAD RESULTS
# ============================================================

results = {}

for method, path in METHODS.items():
    results[method] = read_summary(path)


# Print values in terminal
print()
print("=" * 70)
print("POST-PROCESSING COMPARISON")
print("=" * 70)

for metric in METRICS:

    print(f"\n{metric}")

    for method in METHODS:
        print(
            f"{method:<22}: "
            f"{results[method][metric]:.4f}"
        )


# ============================================================
# PLOT
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(13, 8)
)

axes = axes.ravel()

method_names = list(METHODS.keys())


for ax, (metric, metric_label) in zip(
    axes,
    METRICS.items()
):

    values = [
        results[method][metric]
        for method in method_names
    ]

    # --------------------------------------------------------
    # ONE POINT PER POST-PROCESSING METHOD
    # --------------------------------------------------------

    for i, value in enumerate(values):

        ax.scatter(
            value,
            i,
            s=80,
            zorder=3
        )

        # Exact value next to point
        ax.annotate(
            f"{value:.4f}",
            (value, i),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=9
        )

    # --------------------------------------------------------
    # AXES
    # --------------------------------------------------------

    ax.set_yticks(
        range(len(method_names))
    )

    ax.set_yticklabels(
        method_names
    )

    ax.invert_yaxis()

    ax.set_xlabel(
        metric_label
    )

    ax.set_title(
        metric.upper()
        if metric != "fpSliceRate"
        else "False-positive slice rate"
    )

    ax.grid(
        True,
        color="lightgrey"
    )

    ax.set_axisbelow(True)


# ============================================================
# FINAL FIGURE
# ============================================================

fig.suptitle(
    "Post-processing comparison | split42_seed0",
    fontsize=15
)

fig.tight_layout(
    rect=[0, 0, 1, 0.95]
)

fig.savefig(
    OUTPUT_FILE,
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)

print()
print(f"Saved: {OUTPUT_FILE}")