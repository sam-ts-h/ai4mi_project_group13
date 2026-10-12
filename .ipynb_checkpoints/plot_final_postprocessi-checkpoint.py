from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


BASE = Path("results/final")

SEEDS = [42, 43, 44]

METHODS = {
    "Baseline": "metrics_pre_pp",
    "LCC": "metrics_post_pp_lcc",
    "SC 0.001": "metrics_sc_f001",
    "SC 0.003": "metrics_sc_f003",
    "SC 0.005": "metrics_sc_f005",
}

METRICS = {
    "dsc": "DSC",
    "hd95": "HD95 (mm)",
    "assd": "ASSD (mm)",
    "fpSliceRate": "FP Slice Rate",
}

OUTPUT = BASE / "final_postprocessing_comparison.png"


COLORS = {
    42: "#1f77b4",   # blue
    43: "#FFD700",   # yellow
    44: "#2ca02c",   # green
}

OFFSETS = {
    42: -0.15,
    43: 0.00,
    44: 0.15,
}


rows = []

for seed in SEEDS:

    print(f"\nSEED {seed}")

    for method, folder in METHODS.items():

        csv_path = (
            BASE
            / f"seed{seed}"
            / folder
            / "summary.csv"
        )

        if not csv_path.exists():
            print(f"MISSING: {csv_path}")
            continue

        print(f"Reading: {csv_path}")
        df = pd.read_csv(csv_path)

        for metric in METRICS:
            row = df[
                df["metric"] == metric]

            if row.empty:
                print(
                    f"Missing metric {metric} "
                    f"in {csv_path}"
                )
                continue

            value = float(row["combined"].iloc[0])

            rows.append({
                "seed": seed,
                "method": method,
                "metric": metric,
                "value": value,
            })


results = pd.DataFrame(rows)

if results.empty:
    raise RuntimeError(
        "No metric results found."
    )


# Print numerical overview
print()
print("=" * 110)
print("FINAL POST-PROCESSING COMPARISON")
print("=" * 110)


for metric, label in METRICS.items():

    print()
    print(label)
    print("-" * 110)

    metric_df = results[results["metric"] == metric]

    for method in METHODS:

        values = (metric_df[metric_df["method"] == method]
            .set_index("seed")
            .reindex(SEEDS)["value"]
            .to_numpy(dtype=float)
        )

        valid = values[np.isfinite(values)]

        if len(valid) == 0:
            continue

        mean = np.mean(valid)

        sd = (np.std(valid, ddof=1)
            if len(valid) > 1
            else np.nan)

        print(
            f"{method:<14} "
            f"seed42={values[0]:.4f}   "
            f"seed43={values[1]:.4f}   "
            f"seed44={values[2]:.4f}   "
            f"mean={mean:.4f}   "
            f"SD={sd:.4f}"
        )

# Plot
fig, axes = plt.subplots(
    2,
    2,
    figsize=(14, 9),
)

axes = axes.flatten()

methods = list(METHODS.keys())


for ax, (metric, label) in zip(axes,METRICS.items(),):

    metric_df = results[results["metric"] == metric]

    all_values = []

    for y, method in enumerate(methods):

        values = (metric_df[metric_df["method"] == method]
            .set_index("seed")
            .reindex(SEEDS)["value"]
            .to_numpy(dtype=float)
        )

        valid = values[np.isfinite(values)]

        if len(valid) == 0:
            continue

        all_values.extend(valid)
        mean = np.mean(valid)

        sd = (np.std(valid, ddof=1)
            if len(valid) > 1
            else 0.0)

        # Mean ± SD
        ax.errorbar(
            mean,
            y,
            xerr=sd,
            fmt="D",
            markersize=7,
            color="black",
            capsize=5,
            linewidth=1.5,
            zorder=5,
        )

        # Individual seeds
        for seed, value in zip(SEEDS,values,):

            if not np.isfinite(value):
                continue

            ax.scatter(
                value,
                y + OFFSETS[seed],
                s=65,
                color=COLORS[seed],
                edgecolor="black",
                linewidth=0.5,
                zorder=4,
            )

    # Axis settings
    ax.set_yticks(range(len(methods)), methods,)
    ax.invert_yaxis()
    ax.set_title(label,fontsize=13,)
    ax.set_xlabel(label)
    ax.grid(axis="x",alpha=0.2,)

    # Better zoom for DSC
    if metric == "dsc" and all_values:

        lower = max(0,min(all_values) - 0.02,)
        upper = min(1,max(all_values) + 0.02,)

        ax.set_xlim(lower,upper,)

# Legend
legend_handles = [Line2D([0],[0],marker="o",linestyle="None",markerfacecolor=COLORS[42],markeredgecolor="black",markersize=8,label="Seed 42",),
    Line2D([0],[0],marker="o",linestyle="None",markerfacecolor=COLORS[43],markeredgecolor="black",markersize=8,label="Seed 43",),
    Line2D([0],[0],marker="o",linestyle="None",markerfacecolor=COLORS[44],markeredgecolor="black",markersize=8,label="Seed 44",),
    Line2D([0],[0],marker="D",linestyle="-",color="black",markerfacecolor="black",markeredgecolor="black",markersize=7,label="Mean ± SD",),
]

fig.legend(handles=legend_handles,loc="lower center",ncol=4,frameon=False,)
fig.suptitle("Final model post-processing comparison\n""Points = individual seeds | ""Black diamond = mean ± SD",fontsize=15,)
plt.tight_layout(rect=[0,0.07,1,0.93,])

# Save
OUTPUT.parent.mkdir(parents=True,exist_ok=True,)

plt.savefig(OUTPUT,dpi=300,bbox_inches="tight",)
plt.close()

print()
print("=" * 110)
print("Saved plot:")
print(OUTPUT)
print("=" * 110)