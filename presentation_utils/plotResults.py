import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
EPOCH = 'last'
experimentsDir = Path("../experiments")
plotDir = experimentsDir / "plots" / EPOCH
plotDir.mkdir(parents=True, exist_ok=True) #make plots if not exist too
metricsFolder = "metrics" if EPOCH == 'best' else "metricsLast" #Used to see if picking by the 2ddice is not a bit weird (kept for future lookup)
losses = ['ce', 'ceDice', 'ceDiceBoundary']
splits = ['split42_seed0', 'split43_seed1', 'split44_seed2']
colors = ['#0173B2', '#DE8F05', '#029E73']

# name in summary.csv to x axis label so bit more readable
metrics = {
    'dsc': '3D Dice (4 organs)',
    'hd95': 'HD95 in mm (4 organs, lower is better)',
    'assd': 'ASSD in mm (4 organs, lower is better)',
    'fpSliceRate': 'False positive slice rate (4 organs, lower is better)',
}

# results[metric][loss] so one per split
results = {}
for metric in metrics:
    results[metric] = {}
    for loss in losses:
        results[metric][loss] = []

for loss in losses:
    for split in splits:
        runDir = experimentsDir / split / loss
        #beaware run eval experiments first!!
        with open(runDir / metricsFolder / "summary.csv") as f:
            for row in csv.DictReader(f):
                if row['metric'] in metrics:
                    results[row['metric']][loss].append(float(row['combined']))

for metric, label in metrics.items():
    fig, ax = plt.subplots(figsize=(9, 4.5))
    
    for i, loss in enumerate(losses):
        values = np.array(results[metric][loss])
        #every split a bit above/ below line so dots dont overlap (if very close unhandy)
        for j, split in enumerate(splits):
            ax.scatter(values[j], i + (j - 1) * 0.15, color=colors[j],s=50, zorder=3,label=split if i == 0 else None)
        ax.errorbar(values.mean(), i, xerr=values.std(ddof=1), fmt='D', color='black',markersize=9, capsize=6,linewidth=1.5, zorder=4)

    ax.set_yticks(range(len(losses)))
    ax.set_yticklabels(losses)
    #want it other way around, so go from first to last
    ax.invert_yaxis()
    ax.set_xlabel(label)
    ax.set_title(f"{EPOCH.capitalize()} epoch of each run | diamonds = mean ± SD")
    ax.grid(True, color='lightgrey')
    ax.set_axisbelow(True)
    ax.legend(title="Split / train seed", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    fig.tight_layout()
    fig.savefig(plotDir / f"{metric}.png", dpi=200)
    plt.close(fig)
    print(f"!saved {plotDir / metric}.png")
