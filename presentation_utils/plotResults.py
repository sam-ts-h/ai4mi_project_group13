import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
EPOCH = 'best'
# window runs only did 25 epochs, so the old runs are cut off at 25 too
EPOCHS = 25
experimentsDir = Path("./experiments")
plotDir = experimentsDir / "plots" / f"window_{EPOCH}"
plotDir.mkdir(parents=True, exist_ok=True) #make plots if not exist too
losses = ['ce', 'ceDice', 'ceFocalTversky']
splits = ['split42_seed0', 'split43_seed1', 'split44_seed2']
colors = ['#0173B2', '#DE8F05', '#029E73']

runs = {}
for loss in losses:
    runs[loss] = (experimentsDir, 'metrics25', 'metricsLast25')
    runs[f'{loss} + window'] = (experimentsDir / "experiments_window", 'metrics', 'metricsLast')

# name in summary.csv to x axis label so bit more readable
metrics = {
    'dsc': '3D Dice',
    'hd95': 'HD95 in mm (lower is better)',
    'assd': 'ASSD in mm (lower is better)',
    'fpSliceRate': 'False positive slice rate (lower is better)',
    'sliceDice': '2D val Dice per slice',
}

# results[metric][run] so one per split
results = {}
for metric in metrics:
    results[metric] = {}
    for run in runs:
        results[metric][run] = []

for run, (runsDir, bestFolder, lastFolder) in runs.items():
    metricsFolder = bestFolder if EPOCH == 'best' else lastFolder #Used to see if picking by the 2ddice is not a bit weird (kept for future lookup)
    for split in splits:
        runDir = runsDir / split / run.replace(' + window', '')
        #beaware run eval experiments first!!
        with open(runDir / metricsFolder / "summary.csv") as f:
            for row in csv.DictReader(f):
                if row['metric'] in metrics:
                    results[row['metric']][run].append(float(row['combined']))

        # dice_val is (epoch, slice, class) same as old (to check)
        sliceDice = np.load(runDir / "dice_val.npy")[:EPOCHS, :, 1:].mean(axis=(1, 2))#no background
        results['sliceDice'][run].append(sliceDice.max() if EPOCH == 'best' else sliceDice[-1])

for metric, label in metrics.items():
    fig, ax = plt.subplots(figsize=(9, 6))

    for i, run in enumerate(runs):
        values = np.array(results[metric][run])
        #every split a bit above/ below line so dots dont overlap (if very close unhandy)
        for j, split in enumerate(splits):
            ax.scatter(values[j], i + (j - 1) * 0.15, color=colors[j],s=50, zorder=3,label=split if i == 0 else None)
        ax.errorbar(values.mean(), i, xerr=values.std(ddof=1), fmt='D', color='black',markersize=9, capsize=6,linewidth=1.5, zorder=4)

    ax.set_yticks(range(len(runs)))
    ax.set_yticklabels(runs)
    #want it other way around, so go from first to last
    ax.invert_yaxis()
    ax.set_xlabel(label)
    ax.set_title(f"{EPOCH.capitalize()} epoch of each run (first {EPOCHS} epochs) | diamonds = mean ± SD")
    ax.grid(True, color='lightgrey')
    ax.set_axisbelow(True)
    ax.legend(title="Split / train seed", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    fig.tight_layout()
    fig.savefig(plotDir / f"{metric}.png", dpi=200)
    plt.close(fig)
    print(f"!saved {plotDir / metric}.png")
