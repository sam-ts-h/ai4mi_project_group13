import sys
import subprocess
from pathlib import Path
import numpy as np

#Can reuse later, just make sure we change the path to own runs
experimentsDir = Path("./experiments")
windowDir = experimentsDir / "experiments_window"
gtDir = Path("./data/segthor_train_full/train")

# window runs only did 25 epochs
EPOCHS = 25
losses = ['ce', 'ceDice', 'ceFocalTversky']

#volumen by stitch
jobs = []
for loss in losses:
    for runDir in sorted(windowDir.glob(f"split*/{loss}/")):
        jobs.append((runDir, 'best_epoch', 'volumes', 'metrics'))
        jobs.append((runDir, f'iter{EPOCHS - 1:03d}', 'volumesLast', 'metricsLast'))

    # old runs without window 50 epochs pick best of the first 25 for fairness
    for runDir in sorted(experimentsDir.glob(f"split*/{loss}/")):
        sliceDice = np.load(runDir / "dice_val.npy")[:EPOCHS, :, 1:].mean(axis=(1, 2))
        jobs.append((runDir, f'iter{sliceDice.argmax():03d}', 'volumes25', 'metrics25'))
        jobs.append((runDir, f'iter{EPOCHS - 1:03d}', 'volumesLast25', 'metricsLast25'))
print(f"found {len(jobs)} jobs")

for runDir, predFolder, volumesFolder, metricsFolder in jobs:
    #check rerun, for failure
    if (runDir / metricsFolder / "summary.csv").exists():
        print(f"{runDir} {predFolder} --- metrics already done, skipping")
        continue

    print(f"!!!!!!!! stitching {runDir} {predFolder}")
    subprocess.run([sys.executable, "./stitch.py", "--data_folder", str(runDir / predFolder / "val"), "--dest_folder", str(runDir / volumesFolder),
                    "--num_classes", "255", "--grp_regex", r"(Patient_\d\d)_\d\d\d\d","--source_scan_pattern", str(gtDir / "{id_}" / "GT.nii.gz"),"--crop_centers", str(runDir / "crop_centers.pkl")], check=True)

    print(f"!!!!!!!! metrics {runDir} {predFolder}")
    subprocess.run([sys.executable, "./metrics.py", "--pred_folder", str(runDir / volumesFolder), "--gt_folder", str(gtDir),
                    "--dest_folder", str(runDir / metricsFolder)], check=True)
