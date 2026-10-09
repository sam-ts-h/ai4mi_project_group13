import sys
import subprocess
from pathlib import Path

#Can reuse later, just make sure we change the path to own runs
experimentsDir = Path("./experiments")
gtDir = Path("./data/segthor_train_full/train")

# prediction folder : (volumes folder, metrics folder) - volumen by stitch 
epochs = {
    'best_epoch': ('volumes', 'metrics'),
    'iter049': ('volumesLast', 'metricsLast'),
}
# old ones done already so just these
losses = ['ceWDice', 'ceFocalDice', 'ceFocalTversky']
runDirs = []
for loss in losses:
    runDirs += sorted(experimentsDir.glob(f"split*/{loss}/"))
print(f"found {len(runDirs)} runs")

for runDir in runDirs:
    for predFolder, (volumesFolder, metricsFolder) in epochs.items():
        #check rerun, for failure
        if (runDir / metricsFolder / "summary.csv").exists():
            print(f"{runDir} {predFolder} --- metrics already done, skipping")
            continue

        print(f"!!!!!!!! stitching {runDir} {predFolder}")
        subprocess.run([sys.executable, "../stitch.py", "--data_folder", str(runDir / predFolder / "val"), "--dest_folder", str(runDir / volumesFolder),
                        "--num_classes", "255", "--grp_regex", r"(Patient_\d\d)_\d\d\d\d","--source_scan_pattern", str(gtDir / "{id_}" / "GT.nii.gz"),"--crop_centers", str(runDir / "crop_centers.pkl")], check=True)

        print(f"!!!!!!!! metrics {runDir} {predFolder}")
        subprocess.run([sys.executable, "../metrics.py", "--pred_folder", str(runDir / volumesFolder), "--gt_folder", str(gtDir),
                        "--dest_folder", str(runDir / metricsFolder)], check=True)
