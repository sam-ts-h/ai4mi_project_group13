import sys
import shutil
import subprocess
from pathlib import Path
# slices exact same args as final run so test gets same clip/mean/std/window then predict and stitch into the submission folder.
# test in data/segthor_train_full/test/Patient_XX.nii.gz beofre!!!
splitSeed = 42
PROCESSES = 6
sourceDir = "./data/segthor_train_full"
dataDir = Path("./data") / f"SEGTHOR_full_split{splitSeed}"
runDir = "./experiments/final_ceDiceBoundary_200ep_16kernels/split42_seed0/ceDiceBoundary"
predDir = runDir + "/test"
submissionDir = "./group-13"

# sliced.txt only written when slicing finish
if not (dataDir / "sliced.txt").exists():
    if dataDir.exists():
        shutil.rmtree(dataDir)
    print(f"!!!!! slicing split {splitSeed} with test")
    # no --distmap, not needed for predicting
    subprocess.run([sys.executable, "./slice_segthor.py", "--source_dir", sourceDir, "--dest_dir", str(dataDir), "--retains", "10", "--fold", "0", "--seed", str(splitSeed),
        "--clip", "--resample", "--normalize", "--window", "--window_pct", "10", "99.5", "-p", str(PROCESSES)], check=True)
    (dataDir / "sliced.txt").touch()

print("!!!!!!!! predicting test")
subprocess.run([sys.executable, "./predict.py"], check=True)

# crop centers from the new slicing, the one in runDir has no test slices
print("!!!!!!!! stitching test")
subprocess.run([sys.executable, "./stitch.py", "--data_folder", predDir, "--dest_folder", submissionDir + "/test/pred",
                "--num_classes", "255", "--grp_regex", r"(Patient_\d\d)_\d\d\d\d", "--source_scan_pattern", sourceDir + "/test/{id_}.nii.gz",
                "--crop_centers", str(dataDir / "crop_centers.pkl")], check=True)
