import sys
import shutil
import subprocess
from pathlib import Path
# slices exact same args as final run so test gets same clip/mean/std/window then predict and stitch into the submission folder.
# test in data/segthor_train_full/test/Patient_XX.nii.gz beofre!!!
SUBSET  = sys.argv[1] if len(sys.argv) > 1 else "val"       # default validation fallback
splitSeed = 42
PROCESSES = 6
sourceDir = "./data/segthor"
dataDir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("splits/SEGTHOR_full_split42")
runDir = "./results/final/seed42"
predDir = runDir + "/predict_test"
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
subprocess.run([sys.executable, "./predict.py", "test", str(dataDir)], check=True)

# crop centers from the new slicing, the one in runDir has no test slices
print("!!!!!!!! stitching test")
subprocess.run([sys.executable, "./stitch.py", "--data_folder", predDir,
                "--dest_folder", submissionDir + "/test/pred_raw",
                "--num_classes", "5", "--grp_regex", r"(Patient_\d+)_\d+",
                "--source_scan_pattern", sourceDir + "/test/{id_}.nii.gz",
                "--crop_centers", str(dataDir / "crop_centers.pkl")], check=True)

# and finally: post processing and we are done
print("!!!!!!!! postprocessing")
subprocess.run([sys.executable, "./postprocess_small_components.py",
                "--input_dir", submissionDir + "/test/pred_raw",
                "--output_dir", submissionDir + "/test/pred",
                "--min_voxels", "50", "--min_fraction", "0.005"], check=True)