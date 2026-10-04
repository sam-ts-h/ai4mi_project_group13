import sys
import shutil
import subprocess
from pathlib import Path

#lices one split, trains all losses on it, then deletes that split's data
# before the next one, so only one split is on disk at a time as very big. can restart after tmux crash (like kieran hgad)
# finished runs are skipped, and an unfinished run starts over.

SEEDS = [(42, 0), (43, 1), (44, 2)]
LOSSES = ['ce', 'ceDice', 'ceDiceBoundary']
EPOCHS = 50
# each slicing worker holds the 3D distance maps of a patient (a few GB)dont go too high
PROCESSES = 4

sourceDir = Path("data/segthor_train_full")
experimentsDir = Path("experiments")

for splitSeed, trainSeed in SEEDS:
    dataset = f"SEGTHOR_full_split{splitSeed}"
    dataDir = Path("data") / dataset
    splitDir = experimentsDir / f"split{splitSeed}_seed{trainSeed}"

    todo = []
    for loss in LOSSES:
        if not (splitDir / loss / "done.txt").exists():
            todo.append(loss)
    if not todo:
        print(f">> split {splitSeed}: all runs done, skipping")
        continue

    # sliced.txt is only written when slicing finished, so half sliced data from a crash  redone
    if not (dataDir / "sliced.txt").exists():
        if dataDir.exists():
            shutil.rmtree(dataDir)
        freeSpace = shutil.disk_usage(".").free / 1e9
        print(f"!!!!! slicing split {splitSeed} ({freeSpace:.0f} GB free)")
        subprocess.run([sys.executable, "slice_segthor.py",
                        "--source_dir", str(sourceDir), "--dest_dir", str(dataDir),
                        "--retains", "10", "--fold", "0", "--seed", str(splitSeed),
                        "--clip", "--resample", "--normalize", "--distmap",
                        "-p", str(PROCESSES)], check=True)
        (dataDir / "sliced.txt").touch()

    for loss in todo:
        runDir = splitDir / loss
        print(f"!!!!!!!! training {loss} on split {splitSeed} with seed {trainSeed}")
        subprocess.run([sys.executable, "main.py",
                        "--dataset", dataset, "--seed", str(trainSeed), "--mode", "full",
                        "--loss", loss, "--epochs", str(EPOCHS), "--dest", str(runDir), "--gpu"], check=True)

        # stitch.py needs this to undo the crop, and the data folder gets deleted
        shutil.copy(dataDir / "crop_centers.pkl", runDir)
        (runDir / "done.txt").touch()

    shutil.rmtree(dataDir)
    print(f"!!!!!! split {splitSeed} done, removed {dataDir}")
