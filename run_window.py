#!/usr/bin/env python3
"""
Second-channel ("soft tissue window") experiment for all 3 (split seed, train seed) pairs: slice with --window, then train.

  raw data : data/SEGTHOR_VOLUMES_FINAL
  data     : data/SEGTHOR_full_s<split>_win<tag>      (same split seed => same validation patients as the baseline)
  results  : results/full_s<split>_t<train>_aug_win<tag>   (augmentation = main.py's default, like the _aug baseline)

Needs: the windowed slice_segthor.py, ENet.py `K - in_dim`, dataset.py infer_in_channels, main.py using it (see the guide).
Resumable: .done runs are skipped, finished slicing is reused, a crashed training run restarts from epoch 0.

Usage:  python run_window.py [--dry_run]
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
os.environ.setdefault("TRITON_CACHE_DIR", str(HERE / ".triton_cache"))


def need(file, text, hint):
    if text not in (HERE / file).read_text():
        sys.exit(f"ERROR: {file} is missing the window-channel change ({hint})")


def main(a):
    need("slice_segthor.py", "--window_pct", "use the slice_segthor.py with --window / --window_pct")
    need("ENet.py", "K - in_dim", "conv0 must output K - in_dim channels")
    need("dataset.py", "def infer_in_channels", "add infer_in_channels")
    need("main.py", "infer_in_channels(", "build the net with infer_in_channels(root_dir)")
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    for split, train in zip(a.split_seeds, a.train_seeds):
        data = Path("data") / f"SEGTHOR_full_s{split}_win{a.tag}"
        res = Path("results") / f"full_s{split}_t{train}_aug_win{a.tag}"
        print(f"\n>>> split {split} / train {train}: {data} -> {res}", flush=True)
        if (res / ".done").exists():
            print("   already done, skipping")
            continue
        steps = []
        if not (HERE / data / "spacing.pkl").exists():     # spacing.pkl is written last by slice_segthor.py
            steps.append(("slice", [sys.executable, "slice_segthor.py", "--source_dir", a.src, "--dest_dir", data,
                                    "--retains", a.retains, "--seed", split, "--clip", "--resample", "--normalize",
                                    "--window", "--window_pct", *a.window_pct]))
        else:
            print("   sliced data already present, skipping slicing")
        steps.append(("train", [sys.executable, "-O", "main.py", "--dataset", "SEGTHOR", "--data_dir", data.name, "--dest", res,
                                "--epochs", a.epochs, "--seed", train, "--gpu", "--augmentation", "combination_no_noise",
                                "--augmentation-probability", "0.5"]))
        for kind, cmd in steps:
            cmd = [str(c) for c in cmd]
            print("   $ " + " ".join(cmd), flush=True)
            if a.dry_run:
                continue
            if kind == "slice":
                shutil.rmtree(HERE / data, ignore_errors=True)   # slice_segthor.py refuses to write into an existing folder
                subprocess.run(cmd, check=True, cwd=HERE)
            else:
                first = sorted((HERE / data / "train" / "img").glob("*.npy"))[0]
                shape = np.load(first, mmap_mode="r").shape
                if len(shape) != 3 or shape[0] != 2:
                    sys.exit(f"ERROR: slices in {data} have shape {shape}, expected (2, H, W). Delete {data} and rerun.")
                shutil.rmtree(HERE / res, ignore_errors=True)
                subprocess.run(cmd, check=True, cwd=HERE, env=env)
                (HERE / res / ".done").touch()
    print("\n>>> All done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="data/SEGTHOR_VOLUMES_FINAL")
    p.add_argument("--split_seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--train_seeds", type=int, nargs="+", default=[0, 1, 2], help="paired by position with --split_seeds")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--retains", type=int, default=8)
    p.add_argument("--window_pct", nargs=2, default=["10", "99.5"], help="esophagus HU percentiles (training patients) = window")
    p.add_argument("--tag", default="", help="suffix for a window variant, e.g. _narrow -> ..._win_narrow folders")
    p.add_argument("--dry_run", action="store_true")
    a = p.parse_args()
    assert len(a.split_seeds) == len(a.train_seeds), "--split_seeds and --train_seeds must be the same length"
    main(a)