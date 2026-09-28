#!/usr/bin/env python3
"""Run the paired SEGTHOR augmentation experiment on the VU compute server.

Usage from the project root (after copying this file there):
    python run_augmentation_full.py --smoke
    python run_augmentation_full.py --all

Use the same Python environment as main.py. Run on the compute instance whose
/local/data/frk340 directory holds the three complete split directories.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


BLOCKS = ((42, 0), (43, 1), (44, 2))
CONDITIONS = (
    ("A00_baseline", "none"),
    ("A01_rotation", "rotation"),
    ("A02_translation", "translation"),
    ("A03_scaling", "scaling"),
    ("A04_noise", "noise"),
    ("A05_combination", "combination"),
)
PROJECT = Path(__file__).resolve().parent
DATA_ROOT = Path("/local/data/frk340")
RESULT_ROOT = DATA_ROOT / "ai4mi_results" / "augmentation_full_training"


def check_code():
    if not (PROJECT / "main.py").is_file():
        raise RuntimeError(f"main.py ontbreekt in {PROJECT}; zet dit script in de projectmap.")
    result = subprocess.run(
        [sys.executable, "main.py", "--help"], cwd=PROJECT,
        text=True, capture_output=True,
    )
    if result.returncode:
        raise RuntimeError(f"main.py --help faalt:\n{result.stderr}")
    missing = [x for x in ("SEGTHOR_full_split42", "SEGTHOR_full_split43",
                           "SEGTHOR_full_split44", "noise", "combination")
               if x not in result.stdout]
    if missing:
        raise RuntimeError(
            f"main.py accepteert nog niet: {missing}. Werk main.py en dataset.py bij "
            "op de server voordat je de runs start."
        )
    gpu = subprocess.run(
        [sys.executable, "-c", "import torch; print(torch.cuda.is_available())"],
        cwd=PROJECT, text=True, capture_output=True,
    )
    if gpu.returncode or gpu.stdout.strip() != "True":
        raise RuntimeError(f"GPU niet beschikbaar in deze Python-omgeving: {gpu.stderr}")


def check_data(seed):
    name = f"SEGTHOR_full_split{seed}"
    source = DATA_ROOT / name
    link = PROJECT / "data" / name
    for subset, expected_patients in (("train", 30), ("val", 10)):
        img = source / subset / "img"
        gt = source / subset / "gt"
        if not img.is_dir() or not gt.is_dir():
            raise RuntimeError(f"Ontbrekende data: {img} of {gt}. Is scp al klaar, en zit je op de juiste compute-instance?")
        images = {p.name for p in img.glob("*.png")}
        masks = {p.name for p in gt.glob("*.png")}
        patients = {filename.rsplit("_", 1)[0] for filename in images}
        if not images or images != masks or len(patients) != expected_patients:
            raise RuntimeError(
                f"Onvolledige {name}/{subset}: {len(images)} CT, {len(masks)} masks, "
                f"{len(patients)} patiënten; verwacht {expected_patients} patiënten."
            )
    if not (source / "spacing.pkl").is_file():
        raise RuntimeError(f"Ontbreekt: {source / 'spacing.pkl'}")
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() != source.resolve():
            raise RuntimeError(f"Bestaande link wijst elders heen: {link} -> {link.resolve()}")
    elif link.exists():
        if link.resolve() != source.resolve():
            raise RuntimeError(f"Pad bestaat al, controleer handmatig: {link}")
    else:
        link.symlink_to(source, target_is_directory=True)
    reference = RESULT_ROOT / f"split{seed}_train{dict(BLOCKS)[seed]}" / "reference"
    reference.mkdir(parents=True, exist_ok=True)
    destination_gt = reference / "gt"
    if destination_gt.exists():
        expected = {p.name for p in (source / "val" / "gt").glob("*.png")}
        actual = {p.name for p in destination_gt.glob("*.png")}
        if actual != expected:
            raise RuntimeError(f"Onvolledige referentiemaskers: {destination_gt}")
    else:
        shutil.copytree(source / "val" / "gt", destination_gt)
    shutil.copy2(source / "spacing.pkl", reference / "spacing.pkl")
    print(f"Data gecontroleerd: {name} (30 train, 10 val patiënten).", flush=True)


def run(seed, train_seed, condition, epochs, probability, debug=False):
    label = next(label for label, value in CONDITIONS if value == condition)
    name = "smoke_split42_noise" if debug else f"{label}_{seed}_{train_seed}"
    dest = (RESULT_ROOT / name if debug else
            RESULT_ROOT / f"split{seed}_train{train_seed}" / name)
    done = dest / "run.ok"
    if done.is_file():
        print(f"Al voltooid: {name}", flush=True)
        return
    if dest.exists():
        raise RuntimeError(
            f"{dest} bestaat zonder run.ok: mogelijk onvoltooide run. "
            "Bekijk de log; verwijder/verplaats die map handmatig voordat je opnieuw start."
        )
    dest.mkdir(parents=True)
    (dest / "run.json").write_text(json.dumps({
        "split_seed": seed,
        "train_seed": train_seed,
        "fold": 0,
        "dataset": f"SEGTHOR_full_split{seed}",
        "condition": condition,
        "epochs": epochs,
        "augmentation_probability": probability,
        "validation_reference": str(dest.parent / "reference") if not debug else None,
        "main_sha256": hashlib.sha256((PROJECT / "main.py").read_bytes()).hexdigest(),
        "dataset_sha256": hashlib.sha256((PROJECT / "dataset.py").read_bytes()).hexdigest(),
    }, indent=2) + "\n")
    cmd = [sys.executable, "-u", "main.py", "--dataset", f"SEGTHOR_full_split{seed}",
           "--dest", str(dest), "--seed", str(train_seed), "--epochs", str(epochs),
           "--mode", "full", "--gpu", "--augmentation", condition,
           "--augmentation-probability", str(probability)]
    if debug:
        cmd.append("--debug")
    print(f"Start: {name}; log: {dest / 'run.log'}", flush=True)
    with (dest / "run.log").open("w") as log:
        log.write("Command: " + " ".join(cmd) + "\n")
        log.flush()
        result = subprocess.run(cmd, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{name} faalde (exit {result.returncode}). Bekijk {dest / 'run.log'}")
    if not (dest / "best_epoch" / "val").is_dir() or not (dest / "bestweights.pt").is_file():
        raise RuntimeError(f"{name} stopte zonder complete beste-epoch voorspellingen/gewichten: {dest}")
    done.write_text(f"split_seed={seed}\ntrain_seed={train_seed}\ncondition={condition}\nepochs={epochs}\n")
    print(f"Voltooid: {name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="1 epoch noise, 10 samples")
    mode.add_argument("--all", action="store_true", help="18 volledige runs na geslaagde smoke-test")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--probability", type=float, default=0.5)
    args = parser.parse_args()
    if args.epochs < 1 or not 0 <= args.probability <= 1:
        parser.error("--epochs moet >= 1 en --probability moet tussen 0 en 1 zijn")
    try:
        check_code()
        for seed, _ in BLOCKS[:1] if args.smoke else BLOCKS:
            check_data(seed)
        if args.smoke:
            run(42, 0, "noise", 1, args.probability, debug=True)
        else:
            automated_smoke = RESULT_ROOT / "smoke_split42_noise" / "run.ok"
            manual_smoke = DATA_ROOT / "ai4mi_results" / "manual_smoke_split42_noise"
            if not automated_smoke.is_file() and not all(
                (manual_smoke / filename).is_file()
                for filename in ("best_epoch.txt", "bestweights.pt")
            ):
                raise RuntimeError("Doe eerst: python run_augmentation_full.py --smoke")
            for seed, train_seed in BLOCKS:
                for _, condition in CONDITIONS:
                    run(seed, train_seed, condition, args.epochs, args.probability)
    except RuntimeError as exc:
        parser.exit(1, f"Fout: {exc}\n")


if __name__ == "__main__":
    main()
