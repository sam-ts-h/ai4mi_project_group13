#!/usr/bin/env python3
"""Evaluate complete augmentation runs on the 256x256 validation volumes.

Run from the project directory, where Sam's metrics.py is available:
    python evaluate_augmentation_3d.py

Original SEGTHOR slices are 512x512. The full-split PNGs are 256x256, so
the in-plane spacing is doubled for the surface-distance metrics. These are
3D metrics on the experiment grid, not metrics on native-resolution NIfTI.
"""

import argparse
import csv
import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

import metrics  # Sam's metric implementations; needs scipy and nibabel


BLOCKS = ((42, 0), (43, 1), (44, 2))
CONDITIONS = (
    ("A00_baseline", "none"),
    ("A01_rotation", "rotation"),
    ("A02_translation", "translation"),
    ("A03_scaling", "scaling"),
    ("A04_noise", "noise"),
    ("A05_combination", "combination"),
)
CLASSES = ("background", "esophagus", "heart", "trachea", "aorta")
METRICS = ("dsc", "hd95", "assd", "fpSliceRate")
PATTERN = re.compile(r"^(Patient_\d{2})_(\d{4})$")
ALLOWED_PIXELS = {0, 63, 126, 189, 252}


def csv_write(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def group_slices(folder):
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    grouped = defaultdict(dict)
    paths = list(folder.glob("*.png"))
    if not paths:
        raise ValueError(f"No PNG slices in {folder}")
    for path in paths:
        match = PATTERN.fullmatch(path.stem)
        if not match:
            raise ValueError(f"Unexpected filename: {path}")
        patient, z = match.group(1), int(match.group(2))
        if z in grouped[patient]:
            raise ValueError(f"Duplicate slice {patient}_{z:04d}")
        grouped[patient][z] = path
    return grouped


def volume(paths, indices):
    slices = []
    for z in indices:
        with Image.open(paths[z]) as image:
            arr = np.asarray(image)
        if arr.shape != (256, 256) or not set(np.unique(arr)).issubset(ALLOWED_PIXELS):
            raise ValueError(f"Unexpected mask shape or values: {paths[z]}")
        slices.append((arr // 63).astype(np.uint8))
    return np.stack(slices, axis=2)


def evaluate_run(run, gt_dir, spacing):
    preds = group_slices(run / "best_epoch" / "val")
    gts = group_slices(gt_dir)
    if set(preds) != set(gts) or len(gts) != 10:
        raise ValueError(f"Expected the same 10 patients in {run} and {gt_dir}")

    patient_rows = []
    values = {name: [] for name in METRICS}
    for patient in sorted(gts):
        indices = sorted(gts[patient])
        if indices != list(range(len(indices))) or sorted(preds[patient]) != indices:
            raise ValueError(f"Missing or extra slices for {patient} in {run}")
        if patient not in spacing:
            raise ValueError(f"Missing spacing for {patient}")
        gt = volume(gts[patient], indices)
        pred = volume(preds[patient], indices)
        dx, dy, dz = spacing[patient]
        voxel_spacing = (2 * dx, 2 * dy, dz)
        if not all(np.isfinite(voxel_spacing)) or min(voxel_spacing) <= 0:
            raise ValueError(f"Invalid spacing for {patient}: {voxel_spacing}")

        patient_scores = {name: np.full(5, np.nan) for name in METRICS}
        for k in range(5):
            gt_mask = gt == k
            pred_mask = pred == k
            if not gt_mask.any():
                continue  # Same missing-GT convention as Sam's metrics.py
            patient_scores["dsc"][k] = metrics.dice(gt_mask, pred_mask)
            patient_scores["fpSliceRate"][k] = metrics.falsePositiveSliceRate(gt_mask, pred_mask)
            if pred_mask.any():
                distances = metrics.surfaceDistances(gt_mask, pred_mask, voxel_spacing)
                patient_scores["hd95"][k] = np.percentile(distances, 95)
                patient_scores["assd"][k] = distances.mean()
        for name, scores in patient_scores.items():
            values[name].append(scores)
            for k, organ in enumerate(CLASSES):
                patient_rows.append({"patient": patient, "class": organ, "metric": name,
                                     "value": scores[k]})

    summary_rows = []
    for name in METRICS:
        stacked = np.stack(values[name])
        # Avoid a runtime warning when a class has no defined value.
        def mean_defined(arr):
            return float(np.nanmean(arr)) if np.isfinite(arr).any() else np.nan
        averages = [mean_defined(stacked[:, k]) for k in range(5)]
        combined = mean_defined(stacked[:, 1:])  # equal weight per patient and organ
        summary_rows.append({"metric": name, **dict(zip(CLASSES, averages)),
                             "combined": combined})
    return patient_rows, summary_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path,
                        default=Path("/local/data/frk340/ai4mi_results/augmentation_full_training"))
    parser.add_argument("--data-root", type=Path, default=Path("/local/data/frk340"))
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Evaluate completed runs even if some of the 18 runs are missing")
    args = parser.parse_args()

    runs = []
    missing = []
    for split, seed in BLOCKS:
        for label, condition in CONDITIONS:
            run = args.results_root / f"split{split}_train{seed}" / f"{label}_{split}_{seed}"
            if not (run / "run.ok").is_file():
                missing.append(str(run))
            else:
                runs.append((split, seed, label, condition, run))
    if missing and not args.allow_incomplete:
        parser.error(f"{len(missing)} run(s) lack run.ok; use --allow-incomplete if intentional:\n" +
                     "\n".join(missing))
    if not runs:
        parser.error("No completed runs found")

    all_summaries = []
    all_patients = []
    for split, seed in BLOCKS:
        block_runs = [item for item in runs if item[:2] == (split, seed)]
        if not block_runs:
            continue
        data = args.data_root / f"SEGTHOR_full_split{split}"
        with (data / "spacing.pkl").open("rb") as handle:
            spacing = pickle.load(handle)
        block_summaries = []
        for _, _, label, condition, run in block_runs:
            print(f"Evaluating {run.name}", flush=True)
            patient_rows, summary_rows = evaluate_run(run, data / "val" / "gt", spacing)
            # Per-run files use Sam's column layout, now with corrected organ names.
            output = run / "metrics_3d_256"
            csv_write(output / "perPatient.csv", ("patient", "class", "metric", "value"), patient_rows)
            csv_write(output / "summary.csv", ("metric", *CLASSES, "combined"), summary_rows)
            info = {"split_seed": split, "train_seed": seed,
                    "experiment": label, "condition": condition, "n_patients": 10}
            block_summaries.extend([{**info, **row} for row in summary_rows])
            all_patients.extend([{**info, **row} for row in patient_rows])
        all_summaries.extend(block_summaries)
        fields = ("split_seed", "train_seed", "experiment", "condition", "n_patients",
                  "metric", *CLASSES, "combined")
        csv_write(args.results_root / f"metrics_3d_256_split{split}_train{seed}.csv",
                  fields, block_summaries)

    fields = ("split_seed", "train_seed", "experiment", "condition", "n_patients",
              "metric", *CLASSES, "combined")
    csv_write(args.results_root / "metrics_3d_256_all.csv", fields, all_summaries)
    csv_write(args.results_root / "metrics_3d_256_perPatient_all.csv",
              ("split_seed", "train_seed", "experiment", "condition", "n_patients",
               "patient", "class", "metric", "value"), all_patients)
    print(f"Done: {len(runs)} runs, {len(all_summaries)} summary rows.\n"
          f"Overview: {args.results_root / 'metrics_3d_256_all.csv'}")


if __name__ == "__main__":
    main()
