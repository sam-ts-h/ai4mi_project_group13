#!/usr/bin/env python3
"""
Pre-flight validation for the preprocessing pipeline's crop window.

Confirms that the FINAL crop window (GRID_ROWS x GRID_COLS, centered on the
body centroid -- exactly as slice_patient() in slice_segthor.py computes it)
never clips a real GT-labeled organ voxel, for every patient in the raw
dataset.

This is a stronger check than eda_segthor.py's body-mask validation: that
one confirmed organs sit inside the BODY MASK's full extent. This one
confirms organs sit inside the much smaller, FIXED-SIZE crop window that
actually gets saved -- which is the check that matters, since the body mask
is far larger than the crop window and a mismatched centroid could still
clip organs even when the body-mask check passes.

Works directly on the RAW NIfTI data (like eda_segthor.py) -- does NOT
require `make data/SEGTHOR` to have been run first. Run this BEFORE slicing,
so a too-small grid size is caught before spending time on a full slice run.

Usage:
    $ python validate_crop_window.py --source_dir data/segthor_part1
"""

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from slice_segthor import load_patient_ct  # reuse the exact same loader
from preprocessing_common import (resample_image_slice, resample_mask_slice,
                                  body_mask_2d, body_centroid,
                                  GRID_ROWS, GRID_COLS)


def check_patient(id_: str, source_path: Path) -> list[tuple[int, int]]:
    """
    Returns a list of (z_index, num_clipped_voxels) for every slice where
    the crop window -- computed exactly as slice_patient() does -- would
    clip real GT-labeled organ voxels.
    """
    clipped_ct, gt, (dx, dy, _) = load_patient_ct(id_, source_path, test_mode=False)
    z = clipped_ct.shape[2]
    failures = []

    for idz in range(z):
        img_slice = resample_image_slice(clipped_ct[:, :, idz], (dx, dy))
        gt_slice = resample_mask_slice(gt[:, :, idz], (dx, dy))

        organ_mask = gt_slice > 0
        if not organ_mask.any():
            continue  # nothing to check on a slice with no labeled organ at all

        # Same body-mask + centroid logic slice_patient() actually uses.
        mask = body_mask_2d(img_slice)
        center = body_centroid(mask)
        if center is None:
            center = (img_slice.shape[0] / 2, img_slice.shape[1] / 2)

        # Same window-bounds math as crop_or_pad_to_grid() -- kept in sync
        # manually rather than imported, since crop_or_pad_to_grid() actually
        # performs the crop; here we only need its bounds to check against.
        center_r, center_c = center
        r_start = int(round(center_r - GRID_ROWS / 2))
        c_start = int(round(center_c - GRID_COLS / 2))
        r_end = r_start + GRID_ROWS
        c_end = c_start + GRID_COLS

        ys, xs = np.where(organ_mask)
        outside = (ys < r_start) | (ys >= r_end) | (xs < c_start) | (xs >= c_end)
        if outside.any():
            failures.append((idz, int(outside.sum())))

    return failures


def main(args: argparse.Namespace) -> None:
    source_path = Path(args.source_dir)
    train_dir = source_path / "train"
    # Deliberately checks every patient under train/ -- i.e. the full
    # train+val pool before get_splits() divides it -- since the crop window
    # is identical regardless of which split a patient later lands in.
    patient_ids = sorted(p.name for p in train_dir.glob("Patient_*") if p.is_dir())
    if args.max_patients:
        patient_ids = patient_ids[:args.max_patients]

    print(f"Validating the {GRID_ROWS}x{GRID_COLS} crop window against "
          f"{len(patient_ids)} patients...")

    all_failures = []  # (patient_id, z_index, num_clipped_voxels)
    for pid in tqdm(patient_ids, desc="Checking patients"):
        for idz, n in check_patient(pid, source_path):
            all_failures.append((pid, idz, n))

    if not all_failures:
        print(f"\nPASSED: across all {len(patient_ids)} patients, every GT-labeled organ "
              f"voxel\nfell inside the {GRID_ROWS}x{GRID_COLS} crop window, in every slice.")
    else:
        total_slices = len(all_failures)
        total_voxels = sum(f[2] for f in all_failures)
        affected_patients = sorted(set(f[0] for f in all_failures))
        print(f"\nFAILED on {total_slices} slice(s) across {len(affected_patients)} "
              f"patient(s), {total_voxels} organ voxels total would be clipped:")
        for pid, idz, n in all_failures[:20]:
            print(f"  - {pid}, slice {idz}: {n} organ voxels would fall outside the crop window")
        if total_slices > 20:
            print(f"  ... and {total_slices - 20} more")
        print("\nAffected patients:", affected_patients)
        print("\nConsider: increasing GRID_ROWS/GRID_COLS in preprocessing_common.py, or")
        print("investigating these specific patients/slices directly (e.g. in viewer.py")
        print("or 3D Slicer) before running the full slicing pipeline.")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the preprocessing crop window against real GT, before slicing")
    parser.add_argument('--source_dir', type=str, default='data/segthor_part1',
                        help="Folder containing train/Patient_XX/... (raw NIfTI data)")
    parser.add_argument('--max_patients', type=int, default=None,
                        help="Optional: limit to first N patients, for a quick test run")
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())