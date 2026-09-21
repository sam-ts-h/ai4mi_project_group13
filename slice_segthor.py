#!/usr/bin/env python3.7

# MIT License

# Copyright (c) 2024 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pickle
import random
import argparse
import warnings
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from typing import Callable

import numpy as np
import nibabel as nib
from skimage.io import imsave
from skimage.transform import resize

from utils import map_, tqdm_


def norm_arr(img: np.ndarray) -> np.ndarray:
    casted = img.astype(np.float32)
    shifted = casted - casted.min()
    norm = shifted / shifted.max()
    res = 255 * norm

    assert 0 == res.min(), res.min()
    assert res.max() == 255, res.max()

    return res.astype(np.uint8)


def sanity_ct(ct, x, y, z, dx, dy, dz) -> bool:
    assert ct.dtype in [np.int16, np.int32], ct.dtype
    assert -1000 <= ct.min(), ct.min()
    assert ct.max() <= 31743, ct.max()

    assert 0.896 <= dx <= 1.37, dx  # Rounding error
    assert dx == dy
    assert 2 <= dz <= 3.7, dz

    assert (x, y) == (512, 512)
    assert x == y
    assert 135 <= z <= 284, z

    return True


def sanity_gt(gt, ct) -> bool:
    assert gt.shape == ct.shape
    assert gt.dtype in [np.uint8], gt.dtype

    # Do the test on 3d: assume all organs are present..
    # assert set(np.unique(gt)) == set(range(5))

    return True


resize_: Callable = partial(resize, mode="constant", preserve_range=True, anti_aliasing=False)


def slice_patient(id_: str, dest_path: Path, source_path: Path, shape: tuple[int, int],
                  test_mode: bool = False) -> tuple[float, float, float]:
    id_path: Path = source_path / ("train" if not test_mode else "test") / id_

    ct_path: Path = (id_path / f"{id_}.nii.gz") if not test_mode else (source_path / "test" / f"{id_}.nii.gz")
    nib_obj = nib.load(str(ct_path))
    ct: np.ndarray = np.asarray(nib_obj.dataobj)
    # dx, dy, dz = nib_obj.header.get_zooms()
    x, y, z = ct.shape
    dx, dy, dz = nib_obj.header.get_zooms()

    assert sanity_ct(ct, *ct.shape, *nib_obj.header.get_zooms())

    gt: np.ndarray
    if not test_mode:
        #gt_path: Path = id_path / "GT.nii.gz"
        gt_path: Path = id_path / "GT_correct.nii.gz"
        gt_nib = nib.load(str(gt_path))
        # print(nib_obj.affine, gt_nib.affine)
        gt = np.asarray(gt_nib.dataobj)
        assert sanity_gt(gt, ct)


        assert np.allclose(nib_obj.affine, gt_nib.affine), (
            f"CT/GT affine mismatch for {id_}"
        )
        assert set(np.unique(gt).tolist()) == {0, 1, 2, 3, 4}, (
            f"Unexpected classes for {id_}: {np.unique(gt)}"
        )
    else:
        gt = np.zeros_like(ct, dtype=np.uint8)

    norm_ct: np.ndarray = norm_arr(ct)

    to_slice_ct = norm_ct
    to_slice_gt = gt

    for idz in range(z):
        img_slice = resize_(to_slice_ct[:, :, idz], shape).astype(np.uint8)
        gt_slice = resize_(to_slice_gt[:, :, idz], shape, order=0).astype(np.uint8)
        assert img_slice.shape == gt_slice.shape
        gt_slice *= 63
        assert gt_slice.dtype == np.uint8, gt_slice.dtype
        # assert set(np.unique(gt_slice)) <= set(range(5))
        assert set(np.unique(gt_slice)) <= set([0, 63, 126, 189, 252]), np.unique(gt_slice)

        arrays: list[np.ndarray] = [img_slice, gt_slice]

        subfolders: list[str] = ["img", "gt"]
        assert len(arrays) == len(subfolders)
        for save_subfolder, data in zip(subfolders,
                                        arrays):
            filename = f"{id_}_{idz:04d}.png"

            save_path: Path = Path(dest_path, save_subfolder)
            save_path.mkdir(parents=True, exist_ok=True)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                imsave(str(save_path / filename), data)

    return dx, dy, dz

EXCLUDED_PATIENTS = {
    "Patient_05",
    "Patient_15",
    "Patient_17",
    "Patient_19",
}

def get_splits(
    src_path: Path,
    retains: int,
    fold: int
) -> tuple[list[str], list[str], list[str]]:
    train_path = src_path / "train"

    all_patient_ids = sorted(
        path.name
        for path in train_path.glob("Patient_*")
        if path.is_dir()
    )

    ids = [
        patient_id
        for patient_id in all_patient_ids
        if patient_id not in EXCLUDED_PATIENTS
    ]

    print(f"Found {len(all_patient_ids)} patient folders")
    print(f"Excluded patients: {sorted(EXCLUDED_PATIENTS)}")
    print(f"Usable patients: {ids}")

    assert len(all_patient_ids) == 20, (
        f"Expected 20 patient folders, found {len(all_patient_ids)}"
    )
    assert len(ids) == 16, (
        f"Expected 16 usable patients, found {len(ids)}"
    )
    assert retains == 4, (
        f"This experiment requires 4 validation patients, got {retains}"
    )

    random.shuffle(ids)

    validation_slice = slice(
        fold * retains,
        (fold + 1) * retains
    )
    validation_ids = ids[validation_slice]

    assert len(validation_ids) == retains

    training_ids = [
        patient_id
        for patient_id in ids
        if patient_id not in validation_ids
    ]

    assert len(training_ids) == 12
    assert len(validation_ids) == 4
    assert set(training_ids).isdisjoint(validation_ids)
    assert not set(training_ids) & EXCLUDED_PATIENTS
    assert not set(validation_ids) & EXCLUDED_PATIENTS

    test_path = src_path / "test"
    if test_path.exists():
        test_ids = sorted(
            Path(path.stem).stem
            for path in test_path.glob("*")
        )
    else:
        test_ids = []

    print(f"Training IDs ({len(training_ids)}): {training_ids}")
    print(f"Validation IDs ({len(validation_ids)}): {validation_ids}")
    print(f"Test IDs ({len(test_ids)}): {test_ids}")

    return training_ids, validation_ids, test_ids


def main(args: argparse.Namespace):
    src_path: Path = Path(args.source_dir)
    dest_path: Path = Path(args.dest_dir)

    # Assume the clean up is done before calling the script
    assert src_path.exists()
    assert not dest_path.exists()

    training_ids: list[str]
    validation_ids: list[str]
    test_ids: list[str]
    training_ids, validation_ids, test_ids = get_splits(src_path, args.retains, args.fold)

    dest_path.mkdir(parents=True, exist_ok=False)

    with open(dest_path / "split.txt", "w") as split_file:
        split_file.write(f"seed={args.seed}\n")
        split_file.write(f"fold={args.fold}\n")
        split_file.write(
            f"excluded={','.join(sorted(EXCLUDED_PATIENTS))}\n"
        )
        split_file.write(
            f"train={','.join(training_ids)}\n"
        )
        split_file.write(
            f"val={','.join(validation_ids)}\n"
        )

    resolution_dict: dict[str, tuple[float, float, float]] = {}

    split_ids: list[str]
    for mode, split_ids in zip(["train", "val"], [training_ids, validation_ids]):
        dest_mode: Path = dest_path / mode
        print(f"Slicing {len(split_ids)} pairs to {dest_mode}")

        pfun: Callable = partial(slice_patient,
                                 dest_path=dest_mode,
                                 source_path=src_path,
                                 shape=tuple(args.shape),
                                 test_mode=mode == 'test')
        resolutions: list[tuple[float, float, float]]
        iterator = tqdm_(split_ids)
        match args.process:
            case 1:
                resolutions = list(map(pfun, iterator))
            case -1:
                resolutions = Pool().map(pfun, iterator)
            case _ as p:
                resolutions = Pool(p).map(pfun, iterator)

        for key, val in zip(split_ids, resolutions):
            resolution_dict[key] = val

    with open(dest_path / "spacing.pkl", 'wb') as f:
        pickle.dump(resolution_dict, f, pickle.HIGHEST_PROTOCOL)
        print(f"Saved spacing dictionnary to {f}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Slicing parameters')
    parser.add_argument('--source_dir', type=str, required=True)
    parser.add_argument('--dest_dir', type=str, required=True)

    parser.add_argument('--shape', type=int, nargs="+", default=[256, 256])
    parser.add_argument('--retains', type=int, default=25, help="Number of retained patient for the validation data")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--process', '-p', type=int, default=1,
                        help="The number of cores to use for processing")
    args = parser.parse_args()
    random.seed(args.seed)

    print(args)

    return args


if __name__ == "__main__":
    main(get_args())
