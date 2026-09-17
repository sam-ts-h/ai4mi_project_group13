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

import json
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

from utils import map_, tqdm_
from preprocessing_common import CLIP_MIN, CLIP_MAX, resample_image_slice, resample_mask_slice


def clip_ct(img: np.ndarray) -> np.ndarray:
    """
    Clip raw HU values to the dataset's foreground percentile range, no rescale to 0-255. 
    This replaces the old norm_arr() lossy per-slice min-max normalization: the network now receives real, clipped HU values
    directly, consistent across every slice and every patient (the same real tissue density always maps to the same value
    (assuming consistent HU values from scanner etc), regardless of what else happens to be in that particular slice, 
    which per-slice min-max could not guarantee).
    """
    return np.clip(img.astype(np.float32), CLIP_MIN, CLIP_MAX)


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


def load_patient_ct(id_: str, source_path: Path, test_mode: bool = False):
    """
    Load one patient's raw CT (and GT, unless test_mode). Factored out clipping since both slice_patient() and 
    compute_global_stats() need to load and clip the same raw data.
    """
    id_path: Path = source_path / ("train" if not test_mode else "test") / id_

    ct_path: Path = (id_path / f"{id_}.nii.gz") if not test_mode else (source_path / "test" / f"{id_}.nii.gz")
    nib_obj = nib.load(str(ct_path))
    ct: np.ndarray = np.asarray(nib_obj.dataobj)
    dx, dy, dz = nib_obj.header.get_zooms()

    assert sanity_ct(ct, *ct.shape, dx, dy, dz)

    gt: np.ndarray
    if not test_mode:
        gt_path: Path = id_path / "GT.nii.gz"
        gt_nib = nib.load(str(gt_path))
        gt = np.asarray(gt_nib.dataobj)
        assert sanity_gt(gt, ct)
    else:
        gt = np.zeros_like(ct, dtype=np.uint8)

    # Do the percentile clipping.
    clipped_ct: np.ndarray = clip_ct(ct)

    return clipped_ct, gt, (dx, dy, dz)


def compute_global_stats(training_ids: list[str], source_path: Path) -> tuple[float, float]:
    """
    Two-pass normalization. pass 1: compute a single dataset-wide mean and std, over every pixel of every clipped, resampled 
    training slice (never validation or test, since computing stats from data you'll later evaluate on is data leakage). 
    This re-does the clip + resample work that slice_patient() will do again in pass 2 for the actual save. 
    Running values are used to avoid memory blowup. 
    """
    total_sum = 0.0
    total_sumsq = 0.0
    total_count = 0

    for id_ in tqdm_(training_ids, desc="Computing normalization stats (pass 1/2)"):
        clipped_ct, _, (dx, dy, _) = load_patient_ct(id_, source_path, test_mode=False)
        z = clipped_ct.shape[2]
        for idz in range(z):
            resampled = resample_image_slice(clipped_ct[:, :, idz], (dx, dy))
            # Accumulate in float64: total_count will run into the hundreds of millions of pixels across the full training set, 
            # and a float32 running sum could start losing real precision.
            total_sum += resampled.sum(dtype=np.float64)
            total_sumsq += np.sum(resampled.astype(np.float64) ** 2)
            total_count += resampled.size

    mean = total_sum / total_count
    variance = (total_sumsq / total_count) - mean ** 2
    std = float(np.sqrt(max(variance, 1e-8)))  # guard against a tiny negative from float rounding

    return float(mean), std


def slice_patient(id_: str, dest_path: Path, source_path: Path, shape: tuple[int, int],
                  mean: float, std: float, test_mode: bool = False) -> tuple[float, float, float]:
    clipped_ct, gt, (dx, dy, dz) = load_patient_ct(id_, source_path, test_mode)
    z = clipped_ct.shape[2]

    to_slice_ct = clipped_ct
    to_slice_gt = gt

    img_save_path: Path = Path(dest_path, "img")
    gt_save_path: Path = Path(dest_path, "gt")
    img_save_path.mkdir(parents=True, exist_ok=True)
    gt_save_path.mkdir(parents=True, exist_ok=True)

    for idz in range(z):
        # Resample each slice to a common in-plane spacing. This is what makes the same real-world organ size occupy the same
        # pixel count regardless of which patient's original spacing it came from.
        img_slice = resample_image_slice(to_slice_ct[:, :, idz], (dx, dy))
        gt_slice = resample_mask_slice(to_slice_gt[:, :, idz], (dx, dy))
        assert img_slice.shape == gt_slice.shape

        # Normalize after resampling, using taining mean/std
        # The same fixed mean/std is applied identically whether this call is processing a train or val patient,
        # so val data is normalized exactly the way it will be at real inference time (fixed stats, no peeking at val/test data).
        img_slice = ((img_slice - mean) / std).astype(np.float32)

        gt_slice *= 63
        assert gt_slice.dtype == np.uint8, gt_slice.dtype
        # assert set(np.unique(gt_slice)) <= set(range(5))
        assert set(np.unique(gt_slice)) <= set([0, 63, 126, 189, 252]), np.unique(gt_slice)

        filename_stem = f"{id_}_{idz:04d}"

        # Image saved losslessly as .npy
        np.save(str(img_save_path / f"{filename_stem}.npy"), img_slice)

        # GT stays a normal PNG sincediscrete class labels, so no precision to lose, 
        # and this keeps it directly viewable/compatible with viewer.py as before.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            imsave(str(gt_save_path / f"{filename_stem}.png"), gt_slice)

    return dx, dy, dz


def get_splits(src_path: Path, retains: int, fold: int) -> tuple[list[str], list[str], list[str]]:
    ids: list[str] = sorted(map_(lambda p: p.name, (src_path / 'train').glob('*')))
    print(f"Founds {len(ids)} in the id list")
    print(ids[:10])
    assert len(ids) > retains

    random.shuffle(ids)  # Shuffle before to avoid any problem if the patients are sorted in any way
    validation_slice = slice(fold * retains, (fold + 1) * retains)
    validation_ids: list[str] = ids[validation_slice]
    assert len(validation_ids) == retains

    training_ids: list[str] = [e for e in ids if e not in validation_ids]
    assert (len(training_ids) + len(validation_ids)) == len(ids)

    test_ids: list[str] = sorted(map_(lambda p: Path(p.stem).stem, (src_path / 'test').glob('*')))
    print(f"Founds {len(test_ids)} test ids")
    print(test_ids[:10])

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

    # Pass one over data: Compute dataset normalization stats from training data only (avoid data leakage)
    print(">> Computing normalization statistics over the training set...")
    mean, std = compute_global_stats(training_ids, src_path)
    print(f">> mean={mean:.3f}, std={std:.3f}")

    # Save stats for future efficiency
    dest_path.mkdir(parents=True, exist_ok=True)
    with open(dest_path / "normalization_stats.json", "w") as f:
        json.dump({"mean": mean, "std": std}, f, indent=2)
        print(f"Saved normalization stats to {f.name}")

    resolution_dict: dict[str, tuple[float, float, float]] = {}

    split_ids: list[str]
    for mode, split_ids in zip(["train", "val"], [training_ids, validation_ids]):
        dest_mode: Path = dest_path / mode
        print(f"Slicing {len(split_ids)} pairs to {dest_mode}")

        # Pass two over the data: apply the training-derived mean/std to every patient in train and val 
        pfun: Callable = partial(slice_patient,
                                 dest_path=dest_mode,
                                 source_path=src_path,
                                 shape=tuple(args.shape),
                                 mean=mean,
                                 std=std,
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