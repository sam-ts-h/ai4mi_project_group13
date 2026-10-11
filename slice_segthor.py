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
from skimage.transform import resize
from scipy.ndimage import distance_transform_edt

from utils import map_, tqdm_
from preprocessing_common import (GRID_ROWS, GRID_COLS,
                                  resample_image_slice, resample_mask_slice,
                                  body_mask_2d, body_centroid, crop_or_pad_to_grid)

# Raw HU floor when clipping is OFF for a given ablation run: not a design choice, it's just the dataset's own guaranteed floor.
# Used only to define what "air" means for the padding fill value when --clip is not passed.
RAW_AIR_FLOOR = -1000.0

NUM_CLASSES: int = 5

def clip_ct(img: np.ndarray, clip_min: float, clip_max: float) -> np.ndarray:
    """
    Clip raw HU values to the dataset's foreground percentile range, no rescale to 0-255. 
    This replaces the old norm_arr() lossy per-slice min-max normalization: the network now receives real, clipped HU values
    directly, consistent across every slice and every patient (the same real tissue density always maps to the same value
    (assuming consistent HU values from scanner etc), regardless of what else happens to be in that particular slice, 
    which per-slice min-max could not guarantee).
    """
    return np.clip(img.astype(np.float32), clip_min, clip_max)


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

    return ct, gt, (dx, dy, dz)

def compute_clip_range(training_ids: list[str], source_path: Path, n_samples: int = 20_000) -> tuple[float, float]:
    """
    Clip range = 0.5 / 99.5 percentiles of the foreground (organ-labeled) HU values, computed over the training patients
    only, so validation data never influences the preprocessing (avoid data leakage). Same calculation as analyze_intensity() 
    in eda_segthor.py: per patient and per organ class, subsample up to n_samples voxels, then pool everything. The subsampling uses a fixed 
    rng, so re-running gives the identical range for reproducibility.
    """
    rng = np.random.default_rng(0)
    samples = []
    for id_ in tqdm_(training_ids, desc="Computing clip range"):
        ct, gt, _ = load_patient_ct(id_, source_path, test_mode=False)
        for k in np.unique(gt):
            if k == 0:  # background is not foreground
                continue
            vals = ct[gt == k]
            if vals.size > n_samples:
                vals = rng.choice(vals, n_samples, replace=False)
            samples.append(vals)
    low, high = np.percentile(np.concatenate(samples), [0.5, 99.5])
    return float(low), float(high)

def compute_window_range(training_ids: list[str], source_path: Path, class_id: int,
                         low_pct: float, high_pct: float) -> tuple[float, float]:
    """
    Cutoffs of the soft-tissue window (second input channel): the low/high percentile of the raw HU values inside the 
    ground-truth label class_id (1 = esophagus), pooled over the training patients only, so validation data never influences 
    them (avoid data leakage). Uses every labeled voxel (no subsampling), so the result is exact and deterministic. 
    analyze_soft_tissue_window.py shows the evidence for which percentiles to pick.
    """
    values = []
    for id_ in tqdm_(training_ids, desc="Computing window range"):
        ct, gt, _ = load_patient_ct(id_, source_path, test_mode=False)
        values.append(ct[gt == class_id].astype(np.float32))
    values = np.concatenate(values)
    assert values.size > 0, f"no voxels labeled {class_id} in the training patients"
    low, high = np.percentile(values, [low_pct, high_pct])
    assert high > low, f"empty window [{low}, {high}]"
    return float(low), float(high)


def compute_global_stats(training_ids: list[str], source_path: Path,
                         clip_range: tuple[float, float], use_resample: bool, shape: tuple[int, int]):
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
        ct, _, (dx, dy, _) = load_patient_ct(id_, source_path, test_mode=False)
        ct = clip_ct(ct, *clip_range) if clip_range else ct.astype(np.float32)
        z = ct.shape[2]
        for idz in range(z):
            if use_resample:
                resampled = resample_image_slice(ct[:, :, idz], (dx, dy))
            else:
                resampled = resize(ct[:, :, idz], shape, order=1, mode="constant",
                                   preserve_range=True, anti_aliasing=False).astype(np.float32)
            # Accumulate in float64: total_count will run into the hundreds of millions of pixels across the full training set, 
            # and a float32 running sum could start losing real precision.
            total_sum += resampled.sum(dtype=np.float64)
            total_sumsq += np.sum(resampled.astype(np.float64) ** 2)
            total_count += resampled.size

    mean = total_sum / total_count
    variance = (total_sumsq / total_count) - mean ** 2
    std = float(np.sqrt(max(variance, 1e-8)))  # guard against a tiny negative from float rounding

    return float(mean), std


def slice_patient(id_: str, dest_path: Path, source_path: Path, shape: tuple[int, int], clip_range: tuple[float, float], use_resample: bool, 
                  use_normalize: bool, mean: float, std: float, pad_fill_value: float, use_distmap: bool = False,
                  window_range: tuple[float, float] | None = None, test_mode: bool = False):
    ct, gt, (dx, dy, dz) = load_patient_ct(id_, source_path, test_mode)
    z = ct.shape[2]

    # Second channel (--window): the raw HU clipped to the narrow soft-tissue window. Built from the raw values, before the wide
    # clip below, so it does not depend on the clip range at all.
    win_ct = np.clip(ct.astype(np.float32), *window_range) if window_range else None

    # Do the percentile clipping if needed.
    ct = clip_ct(ct, *clip_range) if clip_range else ct.astype(np.float32)

    img_save_path: Path = Path(dest_path, "img")
    gt_save_path: Path = Path(dest_path, "gt")
    img_save_path.mkdir(parents=True, exist_ok=True)
    gt_save_path.mkdir(parents=True, exist_ok=True)

    # 3D signed distance maps on the raw gt, before any resampling or cropping. spacing in mm so the
    # distances are real 3D mm and stay valid after resampling and croppin
    if use_distmap:
        distSavePath: Path = Path(dest_path, "distmap")
        distSavePath.mkdir(parents=True, exist_ok=True)
        distMaps = np.zeros((NUM_CLASSES, *gt.shape), dtype=np.float16)
        for k in range(NUM_CLASSES):
            mask = gt == k
            # If no borders dont run code, errors
            if mask.any() and not mask.all():
                outside = ~mask
                distOutside = distance_transform_edt(outside, sampling=(dx, dy, dz))
                distInside = distance_transform_edt(mask, sampling=(dx, dy, dz))
                distMaps[k] = distOutside * outside - distInside * mask

    # stitch.py needs the crop center of every slice to put predictions back in the original scan
    cropCenters = {}

    for idz in range(z):
        # Resample each slice to a common in-plane spacing. This is what makes the same real-world organ size occupy the same
        # pixel count regardless of which patient's original spacing it came from.
        if use_resample:
            #use_resample=True:  resample to common in-plane spacing, output size varies per patient.
            img_slice = resample_image_slice(ct[:, :, idz], (dx, dy))
            gt_slice = resample_mask_slice(gt[:, :, idz], (dx, dy))
        else:
            #use_resample=False: reproduce the original behavior from before any of this preprocessing work (plain resize to a fixed `shape`)
            img_slice = resize(ct[:, :, idz], shape, order=1, mode="constant",
                            preserve_range=True, anti_aliasing=False).astype(np.float32)
            gt_slice = resize(gt[:, :, idz], shape, order=0, mode="constant",
                            preserve_range=True, anti_aliasing=False).astype(gt[:, :, idz].dtype)
            
        assert img_slice.shape == gt_slice.shape

        # distances are smooth so linear like the image, not nearest like the gt
        if use_distmap:
            distSlice = np.zeros((NUM_CLASSES, *gt_slice.shape), dtype=np.float32)
            for k in range(NUM_CLASSES):
                classSlice = distMaps[k, :, :, idz].astype(np.float32)
                if use_resample:
                    distSlice[k] = resample_image_slice(classSlice, (dx, dy))
                else:
                    distSlice[k] = resize(classSlice, shape, order=1, mode="constant",
                                          preserve_range=True, anti_aliasing=False)

        if use_resample:
            # Crop/pad is only meaningful after resampling, since original fixed resize already produces same grid size
            # Body detection happens here, on the resampled but not yet normalized HU slice, since  body_mask_2d's thresholds are 
            # in real HU units, which are meaningless once z-scored. 
            mask = body_mask_2d(img_slice)
            center = body_centroid(mask)
            if center is None:
                # No tissue above threshold at all in this slice (e.g. a mostly empty slice right at the very top/bottom of the scan),
                # fall back to the slice's own geometric center.
                center = (img_slice.shape[0] / 2, img_slice.shape[1] / 2)

        # Normalize after resampling, using taining mean/std
        # The same fixed mean/std is applied identically whether this call is processing a train or val patient,
        # so val data is normalized exactly the way it will be at real inference time (fixed stats, no peeking at val/test data).
        if use_normalize: 
            img_slice = ((img_slice - mean) / std).astype(np.float32)
        else:
            img_slice = img_slice.astype(np.float32)

        gt_slice *= 63
        assert gt_slice.dtype == np.uint8, gt_slice.dtype
        # assert set(np.unique(gt_slice)) <= set(range(5))
        assert set(np.unique(gt_slice)) <= set([0, 63, 126, 189, 252]), np.unique(gt_slice)


        # Crop/pad both the image and GT to the same fixed grid, centered on the same body centroid. 
        # Image padding uses pad_fill_value (the normalized-space equivalent of real air HU, precomputed once in main()).
        # GT padding uses 0 (background class) since padded regions have no real anatomy.
        if use_resample:
            img_slice = crop_or_pad_to_grid(img_slice, GRID_ROWS, GRID_COLS, center, fill_value=pad_fill_value)
            gt_slice = crop_or_pad_to_grid(gt_slice, GRID_ROWS, GRID_COLS, center, fill_value=0)
            assert img_slice.shape == (GRID_ROWS, GRID_COLS)
            assert gt_slice.shape == (GRID_ROWS, GRID_COLS)

            # same center as the image so the distances stay on top of the gt.
            # Padding is background far from every organ: so the max (furthest outside) for the organs,
            # and the min (deepest inside) for the background class
            if use_distmap:
                croppedDist = np.zeros((NUM_CLASSES, GRID_ROWS, GRID_COLS), dtype=np.float32)
                for k in range(NUM_CLASSES):
                    if k == 0:
                        fillValue = distSlice[k].min()
                    else:
                        fillValue = distSlice[k].max()
                    croppedDist[k] = crop_or_pad_to_grid(distSlice[k], GRID_ROWS, GRID_COLS, center, fill_value=fillValue)
                distSlice = croppedDist
        # Second channel: same geometry as the first (same resampling, same body centroid, same crop/pad), only the HU window differs.
        # Scaled to [0, 1] with the window bounds, so air (clipped to the window floor) is exactly 0, which is also its pad value.
        if window_range is not None:
            w_lo, w_hi = window_range
            if use_resample:
                win_slice = resample_image_slice(win_ct[:, :, idz], (dx, dy))
            else:
                win_slice = resize(win_ct[:, :, idz], shape, order=1, mode="constant",
                                   preserve_range=True, anti_aliasing=False)
            win_slice = ((win_slice - w_lo) / (w_hi - w_lo)).astype(np.float32)
            if use_resample:
                win_slice = crop_or_pad_to_grid(win_slice, GRID_ROWS, GRID_COLS, center, fill_value=0.0)
            img_slice = np.stack([img_slice, win_slice])  # (2, H, W); without --window the file stays (H, W) as before

        filename_stem = f"{id_}_{idz:04d}"
        if use_resample:
            cropCenters[filename_stem] = center

        # Image saved losslessly as .npy
        np.save(str(img_save_path / f"{filename_stem}.npy"), img_slice)

        # GT stays a normal PNG sincediscrete class labels, so no precision to lose, 
        # and this keeps it directly viewable/compatible with viewer.py as before.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            imsave(str(gt_save_path / f"{filename_stem}.png"), gt_slice)

        if use_distmap:
            np.save(str(distSavePath / f"{filename_stem}.npy"), distSlice.astype(np.float16))

    return (dx, dy, dz), cropCenters


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

    # Clip range from the training patients only (avoid data leakage)
    if args.clip:
        print(">> Computing the clip range over the training set...")
        clip_range = compute_clip_range(training_ids, src_path )
        print(f">> clip range = [{clip_range[0]:.1f}, {clip_range[1]:.1f}]")
    else:
        clip_range = None

    if args.normalize:
        # Pass one over data: Compute dataset normalization stats from training data only (avoid data leakage)
        print(">> Computing normalization statistics over the training set...")
        mean, std = compute_global_stats(training_ids, src_path, clip_range, args.resample, tuple(args.shape))
        print(f">> mean={mean:.3f}, std={std:.3f}")
    else:
        mean, std = None, None
        print(">> --normalize not set, so skipping stats computation")

    # Precompute once what  "real air" (CLIP_MIN) becomse after normalization. Used as the fill value for padded regions in
    # slice_patient(), so padding represents actual air in the same normalized space the network sees, rather than an arbitrary value.
    # Air floor is CLIP_MIN if clipping is on for this run, otherwise the dataset's raw HU floor (RAW_AIR_FLOOR). 
    # If normalizing, that air value also needs to go through the same z-score transform real pixels get; if not, it's used exactly as-is.
    air_value = clip_range[0] if args.clip else RAW_AIR_FLOOR
    pad_fill_value = (air_value - mean) / std if args.normalize else air_value

    # Soft-tissue window for the optional second channel: cutoffs from the training patients only (avoid data leakage)
    if args.window:
        print(">> Computing the soft-tissue window over the training set...")
        window_range = compute_window_range(training_ids, src_path, args.window_class, *args.window_pct)
        print(f">> window = [{window_range[0]:.1f}, {window_range[1]:.1f}] HU "
              f"(percentiles {args.window_pct[0]:g}/{args.window_pct[1]:g} of class {args.window_class})")
    else:
        window_range = None

    # Save stats for future efficiency
    dest_path.mkdir(parents=True, exist_ok=True)
    with open(dest_path / "normalization_stats.json", "w") as f:
        # pad_fill_values has one entry per input channel (the window channel is scaled to [0, 1] with air at 0); the dataset's
        # rotation/translation augmentation uses it to fill empty corners per channel.
        json.dump({"mean": mean, "std": std, "pad_fill_value": pad_fill_value, "window_range": window_range,
                   "pad_fill_values": [pad_fill_value] + ([0.0] if window_range else [])}, f, indent=2)
        print(f"Saved normalization stats to {f.name}")

    resolution_dict: dict[str, tuple[float, float, float]] = {}
    cropCenters = {}

    split_ids: list[str]
    for mode, split_ids in zip(["train", "val"], [training_ids, validation_ids]):
        dest_mode: Path = dest_path / mode
        print(f"Slicing {len(split_ids)} pairs to {dest_mode}")

        # Pass two over the data: apply the training-derived mean/std to every patient in train and val 
        pfun: Callable = partial(slice_patient,
                                 dest_path=dest_mode,
                                 source_path=src_path,
                                 shape=tuple(args.shape),
                                 clip_range=clip_range,
                                 use_resample=args.resample,
                                 use_normalize=args.normalize,
                                 mean=mean,
                                 std=std,
                                 pad_fill_value=pad_fill_value,
                                 use_distmap=args.distmap,
                                 window_range=window_range,
                                 test_mode=mode == 'test')
        iterator = tqdm_(split_ids)
        match args.process:
            case 1:
                results = list(map(pfun, iterator))
            case -1:
                results = Pool().map(pfun, iterator)
            case _ as p:
                results = Pool(p).map(pfun, iterator)

        for key, (resolution, centers) in zip(split_ids, results):
            resolution_dict[key] = resolution
            cropCenters.update(centers)

    with open(dest_path / "spacing.pkl", 'wb') as f:
        pickle.dump(resolution_dict, f, pickle.HIGHEST_PROTOCOL)
        print(f"Saved spacing dictionnary to {f}")

    if args.resample:
        with open(dest_path / "crop_centers.pkl", 'wb') as f:
            pickle.dump(cropCenters, f, pickle.HIGHEST_PROTOCOL)
            print(f"Saved crop centers to {f.name}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Slicing parameters')
    parser.add_argument('--source_dir', type=str, required=True)
    parser.add_argument('--dest_dir', type=str, required=True)

    parser.add_argument('--shape', type=int, nargs="+", default=[544, 352])
    parser.add_argument('--retains', type=int, default=25, help="Number of retained patient for the validation data")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--process', '-p', type=int, default=1,
                        help="The number of cores to use for processing")
    parser.add_argument('--clip', action='store_true',
                        help="Clip raw HU values to the dataset's foreground percentile range.")
    parser.add_argument('--resample', action='store_true',
                        help="Resample to a common in-plane spacing, with anti-aliasing, followed by body-centroid crop/pad to a fixed grid."
                             "If not set, falls back to the original fixed-shape stretch resize.")
    parser.add_argument('--normalize', action='store_true',
                        help="Z-score normalize using training set mean/std.")
    parser.add_argument('--distmap', action='store_true',
                        help="Also save 3D signed distance maps (computed before resampling/cropping) for the boundary loss.")
    parser.add_argument('--window', action='store_true',
                        help="Add a second input channel: raw HU clipped to a narrow soft-tissue window and scaled to [0, 1]. "
                             "Requires --window_pct. The saved image becomes (2, H, W).")
    parser.add_argument('--window_pct', type=float, nargs=2, metavar=('LOW', 'HIGH'), default=None,
                        help="Low/high percentile of the labeled voxels of --window_class that set the window cutoffs "
                             "(see analyze_soft_tissue_window.py for the evidence).")
    parser.add_argument('--window_class', type=int, default=1, help="Class whose HU values define the window (1 = esophagus).")
    args = parser.parse_args()
    if args.window and args.window_pct is None:
        parser.error("--window needs --window_pct LOW HIGH (e.g. --window_pct 2.5 99.5)")
    random.seed(args.seed)

    print(args)

    return args


if __name__ == "__main__":
    main(get_args())