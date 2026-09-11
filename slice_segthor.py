import pickle
import random
import argparse
import warnings
import nibabel as nib

from pathlib import Path
from functools import partial
from multiprocessing import Pool
from typing import Callable
from PIL import Image
from skimage.transform import resize

import numpy as np

from utils import map_, tqdm_


"""
TODO: Implement image normalisation.
CT images have a wide range of intensity values (Hounsfield units)
Goal: normalize an image array to the range [0, 255]  and return it as a dtype=uint8
Which is compatible with standard image formats (PNG)
"""
def norm_arr(img: np.ndarray) -> np.ndarray:

    min_value = img.min()
    max_value = img.max()
    normalized = (img - min_value) / (max_value - min_value) * 255

    return normalized.astype(np.uint8)


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
    assert set(np.unique(gt)) <= set(range(5))

    return True


"""
TODO: Implement patient slicing.
Context:
  - Given an ID and paths, load the NIfTI CT volume and (if not test_mode) the GT volume.
  - Validate with sanity_ct / sanity_gt.
  - Normalise CT with norm_arr().
  - Slice the 3D volumes into 2D slices, resize to `shape`, and save PNGs.
  - Currently we have groundtruth masks marked as {0,1,2,3,4} but those values are hard to distinguish in a grayscale png.
    Multiplying by 63 maps them to {0,63,126,189,252}, which keeps labels visually distinct in a grayscale PNG.
    You can use the following code, which works for already sliced 2d images:
    gt_slice *= 63
    assert gt_slice.dtype == np.uint8, gt_slice.dtype
    assert set(np.unique(gt_slice)) <= set([0, 63, 126, 189, 252]), np.unique(gt_slice)
  - Return the original voxel spacings (dx, dy, dz).

Hints:
  - Use nibabel to load NIfTI images. // imported
  - Use skimage.transform.resize (tip: anti_aliasing might be useful)
  - The PNG files should be stored in the dest_path, organised into separate subfolders: train/img, train/gt, val/img, and val/gt
  - Use consistent filenames: e.g. f"{id_}_{idz:04d}.png" inside subfolders "img" and "gt"; where idz is the slice index.
"""

def slice_patient(id_: str, dest_path: Path, source_path: Path, shape: tuple[int, int], test_mode=False)\
        -> tuple[float, float, float]:

    id_path: Path = source_path / ("train" if not test_mode else "test") / id_
    ct_path: Path = (id_path / f"{id_}.nii.gz")
    assert id_path.exists()
    assert ct_path.exists()

    # --------- FILL FROM HERE -----------
    ct_nifti = nib.load(str(ct_path))
    ct = np.asarray(ct_nifti.dataobj) #dont use ct_nifti.get_fdata(), because it converts to float, sanity ct wants ints
    x,y,z = ct.shape
    dx,dy,dz = ct_nifti.header.get_zooms() [:3]

    sanity_ct(ct, x,y,z,dx,dy,dz)

    if not test_mode:
        gt_path = id_path / "GT.nii.gz"
        assert gt_path.exists()

        gt_nifti = nib.load(str(gt_path))
        gt = np.asarray(gt_nifti.dataobj)
        sanity_gt(gt, ct)

    ct = norm_arr(ct)

    #checks if the normalisation worked
    assert ct.dtype == np.uint8
    assert ct.min() == 0
    assert ct.max() == 255

    img_dest_path = dest_path / "img"
    img_dest_path.mkdir(parents=True, exist_ok=True)

    if not test_mode:
        gt_dest_path = dest_path / "gt"
        gt_dest_path.mkdir(parents=True, exist_ok=True)

    for idz in range(z):
        ct_slice = ct[:, :, idz] #all x and y slices but only one z slice (2d)

        ct_slice_resized = resize(ct_slice, shape, preserve_range=True, anti_aliasing=True)
        ct_slice_resized = ct_slice_resized.astype(np.uint8)

        filename = f"{id_}_{idz:04d}.png"
        Image.fromarray(ct_slice_resized).save(img_dest_path / filename)

        if not test_mode:
            gt_slice = gt[:, :, idz]

            gt_slice_resized = resize(gt_slice, shape, order=0, preserve_range=True, anti_aliasing=False)
            #GT is a label, we use oder=0 to avoid interpolation and use nearest neighbor (0/1/2//3/4) values. 
            #Preserve range to keep the original values and anti aliasing = false because we dont want smoothing

            gt_slice_resized = gt_slice_resized.astype(np.uint8)
            gt_slice_resized *= 63 #from classes to PNG values 

            assert set(np.unique(gt_slice_resized)) <= {0, 63, 126, 189, 252} 

            Image.fromarray(gt_slice_resized).save(gt_dest_path / filename)



    return dx,dy,dz


"""
TODO: Implement a simple train/val split.
Requirements:
  - List patient IDs from <src_path>/train (folder names).
  - Shuffle them (respect a seed set in main()).
  - Take the first `retains` as validation, and the rest as training.
  - Return (training_ids, validation_ids).
"""

def get_splits(src_path: Path, retains: int) -> tuple[list[str], list[str]]:
    train_path = src_path / "train"

    patient_ids = []

    for patient_path in train_path.iterdir():
        if patient_path.is_dir():
            patient_ids.append(patient_path.name)

    patient_ids = sorted(patient_ids)

    random.shuffle(patient_ids)

    validation_ids = patient_ids[:retains]
    training_ids = patient_ids[retains:]

    return (training_ids, validation_ids)



def main(args: argparse.Namespace):
    src_path: Path = Path(args.source_dir)
    dest_path: Path = Path(args.dest_dir)
    if not dest_path.exists():
        dest_path.mkdir(parents=True, exist_ok=True)

    assert src_path.exists()
    assert dest_path.exists()

    training_ids: list[str]
    validation_ids: list[str]
    training_ids, validation_ids = get_splits(src_path, args.retains)


    resolution_dict: dict[str, tuple[float, float, float]] = {}

    for mode, split_ids in zip(["train", "val"], [training_ids, validation_ids]):
        dest_mode: Path = dest_path / mode
        print(f"Slicing {len(split_ids)} pairs to {dest_mode}")

        pfun: Callable = partial(slice_patient,
                                 dest_path=dest_mode,
                                 source_path=src_path,
                                 shape=tuple(args.shape))

        resolutions: list[tuple[float, float, float]]
        iterator = tqdm_(split_ids)
        resolutions = list(map(pfun, iterator))

        for key, val in zip(split_ids, resolutions):
            resolution_dict[key] = val

    with open(dest_path / "spacing.pkl", 'wb') as f:
        pickle.dump(resolution_dict, f, pickle.HIGHEST_PROTOCOL)
        print(f"Saved spacing dictionnary to {f}")




def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "Slicing parameters")

    parser.add_argument('--source_dir', type=str, required=True)
    parser.add_argument('--dest_dir', type=str, required=True)
    parser.add_argument('--shape', type=int, nargs="+", default=[256, 256])
    parser.add_argument('--retains', type=int, default=10, help="Number of retained patient for the validation data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    random.seed(args.seed)
    print(args)

    return args

if __name__ == "__main__":
    main(get_args())