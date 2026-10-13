
from pathlib import Path
import argparse

import numpy as np
import nibabel as nib
from scipy.ndimage import binary_closing


def apply_closing(prediction, radius=1):
    #Apply 3D morphological closing to each organ separately

    result = prediction.copy()

    #Create a spherical kernel
    z, y, x = np.ogrid[ -radius:radius+1,-radius:radius+1,-radius:radius+1]
    kernel = (x*x + y*y + z*z) <= radius*radius

    for organ in range(1, 5):
        mask = prediction == organ

        #Pad to avoid removing voxels at image boundaries
        padded = np.pad(mask, radius, constant_values=False)
        closed = binary_closing(padded, structure=kernel)
        closed = closed[radius:-radius,radius:-radius,radius:-radius]

        #only add voxels that were background
        #we dont want to overwrite other organs!!
        new_voxels = closed & (result == 0)
        result[new_voxels] = organ

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--radius", type=int, default=1)
    args = parser.parse_args()

    if args.radius < 1:
        parser.error("radius must be at least 1")

    args.output.mkdir(parents=True, exist_ok=True)

    for path in sorted(args.input.glob("*.nii.gz")):
        img = nib.load(path)
        prediction = np.asarray(img.dataobj).astype(np.uint8)

        result = apply_closing(prediction, args.radius)

        new_img = nib.Nifti1Image(result,affine=img.affine, header=img.header.copy())
        new_img.set_data_dtype(np.uint8)

        nib.save(new_img, args.output / path.name)

        changed = np.count_nonzero(prediction != result)
        print(f"{path.name}: {changed} voxels changed")


if __name__ == "__main__":
    main()
