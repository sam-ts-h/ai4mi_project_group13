from pathlib import Path
import argparse

import nibabel as nib
import numpy as np
from skimage.measure import label


LABELS = {
    1: "Esophagus",
    2: "Heart",
    3: "Trachea",
    4: "Aorta",
}


def largest_connected_component(mask: np.ndarray) -> np.ndarray:

    labeled = label(mask, connectivity=1)

    if labeled.max() == 0:
        return mask

    component_sizes = np.bincount(labeled.ravel())

    # Label 0 is background, so ignore it
    component_sizes[0] = 0

    largest_component = component_sizes.argmax()

    return labeled == largest_component


def postprocess_volume(input_path: Path, output_path: Path):

    nii = nib.load(str(input_path))

    volume = np.asarray(nii.dataobj)

    # Make sure labels are integers
    volume = np.rint(volume).astype(np.uint8)

    unique_labels = set(np.unique(volume))

    if not unique_labels <= {0, 1, 2, 3, 4}:
        raise ValueError(
            f"Unexpected labels in {input_path}: {sorted(unique_labels)}"
        )

    output = np.zeros_like(volume, dtype=np.uint8)

    print()
    print(f"Processing: {input_path.name}")

    for class_id, class_name in LABELS.items():

        mask = volume == class_id

        before = int(mask.sum())

        if before == 0:
            print(f"{class_name:<10}: not present")
            continue

        largest = largest_connected_component(mask)

        after = int(largest.sum())

        output[largest] = class_id

        print(
            f"{class_name:<10}: "
            f"{before:>8} -> {after:>8} voxels "
            f"(removed {before - after})"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = nii.header.copy()
    header.set_data_dtype(np.uint8)

    result = nib.Nifti1Image(
        output,
        nii.affine,
        header
    )

    nib.save(
        result,
        str(output_path)
    )


def main():

    parser = argparse.ArgumentParser(
        description="Apply 3D Largest Connected Component post-processing."
    )

    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True
    )

    args = parser.parse_args()

    files = sorted(args.input_dir.glob("*.nii*"))

    if len(files) == 0:
        raise FileNotFoundError(
            f"No NIfTI files found in {args.input_dir}"
        )

    print(f"Found {len(files)} volumes")

    for input_path in files:

        output_path = args.output_dir / input_path.name

        postprocess_volume(
            input_path,
            output_path
        )

    print()
    print(f"Done. Saved post-processed volumes to: {args.output_dir}")


if __name__ == "__main__":

    main()