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


def remove_small_components(
    mask: np.ndarray,
    min_voxels: int,
    min_fraction: float
) -> tuple[np.ndarray, list[int], int]:

    # Find all separate 3D connected components
    labeled = label(mask, connectivity=1)

    number_components = labeled.max()

    if number_components == 0:
        return mask, [], 0

    # Size of every component
    component_sizes = np.bincount(labeled.ravel())

    # Ignore background
    component_sizes[0] = 0

    largest_size = component_sizes.max()

    # Component must satisfy BOTH:
    # - absolute minimum size
    # - minimum fraction of largest component
    threshold = max(
        min_voxels,
        int(largest_size * min_fraction)
    )

    output = np.zeros_like(mask, dtype=bool)

    kept_sizes = []

    for component_id in range(1, number_components + 1):

        size = component_sizes[component_id]

        if size >= threshold:

            output[labeled == component_id] = True

            kept_sizes.append(int(size))

    return output, kept_sizes, threshold


def postprocess_volume(
    input_path: Path,
    output_path: Path,
    min_voxels: int,
    min_fraction: float
):

    nii = nib.load(str(input_path))

    volume = np.asarray(nii.dataobj)

    volume = np.rint(volume).astype(np.uint8)

    unique_labels = set(np.unique(volume))

    if not unique_labels <= {0, 1, 2, 3, 4}:
        raise ValueError(
            f"Unexpected labels in {input_path}: "
            f"{sorted(unique_labels)}"
        )

    output = np.zeros_like(
        volume,
        dtype=np.uint8
    )

    print()
    print(f"Processing: {input_path.name}")

    for class_id, class_name in LABELS.items():

        mask = volume == class_id

        before = int(mask.sum())

        if before == 0:
            print(
                f"{class_name:<10}: not present"
            )
            continue

        cleaned, kept_sizes, threshold = (
            remove_small_components(
                mask,
                min_voxels=min_voxels,
                min_fraction=min_fraction
            )
        )

        after = int(cleaned.sum())

        output[cleaned] = class_id

        print(
            f"{class_name:<10}: "
            f"{before:>8} -> {after:>8} voxels | "
            f"removed {before - after:>6} | "
            f"threshold={threshold:>5} | "
            f"kept components={len(kept_sizes)}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    header = nii.header.copy()

    header.set_data_dtype(
        np.uint8
    )

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
        description=(
            "Remove small 3D connected components "
            "from segmentation volumes."
        )
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

    parser.add_argument(
        "--min_voxels",
        type=int,
        default=50,
        help=(
            "Minimum number of voxels "
            "for a component to be kept."
        )
    )

    parser.add_argument(
        "--min_fraction",
        type=float,
        default=0.01,
        help=(
            "Minimum component size as fraction "
            "of the largest component."
        )
    )

    args = parser.parse_args()

    files = sorted(
        args.input_dir.glob("*.nii*")
    )

    if len(files) == 0:
        raise FileNotFoundError(
            f"No NIfTI files found in "
            f"{args.input_dir}"
        )

    print(
        f"Found {len(files)} volumes"
    )

    print(
        f"min_voxels   = {args.min_voxels}"
    )

    print(
        f"min_fraction = {args.min_fraction}"
    )

    for input_path in files:

        output_path = (
            args.output_dir
            / input_path.name
        )

        postprocess_volume(
            input_path,
            output_path,
            min_voxels=args.min_voxels,
            min_fraction=args.min_fraction
        )

    print()

    print(
        "Done. Saved post-processed "
        f"volumes to: {args.output_dir}"
    )


if __name__ == "__main__":
    main()