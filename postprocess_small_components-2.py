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

#Connectivity is 3D 6-connectivity: such that the voxels musst share a face
CONNECTIVITY = 1


def remove_small_components(
    mask: np.ndarray,
    min_voxels: int,
    min_fraction: float
) -> tuple[np.ndarray, list[int], int]:

    #Find all separate 3D components for this organ
    labeled = label(mask, connectivity=CONNECTIVITY)

    number_components = labeled.max()

    if number_components == 0:
        return mask, [], 0

    # Size of every component
    component_sizes = np.bincount(labeled.ravel())

    # Ignore background
    component_sizes[0] = 0

    largest_size = component_sizes.max()
    largest_id = int(np.argmax(component_sizes))

    # Component must satisfy BOTH:
    # - absolute minimum size (the min size of a component)
    # - minimum fraction of largest component
    threshold = max(
        min_voxels,
        int(largest_size * min_fraction)
)

    #Start with an empty mask, later add compontents
    output = np.zeros_like(mask, dtype=bool)

    kept_sizes = []

    #Check size of each component seperate
    for component_id in range(1, number_components + 1):

        size = component_sizes[component_id]

        #always keep the largest component, others only when large enough
        if component_id == largest_id or size >= threshold:
            output[labeled == component_id] = True
            kept_sizes.append(int(size))

    return output, kept_sizes, threshold


#first check wether the segmentation is done correct
def validate_volume(volume: np.ndarray, input_path: Path):

    if volume.ndim != 3:
        raise ValueError(
            f"{input_path.name}: Expected a 3D volume, "
            f"got shape {volume.shape}"
        )

    if not np.all(np.isfinite(volume)):
        raise ValueError(
            f"{input_path.name}: Contains NaN/infinite values."
        )

    if not np.all(volume == np.rint(volume)):
        raise ValueError(
            f"{input_path.name}: Contains non-integer labels."
        )

    unique_labels = set(np.unique(volume).tolist())
    allowed_labels = {0, *LABELS.keys()}

    if not unique_labels.issubset(allowed_labels):
        raise ValueError(
            f"{input_path.name}: Unexpected labels: "
            f"{sorted(unique_labels)}"
        )

    return volume.astype(np.uint8)



def postprocess_volume( input_path: Path, output_path: Path, min_voxels: int,min_fraction: float
):

    #load pred 3D segmentation
    nii = nib.load(str(input_path))
    volume = np.asarray(nii.dataobj)

    #check if the prediction contains only the expected labels
    volume = validate_volume(volume, input_path)

    unique_labels = set(np.unique(volume))

    #creat empty volume for cleaned predictions
    output = np.zeros_like( volume,dtype=np.uint8
    )

    print()
    print(f"Processing: {input_path.name}")

    for class_id, class_name in LABELS.items():

        #select voxels predicted /organ
        mask = volume == class_id

        before = int(mask.sum())

        if before == 0:
            print(
                f"{class_name:<10}: not present"
            )
            continue

        cleaned, kept_sizes, threshold = (
            remove_small_components(mask,min_voxels=min_voxels, min_fraction=min_fraction)
        )

        #count number of voxels left after post-proces
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

    # Check that we only removed voxels and did not change any labels
    assert np.all((output == 0) | (output == volume))
    assert np.count_nonzero(output) <= np.count_nonzero(volume)

    # Keep the original coordinate system information
    qform, qcode = nii.get_qform(coded=True)
    sform, scode = nii.get_sform(coded=True)

    if qform is not None:
        result.set_qform(qform, code=int(qcode))

    if sform is not None:
        result.set_sform(sform, code=int(scode))

    nib.save(
        result,
        str(output_path)
    )


def main():

    #get input and output filders
    parser = argparse.ArgumentParser(
        description="Apply 3D small Connected Component post-processing."
    )

    parser.add_argument( "--input_dir", type=Path, required=True
    )

    parser.add_argument( "--output_dir",type=Path,required=True
    )

    parser.add_argument("--min_voxels",type=int,default=50,help="Min number of voxels for a component to be kept."
    )

    parser.add_argument( "--min_fraction", type=float, default=0.01,help="Min component size as fraction of the largest component."
    )

    args = parser.parse_args()

    # Check if the selected thresholds are valid
    if args.min_voxels < 0:
        parser.error("min_voxels must be >= 0")

    if not np.isfinite(args.min_fraction) or not 0 <= args.min_fraction <= 1:
        parser.error("min_fraction must be between 0 and 1")

    #get full path
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.is_dir():
        raise NotADirectoryError(
            f"Input directory not found: {input_dir}"
        )

    if input_dir == output_dir:
        raise ValueError(
            "Input and output directories must be different."
        )

    #find all #D predictions inthe input folder
    files = sorted( p for p in input_dir.iterdir()
        if p.is_file()
        and p.name.endswith((".nii", ".nii.gz"))
    )

    if not files:
        raise FileNotFoundError( f"No NIfTI files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    existing = [ p.name for p in files
        if (output_dir / p.name).exists()
    ]

    #check we do not want to overwrite results
    if existing:
        raise FileExistsError(
            "Output files already exist: "
            + ", ".join(existing)
            + ". Use a new output directory."
        )

    #process each patient seperately + safe nw predictions
    for input_path in files:

        output_path = output_dir / input_path.name

        postprocess_volume(input_path,output_path, min_voxels=args.min_voxels,min_fraction=args.min_fraction )

    print("\nDone.")
    print(f"Processed {len(files)} volumes.")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()