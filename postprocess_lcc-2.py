from pathlib import Path
import argparse

import nibabel as nib
import numpy as np
from skimage.measure import label


#because nog llc is applied on background, we do not include this
LABELS = {
    1: "Esophagus",
    2: "Heart",
    3: "Trachea",
    4: "Aorta",
}

#Connectivity is 3D 6-connectivity: such that the voxels musst share a face
CONNECTIVITY = 1


#only keep the largest 3D component
def largest_connected_component(mask: np.ndarray):

    labeled = label(mask, connectivity=CONNECTIVITY)

    n_components = int(labeled.max())

    #If there are no predicted voxels for this class
    if n_components == 0:
        return mask.copy(), 0

    component_sizes = np.bincount(labeled.ravel())

    #Label 0 is background, we ignore it
    component_sizes[0] = 0

    largest_id = int(np.argmax(component_sizes))
    cleaned_mask = labeled == largest_id

    return cleaned_mask, n_components #return the binary mask of the largest component and number of componrents before processing


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



#3D LLC independent on each foreground class
def postprocess_volume(input_path: Path, output_path: Path):

    #Load the 3D seg (predicted)
    nii = nib.load(str(input_path))
    volume = np.asarray(nii.dataobj)

    #check the expected labels
    volume = validate_volume(volume, input_path)

    #Output is first empty, later add largest compinent
    output = np.zeros_like(volume, dtype=np.uint8)

    print(f"\nProcessing: {input_path.name}")

    total_removed = 0

    #LCC is applied for every organ seperately
    for class_id, class_name in LABELS.items():

        #So only the voxels of 1 organ 
        mask = volume == class_id

        before = int(mask.sum())

        if before == 0:
            print(f"{class_name:<10}: not present")
            continue

        #Find the largest connected component of this organ
        cleaned, n_components = largest_connected_component(mask)

        after = int(cleaned.sum())
        removed = before - after
        removed_pct = 100 * removed / before

        #add the organ back
        output[cleaned] = class_id

        total_removed += removed

        print(
            f"  {class_name:<10}: "
            f"components={n_components:<4} | "
            f"voxels={before:>8} -> {after:>8} | "
            f"removed={removed:>8} ({removed_pct:.2f}%)"
        )

    #check that we did not change labels, only voxels removing
    assert np.all((output == 0) | (output == volume))
    assert np.count_nonzero(output) <= np.count_nonzero(volume)

    # Preserve spatial information
    header = nii.header.copy()
    header.set_data_dtype(np.uint8)

    result = nib.Nifti1Image(
        output,
        nii.affine,
        header
    )

    # Preserve coordinate system metadata
    qform, qcode = nii.get_qform(coded=True)
    sform, scode = nii.get_sform(coded=True)

    if qform is not None: 
        result.set_qform(qform, code=int(qcode))

    if sform is not None:
        result.set_sform(sform, code=int(scode))

    nib.save(result, str(output_path))

    print(f"  Total foreground voxels removed: {total_removed}")




def main():

    #get input and output filders
    parser = argparse.ArgumentParser( description="Apply 3D Largest Connected Component post-processing."
    )

    parser.add_argument( "--input_dir", type=Path, required=True
    )

    parser.add_argument( "--output_dir",type=Path,required=True
    )

    args = parser.parse_args()

    #get full path
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.is_dir():
        raise NotADirectoryError( f"Input directory not found: {input_dir}"
        )

    if input_dir == output_dir:
        raise ValueError( "Input and output directories must be different."
        )

    #find all #D predictions inthe input folder
    files = sorted( p for p in input_dir.iterdir()
        if p.is_file()
        and p.name.endswith((".nii", ".nii.gz"))
    )

    if not files:
        raise FileNotFoundError( f"No NIfTI files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    existing = [
        p.name for p in files
        if (output_dir / p.name).exists()
    ]

    #check we do not want to overwrite results
    if existing:
        raise FileExistsError( "Output files already exist: " + ", ".join(existing) + ". Use a new output directory."
        )

    #process each patient seperately + safe nw predictions
    for input_path in files:

        output_path = output_dir / input_path.name

        postprocess_volume(input_path, output_path)

    print("\nDone.")
    print(f"Processed {len(files)} volumes.")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()