import nibabel as nib
import numpy as np
from scipy import ndimage

# Pas dit aan naar jouw GT-bestand
gt_path = "data/segthor_part1/train/Patient_01/GT.nii.gz"

# Welke label bevat aorta + ander orgaan?
COMBINED_LABEL = 1

gt = nib.load(gt_path).get_fdata()

print("Volume:", gt.shape)
print("Labels:", np.unique(gt))

print("\nSlices met meer dan 1 component:")

for z in range(gt.shape[2]):

    # Pak één 2D slice
    mask = gt[:, :, z] == COMBINED_LABEL

    # Zoek afzonderlijke gebieden
    components, number = ndimage.label(mask)

    if number > 1:
        sizes = ndimage.sum(mask, components, range(1, number + 1))

        print(
            f"slice {z}: "
            f"{number} components, "
            f"groottes = {sizes.astype(int)}"
        )