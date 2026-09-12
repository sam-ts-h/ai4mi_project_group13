import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

#Inspectation of what the CT and GT data looks like
img_ct = nib.load("data/segthor_part1/data/segthor_part1/train/Patient_01/Patient_01.nii.gz")
img_gt = nib.load("data/segthor_part1/data/segthor_part1/train/Patient_01/GT.nii.gz")
ct = img_ct.get_fdata()
gt = img_gt.get_fdata()
print(ct.shape, img_ct.header.get_zooms())
print(ct.shape, img_gt.header.get_zooms())

ct = np.asarray(img_ct.dataobj)
gt = np.asarray(img_gt.dataobj)

print(ct.dtype, ct.shape)
print("min:", ct.min(), "max:", ct.max(), "mean:", ct.mean())

plt.hist(ct.ravel(), bins=200)
plt.xlabel("Hounsfield Units (HU)")
plt.ylabel("Voxel count")
plt.show()

print(gt.dtype, gt.shape)
print("min:", gt.min(), "max:", gt.max(), "mean:", gt.mean())

plt.hist(gt.ravel(), bins=200)
plt.xlabel("Hounsfield Units (HU)")
plt.ylabel("Voxel count")
plt.show()

#All patiënts overview:
data = Path("data/segthor_part1/data/segthor_part1/train")

all_values = []
for patient in data.iterdir():
    ct_path = patient / f"{patient.name}.nii.gz"
    ct = np.asarray(nib.load(str(ct_path)).dataobj)
    all_values.append(ct.ravel())

all_values = np.concatenate(all_values)
print("min:", all_values.min(), "max:", all_values.max(), "mean:", all_values.mean())

plt.hist(all_values, bins=200)
plt.xlabel("Hounsfield Units (HU)")
plt.ylabel("Voxel count")
plt.show()







