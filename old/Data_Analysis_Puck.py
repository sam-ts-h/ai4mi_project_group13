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
print(gt.shape, img_gt.header.get_zooms())

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

CLASS_NAMES = {0: "background", 1: "esophagus", 2: "heart", 3: "trachea", 4: "aorta"}

labels, counts = np.unique(gt.ravel(), return_counts=True)
plt.bar(labels, counts)
plt.xticks(labels, [CLASS_NAMES[l] for l in labels])
plt.ylabel("Voxel count")
plt.title("Patient 1")
plt.show()

#All patiënts hu values overview:
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

#All patiënts class balance overview:
all_gt_values = []
for patient in data.iterdir():
    gt_path = patient / f"GT.nii.gz"
    gt = np.asarray(nib.load(str(gt_path)).dataobj)
    all_gt_values.append(gt.ravel())

all_gt_values = np.concatenate(all_gt_values)
print("mean:", all_gt_values.mean())

labels, counts = np.unique(all_gt_values, return_counts=True)
plt.bar(labels, counts)
plt.xticks(labels, [CLASS_NAMES[l] for l in labels])
plt.ylabel("Voxel count")
plt.title("Class balance (all patients)")
plt.show()

#Insight in parts:
class_0 = ((all_gt_values == 0).sum() / len(all_gt_values)) * 100
class_1 = ((all_gt_values == 1).sum() / len(all_gt_values)) * 100
class_2 = ((all_gt_values == 2).sum() / len(all_gt_values)) * 100
class_3 = ((all_gt_values == 3).sum() / len(all_gt_values)) * 100

print(f'class 0: {class_0:.2f}%')
print(f'class 1: {class_1:.2f}%')
print(f'class 2: {class_2:.2f}%')
print(f'class 3: {class_3:.2f}%')

#HU distribution per class:
CLASS_NAMES = {0: "class0", 1: "class1", 2: "class2", 3: "class3"}
values_per_class = {label: [] for label in CLASS_NAMES}

for patient in sorted(data.iterdir()):
    ct = np.asarray(nib.load(str(patient / f"{patient.name}.nii.gz")).dataobj)
    gt = np.asarray(nib.load(str(patient / "GT.nii.gz")).dataobj)

    for label in CLASS_NAMES:
        mask = gt == label
        values_per_class[label].append(ct[mask])

for label, name in CLASS_NAMES.items():
    if values_per_class[label]:
        values = np.concatenate(values_per_class[label])

    else:
        np.array([])
    print(f"{name:12s}: n={values.size:>10,d}  mean={values.mean():>8.1f}  std={values.std():>7.1f}")

# boxplot: 1 box per klasse, laat meteen outliers en spreiding zien
present = [label for label in CLASS_NAMES if
           values_per_class[label] and np.concatenate(values_per_class[label]).size > 0]
plt.boxplot([np.concatenate(values_per_class[l]) for l in present],
            tick_labels=[CLASS_NAMES[l] for l in present])
plt.ylabel("Hounsfield Units (HU)")
plt.title("HU-distributie per class")
plt.show()



