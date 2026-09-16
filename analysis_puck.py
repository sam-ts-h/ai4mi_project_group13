import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

data_dir = Path("data/segthor_part1/data/segthor_part1/train")

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

patient_names = []
dx_values, dy_values, dz_values = [], [], []

for patient in sorted(data_dir.iterdir()):
    ct_path = patient / f"{patient.name}.nii.gz"
    nib_obj = nib.load(str(ct_path))
    dx, dy, dz = nib_obj.header.get_zooms()

    patient_names.append(patient.name)
    dx_values.append(dx)
    dy_values.append(dy)
    dz_values.append(dz)

dx_values = np.array(dx_values)
dy_values = np.array(dy_values)
dz_values = np.array(dz_values)

# --- Plot 1: spacing per patiënt, om de spreiding direct te zien ---
plt.figure(figsize=(10, 5))
x = np.arange(len(patient_names))
plt.plot(x, dx_values, "o-", label="dx (x-richting)")
plt.plot(x, dy_values, "s-", label="dy (y-richting)")
plt.plot(x, dz_values, "^-", label="dz (z-richting, tussen slices)")
plt.xticks(x, patient_names, rotation=90)
plt.ylabel("Spacing (mm)")
plt.title("Voxel-spacing per patiënt")
plt.legend()
plt.tight_layout()
plt.savefig("spacing_per_patient.png", dpi=150)
plt.show()

# --- Plot 2: boxplot, voor een compact overzicht van de spreiding ---
plt.figure(figsize=(6, 5))
plt.boxplot([dx_values, dy_values, dz_values], tick_labels=["dx", "dy", "dz"])
plt.ylabel("Spacing (mm)")
plt.title("Spreiding van voxel-spacing over alle patiënten")
plt.tight_layout()
plt.savefig("spacing_boxplot.png", dpi=150)
plt.show()

# --- Samenvatting + voorstel voor target_spacing ---
print("=== Spacing-statistieken over alle patiënten ===")
for name, values in [("dx", dx_values), ("dy", dy_values), ("dz", dz_values)]:
    print(f"  {name}: min={values.min():.3f}  max={values.max():.3f}  "
          f"median={np.median(values):.3f}  mean={values.mean():.3f}")

print("\n=== Voorgestelde target_spacing (op basis van de mediaan) ===")
print(f"  --target_spacing {np.median(dx_values):.2f} {np.median(dy_values):.2f} {np.median(dz_values):.2f}")
