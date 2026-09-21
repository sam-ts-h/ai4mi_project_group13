import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

data_dir = Path("data/segthor_part1/train")

#1 Count the amount of patients in the dataset
patients = 0
for patient in data_dir.iterdir():
    patients += 1

print(f"There are {patients} patiënts in this dataset")

#2 What is the shape of these images:
shape = []
voxels_CT = []
voxels_GT = []
for patient in data_dir.iterdir():
    img_ct = nib.load(patient / f"{patient.name}.nii.gz")
    img_gt = nib.load(patient / "GT.nii.gz")
    ct = img_ct.get_fdata()
    gt = img_gt.get_fdata()
    print(ct.shape, img_ct.header.get_zooms())
    print(gt.shape, img_gt.header.get_zooms())
    shape.append(ct.shape)
    shape.append(gt.shape)
    voxels_CT.append(img_ct.header.get_zooms()[:3])
    voxels_GT.append(img_gt.header.get_zooms()[:3])

counts_CT = Counter(voxels_CT)
counts_GT = Counter(voxels_GT)
print("\nVoxel sizes and number of patients:")

for voxel, count in counts_CT.most_common():
    print(f"{voxel}: {count}")
print("\nVoxel sizes GT and number of patients:")
for voxel_GT, count in counts_GT.most_common():
    print(f"{voxel_GT}: {count}")

#creating a plot of the voxel size count
unique_voxels = sorted(set(counts_CT) | set(counts_GT))

ct_values = []
gt_values = []
for v in unique_voxels:
    ct_values.append(counts_CT.get(v,0))
    gt_values.append(counts_GT.get(v,0))

x = np.arange(len(unique_voxels))
width = 0.4

plt.figure(figsize=(8, 6))

plt.bar(x - width/2, ct_values, width, label="CT")
plt.bar(x + width/2, gt_values, width, label="GT")

plt.xlabel("Voxel size (x, y, z) mm")
plt.ylabel("Patient count")
plt.title("Different voxel sizes: CT vs GT")

plt.xticks(
    x,
    [f"({v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f})"
    for v in unique_voxels],
    rotation=45,
    ha="right"
)
plt.legend()
plt.tight_layout()
plt.savefig("voxel_sizes.png", dpi=300)
plt.close()

#3 HU distribution whole dataset:
CLASS_NAMES = {0: "background", 1: "esophagus", 2: "heart", 3: "trachea"}
values_per_class = {label: [] for label in CLASS_NAMES}

hu_values = []
max_hu = 0
max_patient = 0
max_voxel = 0
for patient in data_dir.iterdir():
    img_ct = nib.load(patient / f"{patient.name}.nii.gz")
    img_gt = nib.load(patient / "GT.nii.gz")
    ct = img_ct.get_fdata()
    gt = img_gt.get_fdata()
    hu_values.append(ct.ravel())
    patient_max = ct.max()

    for label in CLASS_NAMES:
        values_per_class[label].append(ct[gt == label])

    patient_max = ct.max()

    if patient_max > max_hu:
        max_hu = patient_max
        max_patient = patient.name
        max_voxel = np.unravel_index(np.argmax(ct), ct.shape)

all_hu_per_class = {}
for label, values in values_per_class.items():
    if len(values) > 0 and sum(v.size for v in values) > 0:
        all_hu_per_class[label] = np.concatenate(values)

all_hu = np.concatenate(list(all_hu_per_class.values()))
print("Minimum HU:", all_hu.min())
print("Maximum HU:", all_hu.max())
print("Mean HU:", all_hu.mean())
print("Median HU:", np.median(all_hu))
print("Patient with the maximum HU value:", max_patient)

p1 = np.percentile(all_hu, 1)
p99 = np.percentile(all_hu, 99)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    #plot1: alle values
axes[0].boxplot(all_hu, showfliers=True)
axes[0].set_ylim(all_hu.min()-100, all_hu.max())
axes[0].set_title("HU distribution full range")
axes[0].set_ylabel("HU")
axes[0].set_xticks([1])
axes[0].set_xticklabels(["All voxels"])

    #plot2 clipped to p99
axes[1].boxplot(all_hu, showfliers=True)
axes[1].set_ylim(all_hu.min()-100, p99)
axes[1].set_title(f"HU distribution zoomed\n(1st–99th percentile)")
axes[1].set_ylabel("HU")
axes[1].set_xticks([1])
axes[1].set_xticklabels(["All voxels"])

plt.tight_layout()
plt.savefig("HU_boxplot.png", dpi=300)
plt.show()

labels_present = list(all_hu_per_class.keys())
data_per_class = [all_hu_per_class[l] for l in labels_present]
names_present = [CLASS_NAMES[l] for l in labels_present]
positions = range(1, len(labels_present) + 1)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    #plot1: 
axes[0].violinplot(data_per_class, positions=positions, showmedians=True)
axes[0].set_ylim(all_hu.min()-100, all_hu.max())
axes[0].set_title("HU distribution per class")
axes[0].set_ylabel("HU")
axes[0].set_xticks(positions)
axes[0].set_xticklabels(names_present, rotation=45)

    #plot2: zoomed
axes[1].violinplot(data_per_class, positions=positions, showmedians=True)
axes[1].set_ylim(all_hu.min()-100, p99)
axes[1].set_title("HU distribution per class zoomed")
axes[1].set_ylabel("HU")
axes[1].set_xticks(positions)
axes[1].set_xticklabels(names_present, rotation=45)

plt.tight_layout()
plt.savefig("HU_boxplot_per_class.png", dpi=300)
plt.show()

#4 HU values per class
hu_by_class = {
    0: [],
    1: [],
    2: [],
    3: []
}
for patient in data_dir.iterdir():

    img_ct = nib.load(patient / f"{patient.name}.nii.gz")
    img_gt = nib.load(patient / "GT.nii.gz")

    ct = img_ct.get_fdata()
    gt = img_gt.get_fdata()

    for class_id in range(4):
        class_hu = ct[gt == class_id]
        hu_by_class[class_id].append(class_hu)

for class_id in range(4):
    hu_by_class[class_id] = np.concatenate(hu_by_class[class_id])

for class_id in range(4):
    print(
        f"Class {class_id}: "
        f"{len(hu_by_class[class_id]):,} voxels"
    )

#plot figure
plt.figure(figsize=(10, 6))

colors = ["blue", "green", "orange", "red"]
classes = ["background", "esophagus", "heart", "trachea"]

for class_id in range(4):

    plt.hist(
        hu_by_class[class_id],
        bins=200,
        range=(-1000, 500),
        density=True,
        alpha=0.4,
        color=colors[class_id],
        label=classes[class_id]
    )

plt.xlabel("HU values")
plt.ylabel("Density")
plt.title("HU distribution per organ")
plt.legend()

plt.tight_layout()
plt.savefig("HU_classes.png", dpi=300)
plt.show()