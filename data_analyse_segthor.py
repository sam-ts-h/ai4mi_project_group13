#Imports

from pathlib import Path
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#Patients
source_dir = Path("data/segthor_part1/train")

processed_train_dir = Path("data/SEGTHOR/train/img")

train_patient_ids = sorted({
    path.stem.rsplit("_", 1)[0]
    for path in processed_train_dir.glob("*.png")
})

patient_dirs = [
    source_dir / patient_id
    for patient_id in train_patient_ids
]

for patient_dir in patient_dirs:
    print(f"Patient {patient_dir.name} has {len(list(patient_dir.iterdir()))} files")

print(f"Total patients: {len(patient_dirs)}")


#classes
all_classes = set()

for patient_dir in patient_dirs:
    gt_path = patient_dir / "GT.nii.gz"

    gt_image = nib.load(gt_path)
    gt = np.asarray(gt_image.dataobj)

    patient_classes = np.unique(gt)
    all_classes.update(patient_classes)

    print(patient_dir.name, "heeft classes:", patient_classes)


print("\nAlle classes in de dataset:", sorted(all_classes))

class_names = {
    0: "background",
    1: "esophagus",
    2: "heart",
    3: "trachea",
    4: "aorta",
}

#HU values for each class

hu_values_per_class = {}
mean_hu_per_patient = {}

for class_id in all_classes:
    class_id = int(class_id)

    if class_id != 0:
        hu_values_per_class[class_id] = []
        mean_hu_per_patient[class_id] = {}


for patient_dir in patient_dirs:
    ct_path = patient_dir / f"{patient_dir.name}.nii.gz"
    gt_path = patient_dir / "GT.nii.gz"

    ct_image = nib.load(ct_path)
    gt_image = nib.load(gt_path)

    ct = ct_image.get_fdata()
    gt = np.asarray(gt_image.dataobj)

    for class_id in hu_values_per_class:
        hu_values = ct[gt == class_id]
        hu_values_per_class[class_id].append(hu_values)
        mean_hu = np.mean(hu_values)
        mean_hu_per_patient[class_id][patient_dir.name] = mean_hu

print("\nHU-waarden per class over alle patiënten:")



for class_id in sorted(hu_values_per_class):
    hu_values = np.concatenate(hu_values_per_class[class_id])

    print("\nClass:", class_id, "-", class_names[class_id])
    print("Aantal voxels:", len(hu_values))
    print("Minimum HU:", np.min(hu_values))
    print("Maximum HU:", np.max(hu_values))
    print("1e percentiel:", np.percentile(hu_values, 1))
    print("5e percentiel:", np.percentile(hu_values, 5))
    print("95e percentiel:", np.percentile(hu_values, 95))
    print("99e percentiel:", np.percentile(hu_values, 99))
    print("Gemiddelde HU:", np.mean(hu_values))
    print("Mediaan HU:", np.median(hu_values))
    print("Standaardafwijking:", np.std(hu_values))


# Gemiddelde HU per patiënt vergelijken
print("\nGemiddelde HU per patiënt en per class:")

for class_id in sorted(mean_hu_per_patient):
    print("\nClass:", class_id, "-", class_names[class_id])

    patient_means = []

    for patient_name, mean_hu in mean_hu_per_patient[class_id].items():
        print(patient_name, ":", round(mean_hu, 2), "HU")
        patient_means.append(mean_hu)

    print("Laagste gemiddelde:", round(np.min(patient_means), 2), "HU")
    print("Hoogste gemiddelde:", round(np.max(patient_means), 2), "HU")
    print(
        "Verschil:",
        round(np.max(patient_means) - np.min(patient_means), 2),
        "HU",
    )
    print(
        "Standaardafwijking tussen patiënten:",
        round(np.std(patient_means), 2),
        "HU",
    )
    

print("\nPatiënten met meer dan twee bestanden:")

for patient_dir in patient_dirs:
    files = sorted(patient_dir.iterdir())

    if len(files) > 2:
        print(patient_dir.name)
        
        for file in files:
            print(" -", file.name)


# Extra bestanden controleren

print("\nExtra GT van Patient_07 controleren:")

gt_1_image = nib.load(source_dir / "Patient_07" / "GT.nii.gz")
gt_2_image = nib.load(source_dir / "Patient_07" / "GT2.nii.gz")

gt_1 = np.asarray(gt_1_image.dataobj)
gt_2 = np.asarray(gt_2_image.dataobj)

print("Shape eerste GT:", gt_1.shape)
print("Shape tweede GT:", gt_2.shape)
print("Dezelfde shape:", gt_1.shape == gt_2.shape)
print("Dezelfde spacing:", np.allclose(
    gt_1_image.header.get_zooms()[:3],
    gt_2_image.header.get_zooms()[:3],
))
print("Dezelfde affine:", np.allclose(
    gt_1_image.affine,
    gt_2_image.affine,
))
print("Classes eerste GT:", np.unique(gt_1))
print("Classes tweede GT:", np.unique(gt_2))

if gt_1.shape == gt_2.shape:
    print("Exact dezelfde labels:", np.array_equal(gt_1, gt_2))
    print(
        "Percentage verschillende voxels:",
        np.mean(gt_1 != gt_2) * 100,
        "%",
    )

#violin boxplot
rng = np.random.default_rng(42)

plot_rows = []

for class_id in sorted(hu_values_per_class):
    hu_values = np.concatenate(hu_values_per_class[class_id])

    sample_size = min(50000, len(hu_values))
    hu_sample = rng.choice(
        hu_values,
        size=sample_size,
        replace=False,
    )

    for value in hu_sample:
        plot_rows.append({
            "Class ID": class_id,
            "Organ": class_names[class_id],
            "HU": value
        })

plot_df = pd.DataFrame(plot_rows)


plt.figure(figsize=(10, 6))

sns.violinplot(
    data=plot_df,
    x="Organ",
    y="HU",
    inner=None,
    cut=0
)

sns.boxplot(
    data=plot_df,
    x="Organ",
    y="HU",
    width=0.2,
    showcaps=True,
    boxprops={"facecolor": "white", "zorder": 3},
    whiskerprops={"zorder": 3},
    flierprops={
        "marker": ".",
        "markersize": 2,
        "alpha": 0.3,
    }
)

plt.title("HU distribution per organ")
plt.ylabel("HU value")
plt.xlabel("Organ")
plt.grid(axis="y", alpha=0.3)

output_dir = Path("results/segthor_analysis")
output_dir.mkdir(parents=True, exist_ok=True)

plt.tight_layout()
plt.savefig(
    output_dir / "hu_violin_boxplot_per_organ.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()




# Boxplot van HU-waarden per class

rng = np.random.default_rng(42)

class_ids = sorted(hu_values_per_class)

fig, axes = plt.subplots(
    1,
    len(class_ids),
    figsize=(12, 6),
)

for ax, class_id in zip(axes, class_ids):
    hu_values = np.concatenate(hu_values_per_class[class_id])

    sample_size = min(50000, len(hu_values))
    hu_sample = rng.choice(
        hu_values,
        size=sample_size,
        replace=False,
    )

    ax.boxplot(
    hu_sample,
    showfliers=True,
    flierprops={
        "marker": ".",
        "markersize": 2,
        "alpha": 0.3,
    },
)

    ax.set_xticks([1])
    ax.set_xticklabels([class_names[class_id]])
    ax.set_title(f"Class {class_id}")
    ax.set_ylabel("HU-waarde")
    ax.grid(axis="y", alpha=0.3)

plt.suptitle("HU distribution per organ")
plt.tight_layout()

output_dir = Path("results/segthor_analysis")
output_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(
    output_dir / "hu_boxplot_per_organ.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()