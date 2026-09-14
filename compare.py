from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

baseline_dir = Path("data/SEGTHOR_baseline/val")
windowed_dir = Path("data/SEGTHOR_windowed/val")
soft_dir = Path("data/SEGTHOR_soft/val")

# Let op: jouw map heet SEGHTOR, niet SEGTHOR
antialias_dir = Path("data/SEGTHOR_windowed_antialias/val")


# ============================================================
# FIND MATCHING FILES
# ============================================================

baseline_files = {
    p.name for p in (baseline_dir / "img").glob("*.png")
}

windowed_files = {
    p.name for p in (windowed_dir / "img").glob("*.png")
}

soft_files = {
    p.name for p in (soft_dir / "img").glob("*.png")
}

antialias_files = {
    p.name for p in (antialias_dir / "img").glob("*.png")
}


common_files = sorted(
    baseline_files
    & windowed_files
    & soft_files
    & antialias_files
)

print(f"Number of matching slices: {len(common_files)}")


if len(common_files) == 0:
    raise ValueError("No matching slices found between the datasets.")


# ============================================================
# FIND AN INTERESTING SLICE
# Slice with the most ground-truth organ pixels
# ============================================================

best_filename = None
most_gt_pixels = 0

for filename in common_files:

    gt = np.array(
        Image.open(
            baseline_dir / "gt" / filename
        )
    )

    organ_pixels = np.count_nonzero(gt)

    if organ_pixels > most_gt_pixels:
        most_gt_pixels = organ_pixels
        best_filename = filename


print(f"Showing slice: {best_filename}")
print(f"Organ pixels in ground truth: {most_gt_pixels}")


# ============================================================
# LOAD SAME SLICE FROM ALL DATASETS
# ============================================================

baseline = np.array(
    Image.open(
        baseline_dir / "img" / best_filename
    )
)

windowed = np.array(
    Image.open(
        windowed_dir / "img" / best_filename
    )
)

soft = np.array(
    Image.open(
        soft_dir / "img" / best_filename
    )
)

antialias = np.array(
    Image.open(
        antialias_dir / "img" / best_filename
    )
)

gt = np.array(
    Image.open(
        baseline_dir / "gt" / best_filename
    )
)


# ============================================================
# BASIC STATISTICS FOR ONE SLICE
# ============================================================

def print_basic_stats(name, img):

    print(f"\n========== {name} ==========")

    print(f"Min:  {img.min()}")
    print(f"Max:  {img.max()}")
    print(f"Mean: {img.mean():.2f}")
    print(f"Std:  {img.std():.2f}")


print_basic_stats(
    "BASELINE",
    baseline
)

print_basic_stats(
    "WINDOWED [-1000, 1000]",
    windowed
)

print_basic_stats(
    "SOFT [-200, 300]",
    soft
)

print_basic_stats(
    "WINDOWED + ANTIALIAS",
    antialias
)


# ============================================================
# STATISTICS INSIDE GROUND-TRUTH ORGANS
# ============================================================

def print_organ_stats(name, img, gt):

    organ_mask = gt > 0
    organ_pixels = img[organ_mask]

    print(f"\n========== {name} ORGAN STATS ==========")

    print(f"Number of organ pixels: {organ_pixels.size}")
    print(f"Min:  {organ_pixels.min()}")
    print(f"Max:  {organ_pixels.max()}")
    print(f"Mean: {organ_pixels.mean():.2f}")
    print(f"Std:  {organ_pixels.std():.2f}")


print_organ_stats(
    "BASELINE",
    baseline,
    gt
)

print_organ_stats(
    "WINDOWED [-1000, 1000]",
    windowed,
    gt
)

print_organ_stats(
    "SOFT [-200, 300]",
    soft,
    gt
)

print_organ_stats(
    "WINDOWED + ANTIALIAS",
    antialias,
    gt
)


# ============================================================
# SHOW ALL PREPROCESSING METHODS
# ============================================================

plt.figure(figsize=(20, 4))


plt.subplot(1, 5, 1)

plt.imshow(
    baseline,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title("Baseline")
plt.axis("off")


plt.subplot(1, 5, 2)

plt.imshow(
    windowed,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title("HU [-1000, 1000]")
plt.axis("off")


plt.subplot(1, 5, 3)

plt.imshow(
    soft,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title("Soft [-200, 300]")
plt.axis("off")


plt.subplot(1, 5, 4)

plt.imshow(
    antialias,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title("Windowed + Anti-alias")
plt.axis("off")


plt.subplot(1, 5, 5)

plt.imshow(
    antialias,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.imshow(
    gt,
    alpha=0.35
)

plt.title("Anti-alias + GT")
plt.axis("off")


plt.tight_layout()
plt.show()


# ============================================================
# DIRECTLY COMPARE WINDOWED VS WINDOWED + ANTIALIAS
# ============================================================

difference = np.abs(
    windowed.astype(np.int16)
    -
    antialias.astype(np.int16)
)


mean_difference = difference.mean()

max_difference = difference.max()

changed_pixels = (
    np.count_nonzero(difference)
    / difference.size
    * 100
)


print(
    "\n========== ANTI-ALIASING: ONE SLICE =========="
)

print(
    f"Mean absolute difference: "
    f"{mean_difference:.4f}"
)

print(
    f"Maximum pixel difference: "
    f"{max_difference}"
)

print(
    f"Changed pixels: "
    f"{changed_pixels:.2f}%"
)


# ============================================================
# VISUALISE ANTIALIAS DIFFERENCE
# ============================================================

plt.figure(figsize=(12, 4))


plt.subplot(1, 3, 1)

plt.imshow(
    windowed,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title("Windowed")
plt.axis("off")


plt.subplot(1, 3, 2)

plt.imshow(
    antialias,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title("Windowed + Anti-alias")
plt.axis("off")


plt.subplot(1, 3, 3)

plt.imshow(
    difference,
    cmap="gray"
)

plt.title("Difference")
plt.axis("off")


plt.tight_layout()
plt.show()


# ============================================================
# DATASET STATISTICS OVER ALL VALIDATION SLICES
# ============================================================

def calculate_dataset_stats(
    name,
    directory,
    filenames,
    gt_directory
):

    all_organ_pixels = []

    for filename in filenames:

        img = np.array(
            Image.open(
                directory / "img" / filename
            )
        )

        gt = np.array(
            Image.open(
                gt_directory / "gt" / filename
            )
        )

        organ_mask = gt > 0
        organ_pixels = img[organ_mask]

        if organ_pixels.size == 0:
            continue

        all_organ_pixels.append(
            organ_pixels
        )


    if len(all_organ_pixels) == 0:
        print(f"No organ pixels found for {name}")
        return


    all_organ_pixels = np.concatenate(
        all_organ_pixels
    )


    print(
        f"\n========== {name} - ENTIRE DATASET =========="
    )

    print(
        f"Total organ pixels: "
        f"{all_organ_pixels.size}"
    )

    print(
        f"Mean: "
        f"{all_organ_pixels.mean():.2f}"
    )

    print(
        f"Std: "
        f"{all_organ_pixels.std():.2f}"
    )

    print(
        f"Min: "
        f"{all_organ_pixels.min()}"
    )

    print(
        f"Max: "
        f"{all_organ_pixels.max()}"
    )


calculate_dataset_stats(
    "BASELINE",
    baseline_dir,
    common_files,
    baseline_dir
)

calculate_dataset_stats(
    "WINDOWED [-1000, 1000]",
    windowed_dir,
    common_files,
    baseline_dir
)

calculate_dataset_stats(
    "SOFT [-200, 300]",
    soft_dir,
    common_files,
    baseline_dir
)

calculate_dataset_stats(
    "WINDOWED + ANTIALIAS",
    antialias_dir,
    common_files,
    baseline_dir
)


# ============================================================
# ANTIALIAS EFFECT OVER THE ENTIRE VALIDATION DATASET
# ============================================================

def compare_antialias_dataset(
    windowed_dir,
    antialias_dir,
    filenames
):

    total_absolute_difference = 0
    total_pixels = 0
    total_changed_pixels = 0

    maximum_difference = 0


    for filename in filenames:

        original = np.array(
            Image.open(
                windowed_dir / "img" / filename
            )
        ).astype(np.int16)


        antialias = np.array(
            Image.open(
                antialias_dir / "img" / filename
            )
        ).astype(np.int16)


        difference = np.abs(
            original - antialias
        )


        total_absolute_difference += (
            difference.sum()
        )

        total_pixels += (
            difference.size
        )

        total_changed_pixels += (
            np.count_nonzero(
                difference
            )
        )

        maximum_difference = max(
            maximum_difference,
            difference.max()
        )


    mean_absolute_difference = (
        total_absolute_difference
        / total_pixels
    )


    percentage_changed = (
        total_changed_pixels
        / total_pixels
        * 100
    )


    print(
        "\n========== "
        "ANTI-ALIASING EFFECT - ENTIRE DATASET "
        "=========="
    )

    print(
        f"Mean absolute difference: "
        f"{mean_absolute_difference:.4f}"
    )

    print(
        f"Changed pixels: "
        f"{percentage_changed:.2f}%"
    )

    print(
        f"Maximum pixel difference: "
        f"{maximum_difference}"
    )


compare_antialias_dataset(
    windowed_dir,
    antialias_dir,
    common_files
)

# ============================================================
# COMPARE TRAINING RESULTS
# Validation loss + Dice score
# ============================================================

def load_model_results(name, result_dir):
    result_dir = Path(result_dir)

    loss_path = result_dir / "loss_val.npy"
    dice_path = result_dir / "dice_val.npy"

    if not loss_path.exists():
        print(f"\n{name}: loss_val.npy not found in {result_dir}")
        return None

    if not dice_path.exists():
        print(f"\n{name}: dice_val.npy not found in {result_dir}")
        return None

    # Shape:
    # loss_val = [epochs, batches]
    # dice_val = [epochs, images, classes]
    loss_val = np.load(loss_path)
    dice_val = np.load(dice_path)

    # Average validation loss per epoch
    mean_val_loss = loss_val.mean(axis=1)

    # Average Dice per epoch
    # [:, :, 1:] removes background class 0
    mean_val_dice = dice_val[:, :, 1:].mean(axis=(1, 2))

    # Find epoch with highest validation Dice
    best_epoch = np.argmax(mean_val_dice)

    best_dice = mean_val_dice[best_epoch]
    loss_at_best_epoch = mean_val_loss[best_epoch]

    # Dice score per class at best epoch
    class_dice = dice_val[best_epoch].mean(axis=0)

    print(f"\n========== {name} ==========")
    print(f"Best epoch: {best_epoch + 1}")
    print(f"Validation Dice: {best_dice:.4f}")
    print(f"Validation loss: {loss_at_best_epoch:.4f}")

    print("\nDice per class at best epoch:")

    # Skip class 0 = background
    for class_id in range(1, len(class_dice)):
        print(
            f"Class {class_id}: "
            f"{class_dice[class_id]:.4f}"
        )

    return {
        "name": name,
        "loss": mean_val_loss,
        "dice": mean_val_dice,
        "best_epoch": best_epoch,
        "best_dice": best_dice,
        "best_loss": loss_at_best_epoch,
        "class_dice": class_dice,
    }


# ============================================================
# CHANGE THESE PATHS TO YOUR TRAINING OUTPUT FOLDERS
# ============================================================

baseline_results = load_model_results(
    "BASELINE",
    "results/SEGTHOR_baseline"
)

windowed_results = load_model_results(
    "WINDOWED [-1000, 1000]",
    "results/SEGTHOR_windowed"
)

soft_results = load_model_results(
    "SOFT [-200, 300]",
    "results/SEGTHOR_soft"
)

antialias_results = load_model_results(
    "WINDOWED + ANTIALIAS",
    "results/SEGHTOR_windowed_antialias"
)


# ============================================================
# CREATE SUMMARY TABLE IN TERMINAL
# ============================================================

all_results = [
    baseline_results,
    windowed_results,
    soft_results,
    antialias_results,
]

all_results = [
    result for result in all_results
    if result is not None
]

print("\n")
print("=" * 75)
print("MODEL COMPARISON")
print("=" * 75)

print(
    f"{'Model':<30}"
    f"{'Best epoch':<15}"
    f"{'Val loss':<15}"
    f"{'Val Dice':<15}"
)

print("-" * 75)

for result in all_results:
    print(
        f"{result['name']:<30}"
        f"{result['best_epoch'] + 1:<15}"
        f"{result['best_loss']:<15.4f}"
        f"{result['best_dice']:<15.4f}"
    )


# ============================================================
# PLOT VALIDATION DICE
# ============================================================

plt.figure(figsize=(10, 5))

for result in all_results:
    epochs = np.arange(1, len(result["dice"]) + 1)

    plt.plot(
        epochs,
        result["dice"],
        label=result["name"]
    )

plt.xlabel("Epoch")
plt.ylabel("Mean validation Dice")
plt.title("Validation Dice per preprocessing method")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# PLOT VALIDATION LOSS
# ============================================================

plt.figure(figsize=(10, 5))

for result in all_results:
    epochs = np.arange(1, len(result["loss"]) + 1)

    plt.plot(
        epochs,
        result["loss"],
        label=result["name"]
    )

plt.xlabel("Epoch")
plt.ylabel("Validation loss")
plt.title("Validation loss per preprocessing method")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()