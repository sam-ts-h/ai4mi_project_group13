from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image


# ============================================================
# PATHS
# ============================================================

MODELS = {
    "Correct baseline": {
        "data": Path("data/SEGTHOR_correct_baseline/val"),
        "results": Path("results/SEGTHOR_correct_baseline"),
    },
    "Correct HU [-1000, 1000]": {
        "data": Path("data/SEGTHOR_correct_windowed/val"),
        "results": Path("results/SEGTHOR_correct_windowed"),
    },
    "Correct percentile P0.5-P99.5": {
        "data": Path("data/SEGTHOR_correct_percentile/val"),
        "results": Path("results/SEGTHOR_correct_percentile"),
    },
    "Correct HU [-1000, 1000] + anti-aliasing": {
        "data": Path("data/SEGTHOR_correct_windowed_antialias/val"),
        "results": Path("results/SEGTHOR_correct_windowed_antialias"),
    },
}

OUTPUT_DIR = Path("results/comparison_correct_all_4_models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# None = automatically select the validation slice
# with the most GT organ pixels that exists in all 4 datasets.
SELECTED_SLICE = None
SELECTED_PATIENT = "Patient_02"
EXCLUDE_PATIENTS = {"Patient_18"}


SEG_CMAP = ListedColormap([
    (0.0, 0.0, 0.0, 0.0),      # background
    (0.90, 0.20, 0.20, 1.0),   # class 1
    (0.20, 0.75, 0.20, 1.0),   # class 2
    (0.20, 0.45, 0.95, 1.0),   # class 3
    (0.95, 0.80, 0.20, 1.0),   # class 4
])


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path))


def segmentation_to_classes(segmentation: np.ndarray) -> np.ndarray:
    if segmentation.max() > 4:
        segmentation = np.rint(segmentation / 63)
    return segmentation.astype(np.uint8)


def get_prediction_files(result_dir: Path) -> dict[str, Path]:
    prediction_dir = result_dir / "best_epoch" / "val"

    if not prediction_dir.exists():
        raise FileNotFoundError(
            f"Prediction directory not found: {prediction_dir}\n"
            f"Make sure the model has finished training."
        )

    files = {path.name: path for path in prediction_dir.rglob("*.png")}

    if len(files) == 0:
        raise ValueError(f"No prediction PNG files found in {prediction_dir}")

    return files


def plot_overlay(ax, image, mask, title, alpha=0.45):
    ax.imshow(image, cmap="gray", vmin=0, vmax=255)

    masked = np.ma.masked_where(mask == 0, mask)
    ax.imshow(masked, cmap=SEG_CMAP, vmin=0, vmax=4, alpha=alpha)

    for class_id in range(1, 5):
        binary_mask = (mask == class_id).astype(np.uint8)
        if binary_mask.sum() > 0:
            ax.contour(
                binary_mask,
                levels=[0.5],
                linewidths=1.0,
                colors="white"
            )

    ax.set_title(title, fontsize=11)
    ax.axis("off")


def load_metrics(name: str, result_dir: Path) -> dict:
    dice_path = result_dir / "dice_val.npy"
    loss_path = result_dir / "loss_val.npy"

    if not dice_path.exists():
        raise FileNotFoundError(f"Missing file: {dice_path}")
    if not loss_path.exists():
        raise FileNotFoundError(f"Missing file: {loss_path}")

    dice_val = np.load(dice_path)
    loss_val = np.load(loss_path)

    mean_dice = dice_val[:, :, 1:].mean(axis=(1, 2))
    mean_loss = loss_val.mean(axis=1)

    best_epoch = int(np.argmax(mean_dice))
    best_dice = float(mean_dice[best_epoch])
    best_loss = float(mean_loss[best_epoch])

    class_dice = dice_val[best_epoch, :, 1:].mean(axis=0)

    return {
        "name": name,
        "dice": mean_dice,
        "loss": mean_loss,
        "best_epoch": best_epoch,
        "best_dice": best_dice,
        "best_loss": best_loss,
        "class_dice": class_dice,
    }


# ============================================================
# FIND MATCHING FILES
# ============================================================

dataset_files = {}
prediction_files = {}

for model_name, model_info in MODELS.items():
    image_dir = model_info["data"] / "img"
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing dataset directory: {image_dir}")

    dataset_files[model_name] = {path.name for path in image_dir.glob("*.png")}
    prediction_files[model_name] = get_prediction_files(model_info["results"])

common_files = None
for model_name in MODELS:
    current_files = dataset_files[model_name] & set(prediction_files[model_name].keys())
    if common_files is None:
        common_files = current_files
    else:
        common_files &= current_files

common_files = sorted(common_files)

print(f"Matching validation slices across all 4 models: {len(common_files)}")
if len(common_files) == 0:
    raise ValueError("No matching validation slices found.")


# ============================================================
# SELECT ONE GOOD SLICE
# ============================================================

baseline_gt_dir = MODELS["Correct baseline"]["data"] / "gt"

if SELECTED_SLICE is not None:
    if SELECTED_SLICE not in common_files:
        raise ValueError(
            f"{SELECTED_SLICE} is not available for all four models."
        )
    selected_filename = SELECTED_SLICE

else:
    candidate_files = []

    for filename in common_files:
        patient_id = filename.rsplit("_", 1)[0]

        # If a specific patient is selected, only use slices from that patient
        if SELECTED_PATIENT is not None:
            if patient_id != SELECTED_PATIENT:
                continue

        # Otherwise exclude any patients listed here
        elif patient_id in EXCLUDE_PATIENTS:
            continue

        candidate_files.append(filename)

    if len(candidate_files) == 0:
        if SELECTED_PATIENT is not None:
            raise ValueError(
                f"No matching validation slices found for {SELECTED_PATIENT}."
            )
        raise ValueError(
            "No matching validation slices remain after exclusions."
        )

    selected_filename = None
    most_organ_pixels = -1

    for filename in candidate_files:
        gt = load_image(baseline_gt_dir / filename)
        organ_pixels = np.count_nonzero(gt)

        if organ_pixels > most_organ_pixels:
            most_organ_pixels = organ_pixels
            selected_filename = filename

print(f"Selected slice: {selected_filename}")
print(f"Selected patient: {selected_filename.rsplit('_', 1)[0]}")


# ============================================================
# LOAD INPUT IMAGES
# ============================================================

baseline_img = load_image(MODELS["Correct baseline"]["data"] / "img" / selected_filename)
windowed_img = load_image(MODELS["Correct HU [-1000, 1000]"]["data"] / "img" / selected_filename)
percentile_img = load_image(MODELS["Correct percentile P0.5-P99.5"]["data"] / "img" / selected_filename)
windowed_aa_img = load_image(MODELS["Correct HU [-1000, 1000] + anti-aliasing"]["data"] / "img" / selected_filename)

gt = load_image(baseline_gt_dir / selected_filename)
gt_classes = segmentation_to_classes(gt)


# ============================================================
# FIGURE 1
# INPUT IMAGE COMPARISON FOR THE 3 PREPROCESSING METHODS
# ============================================================

fig, axes = plt.subplots(1, 4, figsize=(18, 5))

axes[0].imshow(baseline_img, cmap="gray", vmin=0, vmax=255)
axes[0].set_title("Correct baseline")
axes[0].axis("off")

axes[1].imshow(windowed_img, cmap="gray", vmin=0, vmax=255)
axes[1].set_title("Correct HU [-1000, 1000]")
axes[1].axis("off")

axes[2].imshow(percentile_img, cmap="gray", vmin=0, vmax=255)
axes[2].set_title("Correct percentile P0.5-P99.5")
axes[2].axis("off")

plot_overlay(axes[3], windowed_img, gt_classes, "Ground truth")

fig.suptitle(
    f"Correct-data preprocessing comparison (3 methods)\n{selected_filename}",
    fontsize=14
)
fig.tight_layout()

preprocessing_three_path = OUTPUT_DIR / "preprocessing_comparison_3_methods.png"
fig.savefig(preprocessing_three_path, dpi=200, bbox_inches="tight")


# ============================================================
# FIGURE 2
# WINDOWED VS WINDOWED + ANTI-ALIASING
# ============================================================

difference_img = np.abs(
    windowed_aa_img.astype(np.int16) - windowed_img.astype(np.int16)
).astype(np.uint8)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].imshow(windowed_img, cmap="gray", vmin=0, vmax=255)
axes[0].set_title("Windowed")
axes[0].axis("off")

axes[1].imshow(windowed_aa_img, cmap="gray", vmin=0, vmax=255)
axes[1].set_title("Windowed + anti-aliasing")
axes[1].axis("off")

axes[2].imshow(difference_img, cmap="hot")
axes[2].set_title("|Difference|")
axes[2].axis("off")

plot_overlay(axes[3], windowed_aa_img, gt_classes, "Ground truth")

fig.suptitle(
    f"Windowed vs anti-aliased windowed\n{selected_filename}",
    fontsize=14
)
fig.tight_layout()

window_vs_aa_path = OUTPUT_DIR / "windowed_vs_antialias_difference.png"
fig.savefig(window_vs_aa_path, dpi=200, bbox_inches="tight")


# ============================================================
# LOAD METRICS FOR ALL 4 MODELS
# ============================================================

all_results = []
for model_name, model_info in MODELS.items():
    result = load_metrics(model_name, model_info["results"])
    all_results.append(result)


# ============================================================
# PRINT SUMMARY TABLE
# ============================================================

print()
print("=" * 110)
print("MODEL COMPARISON (ALL 4 CORRECT-DATA MODELS)")
print("=" * 110)
print(f"{'Model':<48}{'Best epoch':<15}{'Val Dice':<15}{'Val loss':<15}")
print("-" * 110)

for result in all_results:
    print(
        f"{result['name']:<48}"
        f"{result['best_epoch'] + 1:<15}"
        f"{result['best_dice']:<15.4f}"
        f"{result['best_loss']:<15.4f}"
    )

print()
print("=" * 110)
print("DICE PER CLASS AT BEST EPOCH")
print("=" * 110)

for result in all_results:
    print(f"\n{result['name']}")
    for class_id, score in enumerate(result["class_dice"], start=1):
        print(f"Class {class_id}: {score:.4f}")


# ============================================================
# SAVE A CSV SUMMARY
# ============================================================

summary_path = OUTPUT_DIR / "metrics_summary.csv"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("model,best_epoch,val_dice,val_loss,class1_dice,class2_dice,class3_dice,class4_dice\n")
    for result in all_results:
        class_scores = ",".join(f"{x:.6f}" for x in result["class_dice"])
        f.write(
            f"{result['name']},{result['best_epoch'] + 1},{result['best_dice']:.6f},"
            f"{result['best_loss']:.6f},{class_scores}\n"
        )


# ============================================================
# FIGURE 3
# VALIDATION DICE FOR ALL 4 MODELS
# ============================================================

plt.figure(figsize=(10, 6))

for result in all_results:
    epochs = np.arange(1, len(result["dice"]) + 1)
    plt.plot(epochs, result["dice"], label=result["name"], linewidth=2)

plt.xlabel("Epoch")
plt.ylabel("Mean validation Dice")
plt.title("Validation Dice comparison (all 4 corrected-data models)")
plt.ylim(0, 1)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

dice_plot_path = OUTPUT_DIR / "validation_dice_all_4_models.png"
plt.savefig(dice_plot_path, dpi=200, bbox_inches="tight")


# ============================================================
# FIGURE 4
# VALIDATION LOSS FOR ALL 4 MODELS
# ============================================================

plt.figure(figsize=(10, 6))

for result in all_results:
    epochs = np.arange(1, len(result["loss"]) + 1)
    plt.plot(epochs, result["loss"], label=result["name"], linewidth=2)

plt.xlabel("Epoch")
plt.ylabel("Mean validation loss")
plt.title("Validation loss comparison (all 4 corrected-data models)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

loss_plot_path = OUTPUT_DIR / "validation_loss_all_4_models.png"
plt.savefig(loss_plot_path, dpi=200, bbox_inches="tight")


# ============================================================
# OPTIONAL: PREDICTION COMPARISON FOR ALL 4 MODELS
# ============================================================

baseline_pred = segmentation_to_classes(
    load_image(prediction_files["Correct baseline"][selected_filename])
)
windowed_pred = segmentation_to_classes(
    load_image(prediction_files["Correct HU [-1000, 1000]"][selected_filename])
)
percentile_pred = segmentation_to_classes(
    load_image(prediction_files["Correct percentile P0.5-P99.5"][selected_filename])
)
windowed_aa_pred = segmentation_to_classes(
    load_image(prediction_files["Correct HU [-1000, 1000] + anti-aliasing"][selected_filename])
)

fig, axes = plt.subplots(1, 5, figsize=(24, 5))

plot_overlay(axes[0], windowed_img, gt_classes, "Ground truth")
plot_overlay(axes[1], baseline_img, baseline_pred, "Baseline prediction")
plot_overlay(axes[2], windowed_img, windowed_pred, "Windowed prediction")
plot_overlay(axes[3], percentile_img, percentile_pred, "Percentile prediction")
plot_overlay(axes[4], windowed_aa_img, windowed_aa_pred, "Windowed + AA prediction")

fig.suptitle(
    f"Best epoch predictions for all 4 models\n{selected_filename}",
    fontsize=14
)
fig.tight_layout()

prediction_path = OUTPUT_DIR / "prediction_comparison_all_4_models.png"
fig.savefig(prediction_path, dpi=200, bbox_inches="tight")


print()
print("Saved comparison figures to:")
print(OUTPUT_DIR)
print(f"- {preprocessing_three_path.name}")
print(f"- {window_vs_aa_path.name}")
print(f"- {dice_plot_path.name}")
print(f"- {loss_plot_path.name}")
print(f"- {prediction_path.name}")
print(f"- {summary_path.name}")

plt.show()