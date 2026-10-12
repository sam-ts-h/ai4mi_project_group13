import argparse
import csv
from pathlib import Path

import numpy as np
import nibabel as nib


nrClasses = 5
classNames = [
    "background",
    "esophagus",
    "heart",
    "trachea",
    "aorta",
]


def dice_2d(gt, pred):
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    denominator = gt.sum() + pred.sum()

    if denominator == 0:
        return np.nan

    intersection = np.logical_and(gt, pred).sum()

    return 2.0 * intersection / denominator


def main(args):

    args.dest_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    predicted_files = sorted(
        args.pred_folder.glob("*.nii.gz")
    )

    assert len(predicted_files) > 0, (
        "No predictions found."
    )

    rows = []

    for pred_path in predicted_files:

        patient_id = pred_path.name.replace(
            ".nii.gz",
            "",
        )

        gt_path = (
            args.gt_folder
            / patient_id
            / "GT.nii.gz"
        )

        gt = np.asarray(
            nib.load(str(gt_path)).dataobj
        )

        pred = np.asarray(
            nib.load(str(pred_path)).dataobj
        )

        assert pred.shape == gt.shape, (
            f"Shape mismatch for {patient_id}: "
            f"{pred.shape} vs {gt.shape}"
        )

        # Axial slices = z-axis
        nr_slices = gt.shape[2]

        for k in range(1, nrClasses):

            class_name = classNames[k]

            slice_scores = []

            for z in range(nr_slices):

                gt_slice = gt[:, :, z] == k
                pred_slice = pred[:, :, z] == k

                # Alleen slices waarop het orgaan
                # daadwerkelijk in de GT aanwezig is
                if not gt_slice.any():
                    continue

                score = dice_2d(
                    gt_slice,
                    pred_slice,
                )

                if not np.isnan(score):
                    slice_scores.append(score)

                    rows.append({
                        "patient": patient_id,
                        "class": class_name,
                        "slice": z,
                        "dsc": score,
                    })

            if len(slice_scores) > 0:

                print(
                    f"{patient_id:<12} "
                    f"{class_name:<12} "
                    f"n={len(slice_scores):<4} "
                    f"mean Dice={np.mean(slice_scores):.4f}"
                )


    # ========================================================
    # Save every individual slice score
    # ========================================================

    per_slice_path = (
        args.dest_folder
        / "perSlice.csv"
    )

    with open(
        per_slice_path,
        "w",
        newline="",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "patient",
                "class",
                "slice",
                "dsc",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)


    # ========================================================
    # Summary
    # ========================================================

    print()
    print("=" * 65)
    print("2D DICE SUMMARY")
    print("=" * 65)

    summary_rows = []

    for class_name in classNames[1:]:

        values = np.array([
            row["dsc"]
            for row in rows
            if row["class"] == class_name
        ])

        if len(values) == 0:
            continue

        mean = values.mean()
        sd = values.std(ddof=1)

        summary_rows.append({
            "class": class_name,
            "mean_dsc": mean,
            "sd_dsc": sd,
            "n_slices": len(values),
        })

        print(
            f"{class_name:<12} "
            f"mean={mean:.4f}  "
            f"SD={sd:.4f}  "
            f"n={len(values)}"
        )


    # Combined foreground Dice
    all_values = np.array([
        row["dsc"]
        for row in rows
    ])

    combined_mean = all_values.mean()
    combined_sd = all_values.std(ddof=1)

    summary_rows.append({
        "class": "combined",
        "mean_dsc": combined_mean,
        "sd_dsc": combined_sd,
        "n_slices": len(all_values),
    })

    print()
    print(
        f"{'combined':<12} "
        f"mean={combined_mean:.4f}  "
        f"SD={combined_sd:.4f}  "
        f"n={len(all_values)}"
    )


    summary_path = (
        args.dest_folder
        / "summary.csv"
    )

    with open(
        summary_path,
        "w",
        newline="",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "class",
                "mean_dsc",
                "sd_dsc",
                "n_slices",
            ],
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    print()
    print(f"Saved: {per_slice_path}")
    print(f"Saved: {summary_path}")


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pred_folder",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--gt_folder",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--dest_folder",
        type=Path,
        required=True,
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())