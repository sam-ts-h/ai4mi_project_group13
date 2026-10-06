"""
SegTHOR data exploration, makes:
    fingerprint.csv     one row per patient, all header + label statistics
    hu_histograms.png   intensity distribution, overall and per class
    mip_grid.png        maximum intensity projection of every patient
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

CLASS_NAMES = {0: "background", 1: "class1", 2: "heart", 3: "trachea", 4: "class4"}


def find_pairs(root: Path):
    """Yield (patient_id, ct_path, gt_path)"""
    for folder in sorted(root.glob("Patient_*")):
        if not folder.is_dir():
            continue
        ct = folder / f"{folder.name}.nii.gz"
        gt = folder / "GT.nii.gz"
        if ct.exists() and gt.exists():
            yield folder.name, ct, gt
        else:
            print(f"  [warn] missing file in {folder.name}: "
                  f"ct={ct.exists()} gt={gt.exists()}")


# ---------------------------------------------------------------- stage 1 + 2

def fingerprint(root: Path) -> pd.DataFrame:
    rows = []
    for pid, ct_path, gt_path in find_pairs(root):
        ct = nib.load(ct_path)
        gt = nib.load(gt_path)

        # --- stage 1: header only, no voxel data touched yet
        row = {
            "pid": pid,
            "shape": ct.shape,
            "spacing": tuple(np.round(ct.header.get_zooms(), 3)),
            "orient": "".join(nib.aff2axcodes(ct.affine)),
            "dtype_ct": str(ct.header.get_data_dtype()),
            "dtype_gt": str(gt.header.get_data_dtype()),
            "shape_match": ct.shape == gt.shape,
            "affine_match": bool(np.allclose(ct.affine, gt.affine, atol=1e-4)),
        }
        sx, sy, sz = ct.header.get_zooms()
        row["anisotropy"] = round(float(sz / min(sx, sy)), 2)
        row["fov_mm"] = tuple(np.round(np.array(ct.shape) * np.array(ct.header.get_zooms()), 0))
        row["voxel_vol_mm3"] = round(float(sx * sy * sz), 4)

        # --- stage 2: load the arrays
        # get_fdata() casts to float64 (3x memory); fine for the CT, wrong for labels.
        img = ct.get_fdata()
        lab = np.asarray(gt.dataobj)            # keep integer dtype

        row["hu_min"] = float(img.min())
        row["hu_max"] = float(img.max())
        row["hu_p005"] = float(np.percentile(img, 0.5))
        row["hu_p995"] = float(np.percentile(img, 99.5))
        row["hu_mean"] = round(float(img.mean()), 1)

        present = np.unique(lab).astype(int)
        row["label_values"] = tuple(present.tolist())
        row["n_classes"] = len(present) - 1      # excluding background

        for k, name in CLASS_NAMES.items():
            mask = lab == k
            n = int(mask.sum())
            row[f"n_{name}"] = n
            row[f"vol_{name}_ml"] = round(n * row["voxel_vol_mm3"] / 1000, 1)
            if n:
                # centre of mass in voxel coordinates
                row[f"cen_{name}"] = tuple(np.round(np.argwhere(mask).mean(0), 1))
                # sanity: does the intensity under this mask look like the organ?
                row[f"hu_{name}"] = round(float(np.median(img[mask])), 1)
            else:
                row[f"cen_{name}"] = None
                row[f"hu_{name}"] = None

        rows.append(row)
        print(f"  {pid}: shape={row['shape']} spacing={row['spacing']} "
              f"labels={row['label_values']}")

    return pd.DataFrame(rows)


def report_outliers(df: pd.DataFrame) -> None:
    """Print the things worth looking at twice."""
    print("\n" + "=" * 70)
    print("CONSISTENCY CHECKS")
    print("=" * 70)

    for col, label in [("shape_match", "CT/GT shape mismatch"),
                       ("affine_match", "CT/GT affine mismatch")]:
        bad = df.loc[~df[col], "pid"].tolist()
        print(f"{label:<28} {bad if bad else 'none'}")

    print(f"{'distinct orientations':<28} {sorted(df.orient.unique())}")
    print(f"{'distinct spacings':<28} {len(df.spacing.unique())} "
          f"(median z = {np.median([s[2] for s in df.spacing]):.2f} mm)")

    incomplete = df.loc[df.n_classes < len(CLASS_NAMES), ["pid", "label_values"]]
    print(f"{'patients missing a class':<28} "
          f"{incomplete.to_dict('records') if len(incomplete) else 'none'}")

    # volume outliers: robust z-score per class
    print("\nVolume outliers (|robust z| > 3):")
    any_found = False
    for name in CLASS_NAMES.values():
        v = df[f"vol_{name}_ml"].astype(float)
        med, mad = v.median(), (v - v.median()).abs().median()
        if mad == 0:
            continue
        z = 0.6745 * (v - med) / mad
        for pid, zz, vv in zip(df.pid[z.abs() > 3], z[z.abs() > 3], v[z.abs() > 3]):
            print(f"  {name:<10} {pid}: {vv:.1f} ml (z={zz:+.1f}, median={med:.1f})")
            any_found = True
    if not any_found:
        print("  none")

    # intensity sanity: is the tissue under each mask plausible?
    print("\nMedian HU under each mask (expect: air < -800, soft tissue 20-80):")
    print(df[["pid"] + [f"hu_{n}" for n in CLASS_NAMES.values()]].to_string(index=False))


# ------------------------------------------------------------------- stage 3

def histograms(root: Path, out: Path, n_patients: int = 8) -> None:
    """Overall HU distribution plus per-class distribution, pooled over a few patients."""
    pooled_bg, pooled = [], {k: [] for k in CLASS_NAMES}
    for i, (pid, ct_path, gt_path) in enumerate(find_pairs(root)):
        if i >= n_patients:
            break
        img = nib.load(ct_path).get_fdata()
        lab = np.asarray(nib.load(gt_path).dataobj)
        pooled_bg.append(np.random.choice(img.ravel(), 200_000, replace=False))
        for k in CLASS_NAMES:
            vals = img[lab == k]
            if vals.size:
                pooled[k].append(vals)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].hist(np.concatenate(pooled_bg), bins=200, color="0.4")
    axes[0].set_yscale("log")
    axes[0].set_title("Whole volume (log count)")
    axes[0].set_xlabel("HU")

    for k, name in CLASS_NAMES.items():
        if pooled[k]:
            axes[1].hist(np.concatenate(pooled[k]), bins=120, alpha=0.5,
                         label=name, density=True)
    axes[1].set_title("Inside each label")
    axes[1].set_xlabel("HU")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out / "hu_histograms.png", dpi=110)
    print(f"\nwrote {out / 'hu_histograms.png'}")


# ------------------------------------------------------------------- stage 4

def mip_grid(root: Path, out: Path, ncols: int = 8) -> None:
    """One coronal maximum-intensity projection per patient, all on one sheet."""
    pairs = list(find_pairs(root))
    nrows = int(np.ceil(len(pairs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.0 * ncols, 2.6 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (pid, ct_path, gt_path) in zip(axes, pairs):
        img = nib.load(ct_path).get_fdata()
        lab = np.asarray(nib.load(gt_path).dataobj)
        # project along the anterior-posterior axis; adjust if orientation differs
        ax.imshow(np.rot90(img.max(axis=1)), cmap="gray", vmin=-200, vmax=400)
        ax.contour(np.rot90(lab.max(axis=1)), levels=[0.5, 1.5, 2.5, 3.5], linewidths=0.6)
        ax.set_title(pid, fontsize=7)
        ax.axis("off")
    for ax in axes[len(pairs):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out / "mip_grid.png", dpi=110)
    print(f"wrote {out / 'mip_grid.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("reports"))
    ap.add_argument("--skip-plots", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("Scanning...")
    df = fingerprint(args.root)
    df.to_csv(args.out / "fingerprint.csv", index=False)
    print(f"\nwrote {args.out / 'fingerprint.csv'}  ({len(df)} patients)")

    report_outliers(df)

    if not args.skip_plots:
        histograms(args.root, args.out)
        mip_grid(args.root, args.out)


if __name__ == "__main__":
    main()
