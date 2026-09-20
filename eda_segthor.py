#!/usr/bin/env python3
"""
Exploratory Data Analysis for the SegTHOR dataset (ai4mi_project).

Run from the project root, with the venv active:
    $ python eda_segthor.py --source_dir data/segthor_part1 --dest_dir eda_output

Reads the RAW 3D NIfTI files directly (not the sliced 256x256 PNGs), so all
geometry/intensity numbers reflect the true original data.

Produces, in --dest_dir:
    - fingerprint.csv          one row per patient: spacing, shape, HU range, FOV stats
    - class_voxel_counts.csv   total voxel count per class, across the dataset
    - hu_by_class_hist.png     HU intensity histogram, split by class
    - spacing_hist.png         dx / dz distributions across patients
    - z_extent_hist.png        number of z-slices per patient
    - z_coverage_<patient>.png per-patient: which z-slices contain which class
    - fov_crop_hist.png        fraction of each slice's area that is body tissue

Class labels (SegTHOR convention):
    0 = background, 1 = esophagus, 2 = heart, 3 = trachea, 4 = aorta
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy import ndimage
from tqdm import tqdm

CLASS_NAMES = ["background", "esophagus", "heart", "trachea", "aorta"]
N_CLASSES = len(CLASS_NAMES)

# A generous soft-tissue-ish window for the FOV/body-mask heuristic.
# (Purely for finding "is this pixel body or empty air/table/metal", not for
# training.) Lower bound excludes air (table/background); upper bound
# excludes metal (surgical hardware, positioning clamps) -- real human tissue,
# even dense cortical bone, essentially never exceeds ~1500-2000 HU.
BODY_HU_THRESHOLD = -300
BODY_HU_UPPER = 1500


def get_patient_ids(train_dir: Path) -> list[str]:
    return sorted(p.name for p in train_dir.glob("Patient_*") if p.is_dir())


def load_patient(train_dir: Path, pid: str) -> tuple[np.ndarray, np.ndarray, tuple, nib.Nifti1Image]:
    #Returns Nifti1Image object that is a data wrapper that holds 1. reference to file on disk, 2. 4x4 affine matrix mapping 
    # pixel coordinates to real-world millimeter coordinates (encodes orientation, spacing and origin)
    # and 3. header (metadata: spacing, data type, dimensions etc)
    ct_nib = nib.load(str(train_dir / pid / f"{pid}.nii.gz"))
    gt_nib = nib.load(str(train_dir / pid / "GT.nii.gz"))

    # nib_obj.dataobj is a special "array proxy", np.asarray then loads array into memory
    ct = np.asarray(ct_nib.dataobj)  #3D np array of shape (X, Y, Z)
    gt = np.asarray(gt_nib.dataobj)

    #.header.get_zooms() reads the "zoom" fields directly from the header metadata
    # These are the voxel spacings, i.e. how many millimeters each voxel represents along each axis.
    spacing = ct_nib.header.get_zooms() 

    return ct, gt, spacing, ct_nib


# ---------------------------------------------------------------------------
# 1. Spacing / geometry fingerprint
# ---------------------------------------------------------------------------

def analyze_geometry(fingerprint_rows: list[dict], dest_dir: Path) -> None:
    dx = [r["dx"] for r in fingerprint_rows]
    dz = [r["dz"] for r in fingerprint_rows]
    z_extent = [r["z_extent"] for r in fingerprint_rows]

    print("\n=== Geometry fingerprint (how big each voxel step is) ===")
    print(f"dx/dy (in-plane spacing, mm): min={min(dx):.3f} max={max(dx):.3f} "
          f"mean={np.mean(dx):.3f} std={np.std(dx):.3f} median = {np.median(dx):.3f}")
    print(f"dz (slice thickness, mm):     min={min(dz):.3f} max={max(dz):.3f} "
          f"mean={np.mean(dz):.3f} std={np.std(dz):.3f} median = {np.median(dz):.3f}")
    print(f"z-extent (num slices):        min={min(z_extent)} max={max(z_extent)} "
          f"mean={np.mean(z_extent):.1f} median = {np.median(z_extent):.3f}")
    print(f"Anisotropy ratio dz/dx:       mean={np.mean(np.array(dz) / np.array(dx)):.2f}x "
          "(how much coarser z-resolution is vs. in-plane)")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(dx, bins=15, alpha=0.7, label="dx/dy")
    axes[0].hist(dz, bins=15, alpha=0.7, label="dz")
    axes[0].set_xlabel("mm")
    axes[0].set_title("Voxel spacing distribution")
    axes[0].legend()

    axes[1].hist(z_extent, bins=15, color="tab:green")
    axes[1].set_xlabel("number of z-slices")
    axes[1].set_title("Patient z-extent distribution")

    fig.tight_layout()
    fig.savefig(dest_dir / "spacing_hist.png")
    plt.close(fig)

# ---------------------------------------------------------------------------
# 2. Intensity (HU) distribution per class
# ---------------------------------------------------------------------------
 
def analyze_intensity(train_dir: Path, patient_ids: list[str], dest_dir: Path,
                       max_patients: int | None = None) -> None:
    print("\n=== Intensity (HU) distribution per class ===")
    # Subsample voxels per class per patient to keep memory bounded.
    samples_per_class = {k: [] for k in range(N_CLASSES)}
    n_sample_per_patient_per_class = 20_000
 
    ids = patient_ids[:max_patients] if max_patients else patient_ids
    #tqdm wraps iterable and prints live progress bar as interations progress
    for pid in tqdm(ids, desc="HU sampling"):
        ct, gt, _, _ = load_patient(train_dir, pid)
        for k in range(N_CLASSES):
            voxels = ct[gt == k] #1D array of all voxel's intensities belonging to class k
            if voxels.size == 0:
                continue
            #can have millions of background voxels and keeping all those across all patients would blow up memory so if more than 20k
            # pick 20k random voxels as a representative random sample
            if voxels.size > n_sample_per_patient_per_class:
                idx = np.random.choice(voxels.size, n_sample_per_patient_per_class, replace=False)
                voxels = voxels[idx]
            samples_per_class[k].append(voxels) #append to running list of voxels per class across patients for histogram / stats per class
 
    # One subplot per class, each with its own y-scale, sharing a common x-range. Using `range=` on ax.hist (not np.clip on the data) means
    # values outside the window are simply excluded from that class's bins, instead of all piling up into one artificial edge spike.
    present_classes = [k for k in range(N_CLASSES) if samples_per_class[k]]
    fig, axes = plt.subplots(1, len(present_classes), figsize=(4 * len(present_classes), 4),
                              sharex=True)
    if len(present_classes) == 1:
        axes = [axes]
 
    for ax, k in zip(axes, present_classes):
        all_vals = np.concatenate(samples_per_class[k])
        p05, p995 = np.percentile(all_vals, [0.5, 99.5])
        print(f"  {CLASS_NAMES[k]:12s}: n={all_vals.size:>9d}  mean={all_vals.mean():8.1f} median={np.median(all_vals):8.1f}  "
              f"std={all_vals.std():7.1f}  min={np.min(all_vals):8.1f} max={np.max(all_vals):8.1f} [0.5pct={p05:8.1f}, 99.5pct={p995:8.1f}]")
        ax.hist(all_vals, bins=100, range=(-1000, 500), color=f"C{k}")
        ax.set_title(CLASS_NAMES[k])
        ax.set_xlabel("HU")
        ax.axvline(all_vals.mean(), color="black", linestyle="--", linewidth=1)
 
    axes[0].set_ylabel("voxel count")
    fig.suptitle("HU distribution by class (own y-scale per class; dashed line = mean)")
    fig.tight_layout()
    fig.savefig(dest_dir / "hu_by_class_hist.png")
    plt.close(fig)
 
    # Foreground-only global stats (all non-background classes combined),
    # mirroring nnU-Net's "foreground intensity" fingerprint statistic.
    fg_vals = np.concatenate([v for k, vs in samples_per_class.items() if k != 0 for v in vs])
    fg_p05, fg_p995 = np.percentile(fg_vals, [0.5, 99.5])
    print(f"\n  Foreground (all organs) 0.5/99.5 percentile HU clip range: "
          f"[{fg_p05:.1f}, {fg_p995:.1f}]")
    print("  (candidate clip range for intensity normalization, a la nnU-Net's CT scheme)")
 
 
# ---------------------------------------------------------------------------
# 3. Class imbalance
# ---------------------------------------------------------------------------

def analyze_class_balance(train_dir: Path, patient_ids: list[str], dest_dir: Path) -> None:
    print("\n=== Class imbalance ===")
    total_counts = np.zeros(N_CLASSES, dtype=np.int64)
    per_patient_counts = []

    for pid in tqdm(patient_ids, desc="Counting class voxels"):
        _, gt, _, _ = load_patient(train_dir, pid)
        counts = np.array([(gt == k).sum() for k in range(N_CLASSES)])
        total_counts += counts
        per_patient_counts.append((pid, *counts)) #unpacks counts list so that it becomes a 6D tuple.

    total = total_counts.sum()
    print(f"{'class':12s} {'voxel count':>15s} {'% of all voxels':>18s} {'% of foreground':>18s}")
    fg_total = total_counts[1:].sum()
    for k in range(N_CLASSES):
        pct_all = 100 * total_counts[k] / total
        pct_fg = 100 * total_counts[k] / fg_total if k != 0 else float("nan")
        print(f"{CLASS_NAMES[k]:12s} {total_counts[k]:15d} {pct_all:17.4f}% "
              f"{'' if k == 0 else f'{pct_fg:17.2f}%'}")

    with open(dest_dir / "class_voxel_counts.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id"] + CLASS_NAMES) #writes a single row given columns values: adds column names first
        writer.writerows(per_patient_counts) #writes each patient count as a seperate row

    print(f"\nSaved per-patient class counts to {dest_dir / 'class_voxel_counts.csv'}")
    print("(Use this to gauge whether a weighted loss / oversampling of rare classes,")
    print(" e.g. esophagus, is warranted.)")


# ---------------------------------------------------------------------------
# 4. Anatomical z-coverage per organ
# ---------------------------------------------------------------------------

def analyze_z_coverage(train_dir: Path, patient_ids: list[str], dest_dir: Path) -> None:
    print("\n=== Anatomical z-coverage per organ ===")
    for pid in patient_ids:
        _, gt, _, _ = load_patient(train_dir, pid)
        z = gt.shape[2]
        presence = np.zeros((N_CLASSES, z), dtype=bool)
        for idz in range(z):
            classes_here = np.unique(gt[:, :, idz])
            for k in classes_here:
                presence[k, idz] = True

        fig, ax = plt.subplots(figsize=(10, 2.5))
        for k in range(1, N_CLASSES):  # skip background row, it's everywhere
            ys = np.where(presence[k])[0] #presence[k] is a 1D array of length z, np.where returns tuple with array of indices per dim.
            # [0] extracts array from tuple.
            ax.scatter(ys, [k] * len(ys), s=4, label=CLASS_NAMES[k])
        ax.set_yticks(range(1, N_CLASSES))
        ax.set_yticklabels(CLASS_NAMES[1:])
        ax.set_xlabel("z-slice index")
        ax.set_title(f"{pid}: which slices contain which organ")
        fig.tight_layout()
        fig.savefig(dest_dir / f"z_coverage_{pid}.png")
        plt.close(fig)

        # For each slice, is at least one organ present (collapse along class dimension after dropping background (first row))
        # .any gives boolean array along z dimension which is then averaged. 
        empty_frac = 1 - presence[1:].any(axis=0).mean() 
        print(f"  {pid}: {empty_frac*100:.1f}% of slices contain NO foreground organ at all")

    print(f"\nSaved per-patient z-coverage plots for all patients.")
    print("(Slices with no foreground organ are candidates for downweighting or exclusion")
    print(" during batch sampling, to avoid the network being dominated by trivial background.)")


# ---------------------------------------------------------------------------
# 5. Field-of-view / cropping analysis
# ---------------------------------------------------------------------------
 
def analyze_fov(fingerprint_rows: list[dict], dest_dir: Path) -> None:
    print("\n=== Field-of-view / cropping potential ===")
    fracs = [r["body_area_frac"] for r in fingerprint_rows]
    print(f"Fraction of each slice occupied by body tissue: "
          f"mean={np.mean(fracs)*100:.1f}%  min={min(fracs)*100:.1f}%  max={max(fracs)*100:.1f}%")
    print("(A low fraction means a lot of the resize budget is spent on empty")
    print(" scanner bed / air margin -- cropping to the body bounding box before resizing")
    print(" would give the network more effective resolution on actual anatomy.)")
 
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(np.array(fracs) * 100, bins=20, color="tab:purple")
    ax.set_xlabel("% of slice area that is body tissue")
    ax.set_title("Field-of-view utilization across patients")
    fig.tight_layout()
    fig.savefig(dest_dir / "fov_crop_hist.png")
    plt.close(fig)
 
 
def body_mask_2d(ct_slice: np.ndarray) -> np.ndarray:
    """
    Pure 2D, per-slice body-mask extraction: threshold (both bounds), keep
    only the single largest connected component in THIS slice, fill enclosed
    holes. Run independently per slice -- no 3D connectivity at all -- since
    that's what actually excludes equipment/artifacts/arms that only look
    separate from the body within a given slice, without risking a 3D fusion
    elsewhere in the volume dragging them back in.
    """
    raw = (ct_slice > BODY_HU_THRESHOLD) & (ct_slice < BODY_HU_UPPER)
    if not raw.any():
        return raw  # all False; caller handles empty slices
    labeled, num_features = ndimage.label(raw, structure=np.ones((3, 3), dtype=int))
    sizes = ndimage.sum(raw, labeled, index=range(1, num_features + 1))
    largest_label = int(np.argmax(sizes)) + 1
    mask = labeled == largest_label
    return ndimage.binary_fill_holes(mask)
 
 
def analyze_body_bbox_extent(train_dir: Path, patient_ids: list[str], dest_dir: Path,
                              target_spacing_mm: float = 1.0) -> None:
    """
    For each patient, for each slice INDEPENDENTLY: threshold the CT (both a
    lower bound to exclude air/table and an upper bound to exclude metal),
    take the single largest 2D connected component (this is the patient's
    body in that slice -- doing this per-slice, not in 3D, avoids a stray
    touching point elsewhere in the volume fusing equipment/artifacts into
    the same label), fill enclosed holes (tracheal lumen, etc). Also measures
    the labeled-organ bounding box for reference (useful for sizing a safety
    margin). Neither uses GT to determine crop POSITION -- only to validate
    the body-mask method and to report a reference size -- so this stays a
    valid, GT-independent procedure at train/val/test time.
 
    Validation: for every slice that actually contains a labeled organ,
    checks that every organ pixel falls INSIDE that same slice's chosen body
    component. (Checking only the most extreme-width slices would be
    misleading, since the widest points of the body -- shoulders, hips -- are
    typically well outside the thorax where these organs live and would
    rarely contain any organ pixels regardless of whether the method worked
    correctly.)
    """
    print("\n=== Body bounding-box extent via per-slice 2D connected components ===")
    body_w_mm, body_h_mm = [], []
    organ_w_mm, organ_h_mm = [], []
    validation_failures = []  # (patient_id, z_index, num_organ_voxels_outside_body)
 
    # Track the single slice (patient + z-index) responsible for each global
    # max, so it can be reported and visually spot-checked afterward.
    src_body_w = {"mm": -1.0, "pid": None, "idz": None}
    src_body_h = {"mm": -1.0, "pid": None, "idz": None}
    src_organ_w = {"mm": -1.0, "pid": None, "idz": None}
    src_organ_h = {"mm": -1.0, "pid": None, "idz": None}
 
    def track(src: dict, value_mm: float, pid: str, idz: int) -> None:
        if value_mm > src["mm"]:
            src.update(mm=value_mm, pid=pid, idz=idz)
 
    for pid in tqdm(patient_ids, desc="Body CC (2D) + bbox + validation"):
        ct, gt, spacing, _ = load_patient(train_dir, pid)
        dx, dy, _ = spacing  # dx == dy per the earlier sanity_ct check
 
        organ_mask = gt > 0
        max_bw = max_bh = max_ow = max_oh = 0
        for idz in range(ct.shape[2]):
            b = body_mask_2d(ct[:, :, idz])
            if b.any():
                ys, xs = np.where(b)
                h_vox = ys.max() - ys.min() + 1
                w_vox = xs.max() - xs.min() + 1
                max_bh = max(max_bh, h_vox)
                max_bw = max(max_bw, w_vox)
                track(src_body_h, h_vox * dx, pid, idz)
                track(src_body_w, w_vox * dx, pid, idz)
 
            o = organ_mask[:, :, idz]
            if o.any():
                ys, xs = np.where(o)
                h_vox = ys.max() - ys.min() + 1
                w_vox = xs.max() - xs.min() + 1
                max_oh = max(max_oh, h_vox)
                max_ow = max(max_ow, w_vox)
                track(src_organ_h, h_vox * dx, pid, idz)
                track(src_organ_w, w_vox * dx, pid, idz)
                # Validation: every organ pixel in this slice must be inside
                # this same slice's chosen body component.
                outside = o & (~b)
                if outside.any():
                    validation_failures.append((pid, idz, int(outside.sum())))
 
        body_w_mm.append(max_bw * dx); body_h_mm.append(max_bh * dx)
        organ_w_mm.append(max_ow * dx); organ_h_mm.append(max_oh * dx)
 
    def report(label, widths_mm, heights_mm, src_w, src_h):
        w, h = np.array(widths_mm), np.array(heights_mm)
        print(f"\n  {label}:")
        print(f"    width  (mm): max={w.max():.1f}  95th-pct={np.percentile(w, 95):.1f}  "
              f"mean={w.mean():.1f}  [from {src_w['pid']}, slice {src_w['idz']}]")
        print(f"    height (mm): max={h.max():.1f}  95th-pct={np.percentile(h, 95):.1f}  "
              f"mean={h.mean():.1f}  [from {src_h['pid']}, slice {src_h['idz']}]")
        side_px_at_target = int(np.ceil(max(w.max(), h.max()) / target_spacing_mm))
        print(f"    -> at {target_spacing_mm}mm target spacing, a square window needs "
              f"~{side_px_at_target}px per side to guarantee no clipping (0 margin)")
 
    report("Body (largest 2D connected component per slice, table/metal excluded)",
           body_w_mm, body_h_mm, src_body_w, src_body_h)
    report("Labeled organs only (reference, for margin sizing)", organ_w_mm, organ_h_mm,
           src_organ_w, src_organ_h)
 
    print("\n=== Validation: are all organ pixels inside the chosen body component? ===")
    if not validation_failures:
        print("  PASSED: every organ-containing slice, across all patients, had 100% of its")
        print("  organ pixels inside the selected body component. The largest-connected-")
        print("  component method appears to reliably isolate the body from the table/equipment.")
    else:
        total_failed_slices = len(validation_failures)
        total_failed_voxels = sum(f[2] for f in validation_failures)
        print(f"  FAILED on {total_failed_slices} slice(s) across "
              f"{len(set(f[0] for f in validation_failures))} patient(s), "
              f"{total_failed_voxels} organ voxels total fell outside the chosen body component:")
        for pid, idz, n in validation_failures[:15]:
            print(f"    - {pid}, slice {idz}: {n} organ voxels outside the body mask")
        if total_failed_slices > 15:
            print(f"    ... and {total_failed_slices - 15} more")
        print("  Investigate these specific patients/slices directly (e.g. view them in the")
        print("  viewer or 3D Slicer) before trusting this body-mask method as-is.")
 
    print("\nUse the 'body' row above as your crop window source (it's GT-independent and")
    print("works identically at train/val/test time); use the 'labeled organs' row only as")
    print("a reference to sanity-check that your chosen window comfortably contains them.")
    print("Round the final chosen pixel size UP to a multiple of 32 for compatibility with")
    print("ENet's 8x downsampling, with headroom for rounding.")

# ---------------------------------------------------------------------------
# 6. Data quality / integrity checks
# ---------------------------------------------------------------------------

def analyze_data_quality(train_dir: Path, patient_ids: list[str]) -> None:
    print("\n=== Data quality / integrity checks ===")
    issues = []
    affine_signs_by_patient: dict[str, tuple] = {}
    for pid in tqdm(patient_ids, desc="Integrity checks"):
        ct, gt, spacing, ct_nib = load_patient(train_dir, pid)

        present_classes = set(np.unique(gt).tolist())
        expected = set(range(N_CLASSES))
        if present_classes != expected:
            missing = expected - present_classes
            issues.append(f"{pid}: missing classes in GT: "
                          f"{[CLASS_NAMES[k] for k in missing]}")

        if ct.shape != gt.shape:
            issues.append(f"{pid}: CT/GT shape mismatch: {ct.shape} vs {gt.shape}")

        # Flag any non-standard axis ordering / flips via the affine's sign pattern.
        # First three values of affine matrix diagonal encode the scaling along each axis, sign tells the orientation, since
        # negative value means flipped scan along that dimension.
        affine_diag_signs = tuple(np.sign(np.diag(ct_nib.affine)[:3]).tolist()) #tuple so that it's hashable
        affine_signs_by_patient[pid] = affine_diag_signs

    if issues:
        print(f"Found {len(issues)} potential issues:")
        for i in issues:
            print(f"  - {i}")
    else:
        print("No missing-class or shape-mismatch issues found across all patients.")

    # Compare affine sign-patterns across all patients: flag anyone who
    # disagrees with the majority orientation as a potential flip/mislabel.
    print("\n=== Affine orientation consistency (which direction a voxel step points in)===")
    sign_counts = {}
    for pid, signs in affine_signs_by_patient.items():
        sign_counts.setdefault(signs, []).append(pid) #Add sign patterns as keys and patients as values
 
    if len(sign_counts) == 1:
        (only_pattern,) = sign_counts.keys()
        print(f"All {len(patient_ids)} patients share the same affine sign pattern: "
              f"{only_pattern} -- no orientation inconsistencies detected.")
    else:
        majority_pattern = max(sign_counts, key=lambda k: len(sign_counts[k]))
        print(f"Found {len(sign_counts)} distinct affine sign patterns across patients "
              f"(expected exactly 1 if all scans share the same orientation convention):")
        for pattern, pids in sorted(sign_counts.items(), key=lambda kv: -len(kv[1])):
            flag = "  <-- majority" if pattern == majority_pattern else "  <-- MISMATCH, investigate"
            print(f"  {pattern}: {len(pids)} patient(s) {pids}{flag}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_fingerprint_row(pid: str, ct: np.ndarray, gt: np.ndarray, spacing: tuple) -> dict:
    dx, dy, dz = spacing
    x, y, z = ct.shape
    body_mask = ct > BODY_HU_THRESHOLD
    body_area_frac = body_mask.sum() / body_mask.size

    return {
        "patient_id": pid,
        "dx": dx, "dy": dy, "dz": dz,
        "x": x, "y": y, "z_extent": z,
        "hu_min": int(ct.min()), "hu_max": int(ct.max()),
        "body_area_frac": float(body_area_frac),
    }


def main(args: argparse.Namespace) -> None:
    train_dir = Path(args.source_dir) / "train"
    dest_dir = Path(args.dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True) #parents = True ensures parent directories are made if not existent yet

    patient_ids = get_patient_ids(train_dir)
    if args.max_patients:
        patient_ids = patient_ids[:args.max_patients]
    print(f"Found {len(patient_ids)} patients in {train_dir}")

    # Single pass to build the per-patient geometry/FOV fingerprint
    # (cheap enough to do eagerly; intensity/class-count passes are separate
    # since they're more expensive and benefit from their own progress bars).
    fingerprint_rows = []
    for pid in tqdm(patient_ids, desc="Building geometry fingerprint"):
        ct, gt, spacing, _ = load_patient(train_dir, pid)
        fingerprint_rows.append(build_fingerprint_row(pid, ct, gt, spacing))

    with open(dest_dir / "fingerprint.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fingerprint_rows[0].keys()))
        writer.writeheader()
        writer.writerows(fingerprint_rows)

    analyze_geometry(fingerprint_rows, dest_dir)
    analyze_fov(fingerprint_rows, dest_dir)
    analyze_body_bbox_extent(train_dir, patient_ids, dest_dir)
    analyze_intensity(train_dir, patient_ids, dest_dir, max_patients=args.max_patients)
    analyze_class_balance(train_dir, patient_ids, dest_dir)
    analyze_z_coverage(train_dir, patient_ids, dest_dir)
    analyze_data_quality(train_dir, patient_ids)

    print(f"\nAll outputs written to {dest_dir}/")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EDA for the SegTHOR dataset")
    parser.add_argument("--source_dir", type=str, default="data/segthor_part1",
                        help="Folder containing train/Patient_XX/... (raw NIfTI data)")
    parser.add_argument("--dest_dir", type=str, default="eda_output",
                        help="Where to write CSVs and plots")
    parser.add_argument("--max_patients", type=int, default=None,
                        help="Optional: limit to first N patients, for a quick test run")
    return parser.parse_args()


if __name__ == "__main__":
    np.random.seed(0)
    main(get_args())