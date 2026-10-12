import argparse
import re
from pathlib import Path
import nibabel as nib
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
from skimage.transform import resize

TARGET_XY_MM = 1.0
TARGET_Z_MM = 2.5
GRID_SHAPE = (512, 512)
LABEL_VALUES = {0, 63, 126, 189, 252}
def resampled_xy_shape(shape, spacing):
    #match the original preprocessing implementation
    dx, dy = spacing[:2]
    return (
        max(1, int(round(shape[0] * dy / TARGET_XY_MM))),
        max(1, int(round(shape[1] * dx / TARGET_XY_MM))),
    )
    
def undo_centered_crop(pred, shape):
    #Invert crop_or_pad_to_grid 
    rows, cols = shape
    out_rows, out_cols = pred.shape
    r_start = int(round(rows / 2 - out_rows / 2))
    c_start = int(round(cols / 2 - out_cols / 2))
    r0, r1 = max(0, r_start), min(rows, r_start + out_rows)
    c0, c1 = max(0, c_start), min(cols, c_start + out_cols)
    out = np.zeros(shape, dtype=pred.dtype)
    if r0 < r1 and c0 < c1:
        out[r0:r1, c0:c1] = pred[
            r0 - r_start:r1 - r_start,
            c0 - c_start:c1 - c_start,
        ]
    return out

#reconstruct the slices of the patient
def reconstruct_patient(patient, paths, gt_folder, output_folder, resample_z):
    source = gt_folder / patient / 'GT.nii.gz'
    original = nib.load(str(source))
    if len(original.shape) != 3:
        raise ValueError(f'Not a 3D scan: {source}')
    X, Y, Z = original.shape
    spacing = original.header.get_zooms()[:3]
    dz = spacing[2]
    expected_z = (
        max(1, int(round((Z - 1) * dz / TARGET_Z_MM)) + 1)
        if resample_z and Z >= 2 else Z
    )
    indexes = set(paths)
    if indexes != set(range(expected_z)):
        raise ValueError(
            f'found {len(indexes)}; missing={sorted(set(range(expected_z)) - indexes)[:10]}'
        )

    xy_shape = resampled_xy_shape((X, Y), spacing)
    pred_volume = np.zeros((X, Y, expected_z), dtype=np.uint8)
    for z, path in sorted(paths.items()):
        with Image.open(path) as img:
            raw = np.asarray(img)
        if raw.shape != GRID_SHAPE or raw.ndim != 2:
            raise ValueError(f'{path}: unexpected PNG shape {raw.shape}')
        if not set(np.unique(raw)).issubset(LABEL_VALUES):
            raise ValueError(f'{path}: unexpected mask labels {np.unique(raw)}')
        labels = (raw // 63).astype(np.uint8)
        unpadded = undo_centered_crop(labels, xy_shape)
        restored = resize(unpadded, (X, Y), order=0, mode='constant', preserve_range=True, anti_aliasing=False,)
        pred_volume[:, :, z] = restored.astype(np.uint8)

    if resample_z and Z >= 2:
        #sample original slice positions
        positions = np.arange(Z, dtype=np.float64) * dz / TARGET_Z_MM
        indexes_original = map_coordinates(
            np.arange(expected_z, dtype=np.float64), [positions],
            order=0, mode='nearest',
        ).astype(int)
        pred_volume = pred_volume[:, :, indexes_original]

    if pred_volume.shape != (X, Y, Z):
        raise ValueError(f'{patient}: wrong final shape: {pred_volume.shape}')
    result = nib.Nifti1Image(pred_volume, original.affine, original.header.copy())
    result.set_data_dtype(np.uint8)
    output_folder.mkdir(parents=True, exist_ok=True)
    nib.save(result, str(output_folder / f'{patient}.nii.gz'))
    print(f'{patient}: {len(paths)} PNGs -> {pred_volume.shape}')
    
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_folder', type=Path, required=True)
    p.add_argument('--gt_folder', type=Path, required=True)
    p.add_argument('--dest_folder', type=Path, required=True)
    p.add_argument('--resample_z', action='store_true')
    args = p.parse_args()
    if not args.data_folder.is_dir():
        p.error(f'No input directory: {args.data_folder}')
    groups = {}
    pattern = re.compile(r'^(Patient_\d\d)_(\d{4})\.png$')
    for path in args.data_folder.glob('*.png'):
        match = pattern.fullmatch(path.name)
        if not match:
            raise ValueError(f'Unexpected PNG name: {path}')
        patient, z_str = match.groups()
        z = int(z_str)
        group = groups.setdefault(patient, {})
        if z in group:
            raise ValueError(f'Duplicate slice: {path}')
        group[z] = path
    if not groups:
        p.error('No PNG predictions found')
    print(f'Found {len(groups)} patients; resample_z={args.resample_z}')
    for patient, paths in sorted(groups.items()):
        reconstruct_patient(patient, paths, args.gt_folder, args.dest_folder, args.resample_z)

if __name__ == '__main__':
    main()
