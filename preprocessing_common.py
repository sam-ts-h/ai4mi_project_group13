#!/usr/bin/env python3
"""
Shared preprocessing constants and utilities for the Segthor pipeline.
These are used by both the real preprocessing script (which produces the .npy training data) and any visualization/sanity-checks. 
"""
 
import numpy as np
from scipy import ndimage
from skimage.transform import resize

# Constants, derived from the EDA (eda_segthor.py) run on the training set
 
# Foreground 0.5/99.5 percentile HU clip range (see analyze_intensity()).
CLIP_MIN = -1000.0
CLIP_MAX = 239.0

# Target in-plane spacing (mm/voxel) everything gets resampled to. dz is deliberately not resampled since this pipeline still
# treats every slice as an independent 2D training sample, so z spacing never enters the data the network sees.
TARGET_SPACING_MM = 1.0

# Final fixed grid size every preprocessed slice is padded/cropped to, so batches can be stacked. 
# Derived from the EDA's "body, largest 2D connected component per slice" bounding-box measurement (335.0mm x 514.2mm)
# This is already generous since it covers the full body, not just organs). 
# Rounded up to a multiple of 32 for ENet's 8x downsampling, since this using that architecture now, can change later.
GRID_ROWS = 544
GRID_COLS = 352
 
# Empriical body-mask heuristic
BODY_HU_THRESHOLD = -300
BODY_HU_UPPER = 1500

def body_mask_2d(ct_slice: np.ndarray) -> np.ndarray:
    """
    Pure 2D, per-slice body-mask extraction: threshold (both bounds), keep only the single largest connected component in this slice 
    and fill enclosed holes (tracheal lumen, etc). See eda_segthor.py's analyze_body_bbox_extent for the validation this method 
    was checked against.
    """
    raw = (ct_slice > BODY_HU_THRESHOLD) & (ct_slice < BODY_HU_UPPER)
    if not raw.any():
        return raw
    labeled, num_features = ndimage.label(raw, structure=np.ones((3, 3), dtype=int))
    sizes = ndimage.sum(raw, labeled, index=range(1, num_features + 1))
    largest_label = int(np.argmax(sizes)) + 1
    mask = labeled == largest_label
    return ndimage.binary_fill_holes(mask)
 
 
def body_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    """
    Center of mass of a boolean body mask, as (row, col) in the array's own index space. Returns None if the mask is empty 
    (e.g. a slice with no tissue above threshold at all (will fall back to slice center)).
    """
    if not mask.any():
        return None
    return ndimage.center_of_mass(mask)  # returns (row, col) for a 2D input
 
 
def crop_or_pad_to_grid(arr: np.ndarray, target_rows: int, target_cols: int,
                         center_rc: tuple[float, float], fill_value: float) -> np.ndarray:
    """
    Return a new (target_rows, target_cols) array centered on center_rc (in arr's own index space): 
    pads with fill_value wherever the window extends beyond arr's real extent, crops wherever arr is larger than the window. 
    Note this does not resample/interpolate anything, every real pixel that ends up in the output keeps its original value and 
    spacing exactly. Works for both float image data and integer mask data, given an appropriate fill_value for each.
    """
    src_rows, src_cols = arr.shape
    center_r, center_c = center_rc
 
    r_start = int(round(center_r - target_rows / 2))
    c_start = int(round(center_c - target_cols / 2))
    r_end = r_start + target_rows
    c_end = c_start + target_cols
 
    out = np.full((target_rows, target_cols), fill_value, dtype=arr.dtype)
 
    # Overlap between the desired window [r_start,r_end)x[c_start,c_end) and
    # the source array's real extent [0,src_rows)x[0,src_cols).
    src_r0, src_r1 = max(r_start, 0), min(r_end, src_rows)
    src_c0, src_c1 = max(c_start, 0), min(c_end, src_cols)
 
    if src_r0 >= src_r1 or src_c0 >= src_c1:
        return out  # degenerate: center far outside the array, no overlap at all
 
    out_r0, out_r1 = src_r0 - r_start, src_r1 - r_start
    out_c0, out_c1 = src_c0 - c_start, src_c1 - c_start
 
    out[out_r0:out_r1, out_c0:out_c1] = arr[src_r0:src_r1, src_c0:src_c1]
    return out



# Resampling to a common in-plane spacing:
# dx/dy varies per patient, so the same real organ occupies a different pixel count depending purely on which patient 
# you're looking at. Resampling fixes this and anti-aliasing ensures not too much fine detail is lost.
 
def compute_resampled_shape(orig_shape: tuple[int, int], orig_spacing_xy: tuple[float, float],
                             target_spacing_mm: float = TARGET_SPACING_MM) -> tuple[int, int]:
    """
    Given a 2D slice's shape and its (dx, dy) in-plane spacing, compute the
    new shape it should be resized to so each voxel represents
    target_spacing_mm instead of its original spacing -- preserving the same
    real-world physical extent, just at a different voxel density. Rounds to
    the nearest whole pixel.
    """
    orig_rows, orig_cols = orig_shape
    dx, dy = orig_spacing_xy
    new_rows = max(1, int(round(orig_rows * dy / target_spacing_mm)))
    new_cols = max(1, int(round(orig_cols * dx / target_spacing_mm)))
    return new_rows, new_cols
 
 
def resample_image_slice(slice2d: np.ndarray, orig_spacing_xy: tuple[float, float],
                          target_spacing_mm: float = TARGET_SPACING_MM) -> np.ndarray:
    """
    Resample a 2D CT slice (real intensity values) to target_spacing_mm.
    Linear interpolation (order=1), with anti-aliasing enabled: skimage only
    applies meaningful filtering when actually downsampling, so leaving it on
    unconditionally is safe and simpler than branching on up- vs down-
    sampling per patient/axis.
    """
    new_shape = compute_resampled_shape(slice2d.shape, orig_spacing_xy, target_spacing_mm)
    resampled = resize(slice2d, new_shape, order=1, mode="constant",
                       preserve_range=True, anti_aliasing=True)
    return resampled.astype(np.float32)
 
 
def resample_mask_slice(slice2d: np.ndarray, orig_spacing_xy: tuple[float, float],
                         target_spacing_mm: float = TARGET_SPACING_MM) -> np.ndarray:
    """
    Resample a 2D GT mask (discrete class indices) to target_spacing_mm.
    Nearest-neighbor (order=0), anti-aliasing OFF: both interpolation choices
    exist specifically to avoid inventing fractional/blended class values at
    organ boundaries -- see the earlier discussion on why masks always need
    this, image data never does.
    """
    new_shape = compute_resampled_shape(slice2d.shape, orig_spacing_xy, target_spacing_mm)
    resampled = resize(slice2d, new_shape, order=0, mode="constant",
                       preserve_range=True, anti_aliasing=False)
    return resampled.astype(slice2d.dtype)
 