#!/usr/bin/env python3

# MIT License

# Copyright (c) 2025 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import json
import random
import torch

from pathlib import Path
from typing import Callable, Union

from torch import Tensor
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


def make_dataset(root, subset) -> list[tuple[Path, Path | None]]:
    assert subset in ['train', 'val', 'test']

    root = Path(root)
    print(f"> {root=}")

    img_path = root / subset / 'img'
    full_path = root / subset / 'gt'

    # Images are .npy (raw, clipped HU float32 arrays) for segthor datasets produced by the updated slice_segthor.py. 
    # TOY2 is still produced separately by gen_two_circles.py as plain 0-255 .png, so try .npy first, and fall back to .png, 
    # so both dataset types keep working without changes to the TOY2 generation script.
    images: list[Path] = sorted(img_path.glob("*.npy"))
    if not images:
        images = sorted(img_path.glob("*.png"))

    full_labels: list[Path | None]
    if subset != 'test':
        full_labels = sorted(full_path.glob("*.png"))
    else:
        full_labels = [None] * len(images)

    return list(zip(images, full_labels))


class SliceDataset(Dataset):
    def __init__(self, subset, root_dir, img_transform=None,
                 gt_transform=None, augmentation="combination_no_noise", augmentation_probability=0.5, equalize=False, debug=False, context_size: int =1):
        self.root_dir: str = root_dir
        self.img_transform: Callable = img_transform
        self.gt_transform: Callable = gt_transform
        self.augmentation: str = augmentation
        self.augmentation_probability: float = augmentation_probability
        self.subset = subset
        self.context_size: int = context_size

        if not 0.0 <= self.augmentation_probability <= 1.0:
            raise ValueError(
                f"augmentation_probability must be in [0.0, 1.0], got {self.augmentation_probability}"
            )
        self.equalize: bool = equalize

        self.test_mode: bool = subset == 'test'

        self.files = make_dataset(root_dir, subset)
        if debug:
            self.files = self.files[:10]

        # Fill value for augmentation's empty corners (rotation/translation),
        # matching the SAME "air" value slice_segthor.py's crop/pad already
        # uses -- NOT a flat 0, since 0 means something different on raw/
        # normalized HU data than it does on the legacy [0,1] TOY2 data.
        # Falls back to 0 for TOY2, which has no normalization_stats.json at
        # all (it's produced by the separate gen_two_circles.py pipeline).
        stats_path = Path(root_dir) / "normalization_stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                self.pad_fill_value = json.load(f)["pad_fill_value"]
        else:
            self.pad_fill_value = 0.0

        print(f">> Created {subset} dataset with {len(self)} images (context_size={context_size})...")

    def __len__(self):
        return len(self.files)
    
    def _patient_id(self, img_path: Path) -> str:
        return img_path.stem.rsplit('_', 1)[0]

    def _get_neighbour_path(self, index: int, offset: int) -> Path:
        center_path, _ = self.files[index]
        neighbor_index = index + offset
 
        if neighbor_index < 0 or neighbor_index >= len(self.files):
            return center_path
        neighbor_path, _ = self.files[neighbor_index]
        if self._patient_id(neighbor_path) != self._patient_id(center_path):
            return center_path
        return neighbor_path

    def __getitem__(self, index) -> dict[str, Union[Tensor, int, str]]:
        img_path, gt_path = self.files[index]
 
        half = self.context_size // 2
        slice_paths = [self._get_neighbour_path(index, offset)
                       for offset in range(-half, half + 1)]
 
        #load every slice in the stack as a raw, no transformations
        opened = []
        is_npy_list = []
        for p in slice_paths:
            is_npy = p.suffix == ".npy"
            is_npy_list.append(is_npy)
            if is_npy:
                img_arr = np.load(p)
                opened.append(torch.from_numpy(img_arr)[None, ...])  # (1, H, W)
            else:
                opened.append(Image.open(p))
        is_npy = is_npy_list[0]
 
        if not self.test_mode:
            gt_open = Image.open(gt_path)
 
            augmentation = self.augmentation
 
            if (
                augmentation in {"rotation", "translation", "scaling", "noise"}
                and random.random() >= self.augmentation_probability
            ):
                augmentation = "none"
 
            def wh(img_open):
                return (img_open.shape[-1], img_open.shape[-2]) if is_npy else (img_open.width, img_open.height)
 
            #Decide the augmentation parameters per sample, apply
            # them to every slice in the stack and to the GT, so
            # the whole stack stays consistent with the label.
            angle = 0.0
            translate = [0, 0]
            scale_factor = 1.0
            do_affine = False
 
            if augmentation == "rotation":
                angle = random.uniform(-10, 10)
                do_affine = True
 
            elif augmentation == "translation":
                w, h = wh(opened[half])
                max_dx = int(0.05 * w)
                max_dy = int(0.05 * h)
                translate = [random.randint(-max_dx, max_dx), random.randint(-max_dy, max_dy)]
                do_affine = True
 
            elif augmentation == "scaling":
                scale_factor = random.uniform(0.9, 1.1)
                do_affine = True
 
            elif augmentation in {"combination", "combination_no_noise"}:
                probability = self.augmentation_probability
                apply_rotation = random.random() < probability
                apply_translation = random.random() < probability
                apply_scaling = random.random() < probability
 
                angle = random.uniform(-10, 10) if apply_rotation else 0.0
 
                if apply_translation:
                    w, h = wh(opened[half])
                    max_dx = int(0.05 * w)
                    max_dy = int(0.05 * h)
                    translate = [random.randint(-max_dx, max_dx), random.randint(-max_dy, max_dy)]
 
                scale_factor = random.uniform(0.9, 1.1) if apply_scaling else 1.0
                do_affine = apply_rotation or apply_translation or apply_scaling
 
            elif augmentation not in {"none", "noise"}:
                raise ValueError(f"Unknown augmentation: {augmentation}")
 
            if do_affine:
                opened = [
                    TF.affine(o, angle=angle, translate=translate, scale=scale_factor,
                             shear=[0.0, 0.0], interpolation=InterpolationMode.BILINEAR,
                             fill=self.pad_fill_value)
                    for o in opened
                ]
                gt_open = TF.affine(
                    gt_open, angle=angle, translate=translate, scale=scale_factor,
                    shear=[0.0, 0.0], interpolation=InterpolationMode.NEAREST, fill=0
                )
        else:
            augmentation = self.augmentation
 
        # Build the slice channels and stack them -- same per-slice
        # logic as main's single-slice version repeated across the stack.
        channels = []
        for o, this_is_npy in zip(opened, is_npy_list):
            if this_is_npy:
                channels.append(o.float())
            else:
                channels.append(self.img_transform(np.array(o)))
        img: Tensor = torch.cat(channels, dim=0)
 
        if self.subset == "train":
            apply_noise = (
                augmentation == "noise"
                or (augmentation == "combination" and random.random() < self.augmentation_probability)
            )
            if apply_noise:
                img = img + torch.randn_like(img) * 0.03
                if not is_npy:
                    img = img.clamp(0.0, 1.0)
 
        data_dict = {"images": img,
                     "stems": img_path.stem}
 
        if not self.test_mode:
            gt: Tensor = self.gt_transform(gt_open)
 
            _, W, H = img.shape
            K, _, _ = gt.shape
            assert gt.shape == (K, W, H)
 
            data_dict["gts"] = gt
 
        return data_dict