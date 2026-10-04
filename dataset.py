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
                 gt_transform=None, augmentation="combination_no_noise", augmentation_probability=0.5, equalize=False, debug=False,loadDistMaps=False):
        self.root_dir: str = root_dir
        self.img_transform: Callable = img_transform
        self.gt_transform: Callable = gt_transform
        self.augmentation: str = augmentation
        self.augmentation_probability: float = augmentation_probability
        self.subset = subset

        if not 0.0 <= self.augmentation_probability <= 1.0:
            raise ValueError(
                f"augmentation_probability must be in [0.0, 1.0], got {self.augmentation_probability}"
            )
        self.equalize: bool = equalize
        self.loadDistMaps: bool = loadDistMaps

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

        print(f">> Created {subset} dataset with {len(self)} images...")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index) -> dict[str, Union[Tensor, int, str]]:
        img_path, gt_path = self.files[index]

        # Load the raw image: .npy for SEGTHOR's raw HU data, .png for
        # TOY2's legacy 0-255 data. GT is always .png, regardless of format.
        is_npy = img_path.suffix == ".npy"
        if is_npy:
            img_arr = np.load(img_path)
            img_open = torch.from_numpy(img_arr)[None, ...]  # (1, H, W) tensor -- TF.* accepts this directly
        else:
            img_open = Image.open(img_path)  # PIL Image, legacy TOY2 path

        # if we are not in test mode, open the ground truth/mask
        if not self.test_mode:
            gt_open = Image.open(gt_path)

            augmentation = self.augmentation

            if (
                augmentation in {
                    "rotation",
                    "translation",
                    "scaling",
                    "noise",
                }
                and random.random() >= self.augmentation_probability
            ):
                augmentation = "none"

            def wh():  # (width, height) -- PIL and Tensor expose this differently
                return (img_open.shape[-1], img_open.shape[-2]) if is_npy else (img_open.width, img_open.height)

            if augmentation == "rotation":
                angle = random.uniform(-10, 10)  # random angle between -10 and 10

                img_open = TF.rotate(img_open,  # image to rotate
                                     angle=angle,  # random angle
                                     interpolation=InterpolationMode.BILINEAR,  # smooth pixel values based on surrounding pixels
                                     fill=self.pad_fill_value)  # fill empty corners with real "air", not a flat 0
                gt_open = TF.rotate(gt_open,  # image to rotate
                                    angle=angle,  # random angle
                                    interpolation=InterpolationMode.NEAREST,  # nearest for the mask: keep discrete class values
                                    fill=0)

            elif augmentation == "translation":
                # Move the image by at most 5% of its width and height
                w, h = wh()
                max_dx = int(0.05 * w)
                max_dy = int(0.05 * h)

                translate = [
                    random.randint(-max_dx, max_dx),
                    random.randint(-max_dy, max_dy)
                ]

                img_open = TF.affine(
                    img_open,
                    angle=0,
                    translate=translate,
                    scale=1.0,
                    shear=[0.0, 0.0],
                    interpolation=InterpolationMode.BILINEAR,
                    fill=self.pad_fill_value
                )

                gt_open = TF.affine(
                    gt_open,
                    angle=0,
                    translate=translate,
                    scale=1.0,
                    shear=[0.0, 0.0],
                    interpolation=InterpolationMode.NEAREST,
                    fill=0
                )
            elif augmentation == "scaling":
                # scale the image by a random factor between 0.9 and 1.1
                scale_factor = random.uniform(0.9, 1.1)

                img_open = TF.affine(
                                    img_open,
                                    angle=0,
                                    translate=[0, 0],
                                    scale=scale_factor,
                                    shear=[0.0, 0.0],
                                    interpolation=InterpolationMode.BILINEAR,
                                    fill=self.pad_fill_value
                                )
                gt_open = TF.affine(
                                    gt_open,
                                    angle=0,
                                    translate=[0, 0],
                                    scale=scale_factor,
                                    shear=[0.0, 0.0],
                                    interpolation=InterpolationMode.NEAREST,
                                    fill=0
                                )
            elif augmentation in {"combination", "combination_no_noise"}:
                probability = self.augmentation_probability

                apply_rotation = random.random() < probability
                apply_translation = random.random() < probability
                apply_scaling = random.random() < probability

                angle = (
                    random.uniform(-10, 10)
                    if apply_rotation
                    else 0.0
                )

                if apply_translation:
                    w, h = wh()
                    max_dx = int(0.05 * w)
                    max_dy = int(0.05 * h)
                    translate = [
                        random.randint(-max_dx, max_dx),
                        random.randint(-max_dy, max_dy)
                    ]
                else:
                    translate = [0, 0]

                scale_factor = (
                    random.uniform(0.9, 1.1)
                    if apply_scaling
                    else 1.0
                )

                # Only interpolate when at least one augmentation was selected.
                # Otherwise the original image and mask remain unchanged.
                if (
                    apply_rotation
                    or apply_translation
                    or apply_scaling
                ):
                    img_open = TF.affine(
                        img_open,
                        angle=angle,
                        translate=translate,
                        scale=scale_factor,
                        shear=[0.0, 0.0],
                        interpolation=InterpolationMode.BILINEAR,
                        fill=self.pad_fill_value
                    )

                    gt_open = TF.affine(
                        gt_open,
                        angle=angle,
                        translate=translate,
                        scale=scale_factor,
                        shear=[0.0, 0.0],
                        interpolation=InterpolationMode.NEAREST,
                        fill=0
                    )

            elif augmentation not in {"none", "noise"}:
                raise ValueError(f"Unknown augmentation: {augmentation}")
        else:
            augmentation = self.augmentation  # test mode: no GT, but still need this defined below

        # .npy: already a correctly-scaled float32 HU tensor -- skip
        # img_transform (it calls .astype(), which only exists on NumPy
        # arrays, not tensors). .png (TOY2): convert back to a NumPy array
        # first, since img_transform expects one and applies the legacy
        # /255 rescale internally.
        if is_npy:
            img: Tensor = img_open.float()
        else:
            img: Tensor = self.img_transform(np.array(img_open))

        if self.subset == "train":
            apply_noise = (
                augmentation == "noise"
                or (
                    augmentation == "combination"
                    and random.random() < self.augmentation_probability
                )
            )

            if apply_noise:
                img = img + torch.randn_like(img) * 0.03
                if not is_npy:
                    # Only the legacy [0,1]-scaled TOY2 path has a meaningful
                    # fixed range to clamp back into; HU/normalized data has
                    # no such fixed bound.
                    img = img.clamp(0.0, 1.0)

        data_dict = {"images": img,
                     "stems": img_path.stem}

        if not self.test_mode:
            gt: Tensor = self.gt_transform(gt_open)

            _, W, H = img.shape
            K, _, _ = gt.shape
            assert gt.shape == (K, W, H)

            data_dict["gts"] = gt

            if self.loadDistMaps:
                distMapPath = gt_path.parent.parent / 'distmap' / f"{gt_path.stem}.npy"
                distMap = np.load(distMapPath).astype(np.float32)
                assert distMap.shape == gt.shape, 'shape of distmap and gt are not the same.... dataset.py says no'
                data_dict["distMaps"] = torch.from_numpy(distMap)

        return data_dict
