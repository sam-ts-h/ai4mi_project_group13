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

import random

from pathlib import Path
from typing import Callable, Union

from torch import Tensor
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


def make_dataset(root, subset) -> list[tuple[Path, Path | None]]:
    assert subset in ['train', 'val', 'test']

    root = Path(root)
    print(f"> {root=}")

    img_path = root / subset / 'img'
    full_path = root / subset / 'gt'

    images: list[Path] = sorted(img_path.glob("*.png"))
    full_labels: list[Path | None]
    if subset != 'test':
        full_labels = sorted(full_path.glob("*.png"))
    else:
        full_labels = [None] * len(images)

    return list(zip(images, full_labels))


class SliceDataset(Dataset):
    def __init__(self, subset, root_dir, img_transform=None,
                 gt_transform=None, augmentation="none", equalize=False, debug=False):
        self.root_dir: str = root_dir
        self.img_transform: Callable = img_transform
        self.gt_transform: Callable = gt_transform
        self.augmentation: str = augmentation
        self.equalize: bool = equalize

        self.test_mode: bool = subset == 'test'

        self.files = make_dataset(root_dir, subset)
        if debug:
            self.files = self.files[:10]

        print(f">> Created {subset} dataset with {len(self)} images...")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index) -> dict[str, Union[Tensor, int, str]]:
        img_path, gt_path = self.files[index]

        img_open = Image.open(img_path)

        #if we are not in test mode, open the ground truth/mask
        if not self.test_mode:
            gt_open = Image.open(gt_path)

            if self.augmentation == "rotation":
                angle = random.uniform(-10,10) #random angle between -10 and 10

                img_open = TF.rotate(img_open, #image to rotate
                                     angle=angle, #random angle
                                     interpolation = InterpolationMode.BILINEAR, #with rotate, the pixels are not always at the right spot, this interpolation method will make smooth pixel values (based on surrounding pixels)
                                     fill = 0) #after rotation --> some corners will be empty --> fill them black (value of 0)
                gt_open = TF.rotate(gt_open, #image to rotate
                                    angle=angle, #random angle
                                    interpolation = InterpolationMode.NEAREST, #use nearest for the mask, because we dont want smooth values but the original class values [0, 63, 126, 189, 252].
                                    fill = 0)

            elif self.augmentation == "translation":
                # Move the image by at most 5% of its width and height
                max_dx = int(0.05 * img_open.width)
                max_dy = int(0.05 * img_open.height)

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
                    fill=0
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
            elif self.augmentation == "scaling":
                #scale the image by a random factor between 0.9 and 1.1
                scale_factor = random.uniform(0.9,1.1)

                img_open = TF.affine(
                                    img_open,
                                    angle=0,
                                    translate= [0,0],
                                    scale=scale_factor,
                                    shear=[0.0, 0.0],
                                    interpolation=InterpolationMode.BILINEAR,
                                    fill=0
                                )
                gt_open = TF.affine(
                                    gt_open,
                                    angle=0,
                                    translate= [0,0],
                                    scale=scale_factor,
                                    shear=[0.0, 0.0],
                                    interpolation=InterpolationMode.NEAREST,
                                    fill=0
                                )
            elif self.augmentation != "none":
                raise ValueError(
                    f"Unknown augmentation: {self.augmentation}"
                )

        img: Tensor = self.img_transform(img_open)

        data_dict = {"images": img,
                     "stems": img_path.stem}

        if not self.test_mode:
            gt: Tensor = self.gt_transform(gt_open)

            _, W, H = img.shape
            K, _, _ = gt.shape
            assert gt.shape == (K, W, H)

            data_dict["gts"] = gt

        return data_dict
