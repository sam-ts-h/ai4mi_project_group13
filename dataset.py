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

from pathlib import Path
from typing import Callable, Union

from torch import Tensor
from PIL import Image
from torch.utils.data import Dataset
import torch
import numpy as np
import torchvision.transforms.functional as TF


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

AUG_CONFIGS: dict[str, dict | None] = {
    "none": None,
    "gamma": {"gamma": 0.3},
    "brightness": {"brightness": 0.1},
    "contrast": {"contrast": 0.2},
    "blur": {"blur": 0.4},
    "noise": {"noise": 0.02},
    "all": {"gamma": 0.3, "brightness": 0.1, "contrast": 0.2, "blur": 0.4, "noise": 0.02},
}

def augment_image(img: Tensor, cfg: dict, rng) -> Tensor:
    
    # brightness
    if (p := cfg.get("brightness")) and rng.random() < 0.3:
        img = img + rng.uniform(-p, p)

    # contrast
    if (p := cfg.get("contrast")) and rng.random() < 0.3:
        mean = img.mean()
        img = (img - mean) * rng.uniform(1 - p, 1 + p) + mean

    # gamma
    if (p := cfg.get("gamma")) and rng.random() < 0.3:
        img = img.clamp(0, 1) ** rng.uniform(1 - p, 1 + p)

    # blur
    if (p := cfg.get("blur")) and rng.random() < 0.2:
        sigma = rng.uniform(0.3, p)
        img = TF.gaussian_blur(img, kernel_size=5, sigma=sigma)

    # noise
    if (p := cfg.get("noise")) and rng.random() < 0.2:
        img = img + torch.randn_like(img) * p

    # blur
    if (p := cfg.get("blur")) and rng.random() < 0.2:
        sigma = rng.uniform(0.3, p)
        img = TF.gaussian_blur(img, kernel_size=5, sigma=sigma)

    return img.clamp(0, 1)

class SliceDataset(Dataset):
    def __init__(self, subset, root_dir, img_transform=None,
                 gt_transform=None, augment: dict | None = None, equalize=False, debug=False):
        self.root_dir: str = root_dir
        self.img_transform: Callable = img_transform
        self.gt_transform: Callable = gt_transform
        self.augmentation: dict = augment
        self.equalize: bool = equalize
        self.rng = np.random.default_rng()

        self.test_mode: bool = subset == 'test'

        self.files = make_dataset(root_dir, subset)
        if debug:
            self.files = self.files[:10]

        print(f">> Created {subset} dataset with {len(self)} images...")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index) -> dict[str, Union[Tensor, int, str]]:
        img_path, gt_path = self.files[index]

        # ! from .uint8 (0 - 255) to .float32 (0 - 1) or after z-scoring (-3 to 3) or something similar
        img: Tensor = self.img_transform(Image.open(img_path))

        # TODO: implement augmentation
        if self.augmentation:
            img = augment_image(img, self.augmentation, self.rng)

        data_dict = {"images": img,
                     "stems": img_path.stem}

        

        if not self.test_mode:
            gt: Tensor = self.gt_transform(Image.open(gt_path))

            _, W, H = img.shape
            K, _, _ = gt.shape
            assert gt.shape == (K, W, H)

            data_dict["gts"] = gt

        return data_dict
