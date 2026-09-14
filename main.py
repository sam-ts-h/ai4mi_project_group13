#!/usr/bin/env python3

# MIT License

# Copyright (c) 2025 Hoel Kervadec, Caroline Magg

import argparse
import warnings
import random
from typing import Any
from pathlib import Path
from pprint import pprint
from shutil import copytree, rmtree

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from torch.utils.data import DataLoader

from functools import partial

from dataset import SliceDataset
from ShallowNet import shallowCNN
from ENet import ENet
from utils import (
    Dcm,
    class2one_hot,
    probs2one_hot,
    probs2class,
    tqdm_,
    dice_coef,
    save_images
)

from losses import CrossEntropy


# ============================================================
# DATASET SETTINGS
# ============================================================

datasets_params: dict[str, dict[str, Any]] = {}

# K = number of classes
datasets_params["TOY2"] = {
    'K': 2,
    'net': shallowCNN,
    'B': 2,
    'kernels': 8,
    'factor': 2
}

# Original SEGTHOR option
datasets_params["SEGTHOR"] = {
    'K': 5,
    'net': ENet,
    'B': 8,
    'kernels': 8,
    'factor': 2
}

datasets_params["SEGTHOR_CLEAN"] = {
    'K': 5,
    'net': ENet,
    'B': 8,
    'kernels': 8,
    'factor': 2
}


# ============================================================
# PREPROCESSING DATASETS FOR COMPARISON
# ============================================================

datasets_params["SEGTHOR_baseline"] = {
    'K': 5,
    'net': ENet,
    'B': 8,
    'kernels': 8,
    'factor': 2
}

datasets_params["SEGTHOR_windowed"] = {
    'K': 5,
    'net': ENet,
    'B': 8,
    'kernels': 8,
    'factor': 2
}

datasets_params["SEGTHOR_soft"] = {
    'K': 5,
    'net': ENet,
    'B': 8,
    'kernels': 8,
    'factor': 2
}

datasets_params["SEGTHOR_windowed_antialias"] = {
    'K': 5,
    'net': ENet,
    'B': 8,
    'kernels': 8,
    'factor': 2
}


# ============================================================
# IMAGE TRANSFORMS
# ============================================================

def img_transform(img):
    img = img.convert('L')
    img = np.array(img)[np.newaxis, ...]

    # Convert PNG range 0-255 to 0-1
    img = img / 255

    img = torch.tensor(
        img,
        dtype=torch.float32
    )

    return img


def gt_transform(K, img):
    img = np.array(img)[...]

    # Convert ground-truth PNG values back to class IDs
    img = (
        img / (255 / (K - 1))
        if K != 5
        else img / 63
    )

    img = torch.tensor(
        img,
        dtype=torch.int64
    )[None, ...]

    img = class2one_hot(
        img,
        K=K
    )

    return img[0]


# ============================================================
# SET RANDOM SEED
# ============================================================

def set_seed(seed: int):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Makes GPU training more reproducible
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# SETUP
# ============================================================

def setup(args) -> tuple[
    nn.Module,
    Any,
    Any,
    DataLoader,
    DataLoader,
    int
]:

    gpu: bool = (
        args.gpu
        and torch.cuda.is_available()
    )

    device = (
        torch.device("cuda")
        if gpu
        else torch.device("cpu")
    )

    print(
        f">> Picked {device} "
        f"to run experiments"
    )


    # --------------------------------------------------------
    # Dataset/model settings
    # --------------------------------------------------------

    K: int = (
        datasets_params[args.dataset]['K']
    )

    kernels: int = (
        datasets_params[args.dataset]['kernels']
        if 'kernels'
        in datasets_params[args.dataset]
        else 8
    )

    factor: int = (
        datasets_params[args.dataset]['factor']
        if 'factor'
        in datasets_params[args.dataset]
        else 2
    )


    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    net = datasets_params[
        args.dataset
    ]['net'](
        1,
        K,
        kernels=kernels,
        factor=factor
    )

    net.init_weights()

    net.to(device)


    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    lr = 0.0005

    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=lr,
        betas=(0.9, 0.999)
    )


    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    B: int = (
        datasets_params[args.dataset]['B']
    )

    root_dir = (
        Path("data")
        / args.dataset
    )

    print(
        f">> Loading dataset from: "
        f"{root_dir}"
    )


    train_set = SliceDataset(
        'train',
        root_dir,
        img_transform=img_transform,
        gt_transform=partial(
            gt_transform,
            K
        ),
        debug=args.debug
    )


    # Generator makes shuffle reproducible
    generator = torch.Generator()
    generator.manual_seed(args.seed)


    train_loader = DataLoader(
        train_set,
        batch_size=B,
        num_workers=5,
        shuffle=True,
        generator=generator
    )


    val_set = SliceDataset(
        'val',
        root_dir,
        img_transform=img_transform,
        gt_transform=partial(
            gt_transform,
            K
        ),
        debug=args.debug
    )


    val_loader = DataLoader(
        val_set,
        batch_size=B,
        num_workers=5,
        shuffle=False
    )


    args.dest.mkdir(
        parents=True,
        exist_ok=True
    )


    return (
        net,
        optimizer,
        device,
        train_loader,
        val_loader,
        K
    )


# ============================================================
# TRAINING
# ============================================================

def runTraining(args):

    print(
        f">>> Setting up to train on "
        f"{args.dataset} with {args.mode}"
    )

    print(
        f">>> Random seed: "
        f"{args.seed}"
    )


    net, optimizer, device, train_loader, val_loader, K = setup(args)


    # --------------------------------------------------------
    # Loss function
    # --------------------------------------------------------

    if args.mode == "full":

        loss_fn = CrossEntropy(
            idk=list(range(K))
        )

    elif (
        args.mode == "partial"
        and args.dataset.startswith("SEGTHOR")
    ):

        # Do not supervise heart (class 2)
        loss_fn = CrossEntropy(
            idk=[0, 1, 3, 4]
        )

    else:
        raise ValueError(
            args.mode,
            args.dataset
        )


    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    log_loss_tra: Tensor = torch.zeros(
        (
            args.epochs,
            len(train_loader)
        )
    )

    log_dice_tra: Tensor = torch.zeros(
        (
            args.epochs,
            len(train_loader.dataset),
            K
        )
    )

    log_loss_val: Tensor = torch.zeros(
        (
            args.epochs,
            len(val_loader)
        )
    )

    log_dice_val: Tensor = torch.zeros(
        (
            args.epochs,
            len(val_loader.dataset),
            K
        )
    )


    best_dice: float = 0


    # ========================================================
    # EPOCH LOOP
    # ========================================================

    for e in range(args.epochs):

        for m in ['train', 'val']:

            match m:

                case 'train':

                    net.train()

                    opt = optimizer
                    cm = Dcm

                    desc = (
                        f">> Training "
                        f"({e:4d})"
                    )

                    loader = train_loader

                    log_loss = log_loss_tra
                    log_dice = log_dice_tra


                case 'val':

                    net.eval()

                    opt = None
                    cm = torch.no_grad

                    desc = (
                        f">> Validation "
                        f"({e:4d})"
                    )

                    loader = val_loader

                    log_loss = log_loss_val
                    log_dice = log_dice_val


            # ------------------------------------------------
            # Batch loop
            # ------------------------------------------------

            with cm():

                j = 0

                tq_iter = tqdm_(
                    enumerate(loader),
                    total=len(loader),
                    desc=desc
                )


                for i, data in tq_iter:

                    img = data[
                        'images'
                    ].to(device)

                    gt = data[
                        'gts'
                    ].to(device)


                    if opt:
                        opt.zero_grad()


                    # ----------------------------------------
                    # Sanity checks
                    # ----------------------------------------

                    assert (
                        0 <= img.min()
                        and img.max() <= 1
                    )

                    B, _, W, H = img.shape


                    # ----------------------------------------
                    # Forward pass
                    # ----------------------------------------

                    pred_logits = net(img)

                    pred_probs = F.softmax(
                        pred_logits,
                        dim=1
                    )


                    # ----------------------------------------
                    # Dice score
                    # ----------------------------------------

                    pred_seg = probs2one_hot(
                        pred_probs
                    )

                    log_dice[
                        e,
                        j:j + B,
                        :
                    ] = dice_coef(
                        pred_seg,
                        gt
                    )


                    # ----------------------------------------
                    # Loss
                    # ----------------------------------------

                    loss = loss_fn(
                        pred_probs,
                        gt
                    )

                    log_loss[
                        e,
                        i
                    ] = loss.item()


                    # ----------------------------------------
                    # Backpropagation
                    # ----------------------------------------

                    if opt:

                        loss.backward()

                        opt.step()


                    # ----------------------------------------
                    # Save validation predictions
                    # ----------------------------------------

                    if m == 'val':

                        with warnings.catch_warnings():

                            warnings.filterwarnings(
                                'ignore',
                                category=UserWarning
                            )

                            predicted_class: Tensor = (
                                probs2class(
                                    pred_probs
                                )
                            )

                            mult: int = (
                                63
                                if K == 5
                                else (
                                    255
                                    / (K - 1)
                                )
                            )

                            save_images(
                                predicted_class
                                * mult,

                                data['stems'],

                                args.dest
                                / f"iter{e:03d}"
                                / m
                            )


                    j += B


                    # ----------------------------------------
                    # Progress bar statistics
                    # ----------------------------------------

                    postfix_dict: dict[
                        str,
                        str
                    ] = {

                        "Dice":
                            f"{log_dice[e, :j, 1:].mean():05.3f}",

                        "Loss":
                            f"{log_loss[e, :i + 1].mean():5.2e}"
                    }


                    if K > 2:

                        postfix_dict |= {

                            f"Dice-{k}":
                                f"{log_dice[e, :j, k].mean():05.3f}"

                            for k in range(
                                1,
                                K
                            )
                        }


                    tq_iter.set_postfix(
                        postfix_dict
                    )


        # ====================================================
        # SAVE RESULTS AFTER EACH EPOCH
        # ====================================================

        np.save(
            args.dest / "loss_tra.npy",
            log_loss_tra
        )

        np.save(
            args.dest / "dice_tra.npy",
            log_dice_tra
        )

        np.save(
            args.dest / "loss_val.npy",
            log_loss_val
        )

        np.save(
            args.dest / "dice_val.npy",
            log_dice_val
        )


        # ====================================================
        # FIND BEST MODEL
        # Background class is excluded
        # ====================================================

        current_dice: float = (
            log_dice_val[
                e,
                :,
                1:
            ]
            .mean()
            .item()
        )


        if current_dice > best_dice:

            message = (
                f">>> Improved dice at epoch "
                f"{e}: "
                f"{best_dice:05.3f}"
                f"->{current_dice:05.3f} DSC"
            )

            print(message)

            best_dice = current_dice


            # Save information about best epoch
            with open(
                args.dest
                / "best_epoch.txt",
                'w'
            ) as f:

                f.write(message)


            # Save predictions from best epoch
            best_folder = (
                args.dest
                / "best_epoch"
            )

            if best_folder.exists():
                rmtree(best_folder)


            copytree(
                args.dest
                / f"iter{e:03d}",

                Path(best_folder)
            )


            # Save model
            torch.save(
                net,
                args.dest
                / "bestmodel.pkl"
            )

            torch.save(
                net.state_dict(),
                args.dest
                / "bestweights.pt"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        '--epochs',
        default=20,
        type=int
    )


    parser.add_argument(
        '--dataset',
        default='TOY2',
        choices=datasets_params.keys()
    )


    parser.add_argument(
        '--mode',
        default='full',
        choices=[
            'partial',
            'full'
        ]
    )


    parser.add_argument(
        '--dest',
        type=Path,
        required=True,
        help=(
            "Destination directory "
            "to save results, "
            "predictions and weights."
        )
    )


    parser.add_argument(
        '--gpu',
        action='store_true'
    )


    parser.add_argument(
        '--debug',
        action='store_true',
        help=(
            "Keep only a fraction "
            "(10 samples) of the datasets "
            "to test the training code."
        )
    )


    parser.add_argument(
        '--seed',
        default=0,
        type=int,
        help=(
            "Random seed used to make "
            "model comparison reproducible."
        )
    )


    args = parser.parse_args()


    pprint(args)


    # Important:
    # use exactly the same seed
    # for every preprocessing experiment
    set_seed(args.seed)


    runTraining(args)


if __name__ == '__main__':
    main()