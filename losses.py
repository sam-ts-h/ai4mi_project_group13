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


from torch import einsum

from utils import simplex, sset
SMOOTH = 1e-8


class CrossEntropy():
    def __init__(self, **kwargs):
        # Self.idk is used to filter out some classes of the target mask. Use fancy indexing
        self.idk = kwargs['idk']
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        log_p = (pred_softmax[:, self.idk, ...] + 1e-10).log()
        mask = weak_target[:, self.idk, ...].float()

        loss = - einsum("bkwh,bkwh->", mask, log_p)
        loss /= mask.sum() + 1e-10

        return loss


class PartialCrossEntropy(CrossEntropy):
    def __init__(self, **kwargs):
        super().__init__(idk=[1], **kwargs)


class DiceLoss():
    def __init__(self, **kwargs):
        self.idk = kwargs['idk']
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, predSoftmax, target):
        assert predSoftmax.shape == target.shape
        assert simplex(predSoftmax)
        assert sset(target, [0, 1])

        pred = predSoftmax[:, self.idk, ...]
        mask = target[:, self.idk, ...].float()

        #axes to sum away (batch,w,h )keeps axis 1 so we get one number per class
        sumDims = (0, 2, 3)

        intersection = (pred * mask).sum(dim=sumDims)
        predSum = pred.sum(dim=sumDims)
        maskSum = mask.sum(dim=sumDims)

        dice = (2 * intersection + SMOOTH) / (predSum + maskSum + SMOOTH)

        return 1 - dice.mean()


class BoundaryLoss():
    def __init__(self, **kwargs):
        self.idk = kwargs['idk']
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, predSoftmax, distMaps):
        assert predSoftmax.shape == distMaps.shape
        assert simplex(predSoftmax)

        pred = predSoftmax[:, self.idk, ...]
        dist = distMaps[:, self.idk, ...].float()

        # we do the pred times the distance, might be below zero but thats fine
        return (pred * dist).mean()


class CombinedLoss():
    def __init__(self, ce, dice=None, boundary=None, alpha=0.0):
        self.ce = ce
        self.dice = dice
        self.boundary = boundary
        self.alpha = alpha

    def partNames(self) -> list[str]:
        #just easy helper for the names in the loss dict
        names = ['ce']
        if self.dice is not None:
            names.append('dice')
        if self.boundary is not None:
            names.append('boundary')

        return names

    def __call__(self, predSoftmax, target, distMaps=None):
        parts = {'ce': self.ce(predSoftmax, target)}
        total = parts['ce']

        if self.dice is not None:
            parts['dice'] = self.dice(predSoftmax, target)
            total = total + parts['dice']

        if self.boundary is not None:
            assert distMaps is not None
            parts['boundary'] = self.boundary(predSoftmax, distMaps)
            #rebalance: region terms fade as boundary rises, so total stays comparable
            total = (1 - self.alpha) * total + self.alpha * parts['boundary']

        return total, parts
