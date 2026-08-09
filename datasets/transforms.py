"""Keypoint-aware transforms.

Each transform's __call__ takes (img, keypoints) and returns (img, keypoints).
Image-only ops (color jitter, normalize, ...) pass keypoints through unchanged.
Keypoints are an (N, 2) array in (x, y) pixel coordinates of the current image.
"""

import random

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torchvision import transforms as T


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMAGENET_MEAN_BGR = (0.406, 0.456, 0.485)
IMAGENET_STD_BGR = (0.225, 0.224, 0.229)

# Project-wide output stride for the density model. Trainer, Bird, and
# DMCountLoss must all use the same value, so it lives in one place.
DOWNSAMPLE_RATIO = 8


def _resize_with_kp(img, keypoints, new_wd, new_ht):
    """Resize PIL `img` to (new_wd, new_ht) and rescale keypoints by the same factor.

    No-op (returns inputs unchanged) when the target size matches the current one.
    """
    wd, ht = img.size
    if (new_wd, new_ht) == (wd, ht):
        return img, keypoints
    img = img.resize((new_wd, new_ht), Image.BICUBIC)
    if len(keypoints):
        keypoints = keypoints * np.array([new_wd / wd, new_ht / ht])
    return img, keypoints


def _crop_with_kp(img, keypoints, i, j, size):
    """Crop PIL `img` to a `size`x`size` window at (top=i, left=j); drop kps outside it."""
    img = F.crop(img, i, j, size, size)
    if len(keypoints):
        keypoints = keypoints - np.array([j, i])
        mask = (keypoints[:, 0] >= 0) & (keypoints[:, 0] < size) & (keypoints[:, 1] >= 0) & (keypoints[:, 1] < size)
        keypoints = keypoints[mask]
    else:
        keypoints = np.empty((0, 2))
    return img, keypoints


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, keypoints):
        for t in self.transforms:
            img, keypoints = t(img, keypoints)
        return img, keypoints


class RandomScale:
    """Resize image and rescale keypoints by a uniformly random factor."""

    def __init__(self, scale_range=(0.8, 1.25)):
        self.lo, self.hi = scale_range

    def __call__(self, img, keypoints):
        scale = random.uniform(self.lo, self.hi)
        wd, ht = img.size
        new_wd = max(int(round(wd * scale)), 1)
        new_ht = max(int(round(ht * scale)), 1)
        return _resize_with_kp(img, keypoints, new_wd, new_ht)


class ResizeLongestEdge:
    """Resize so the longer edge equals `size`, preserving aspect ratio.

    Keypoints are scaled by the same factor. Used at val/test time to probe
    model accuracy at lower input resolutions.
    """

    def __init__(self, size: int):
        self.size = size

    def __call__(self, img, keypoints):
        wd, ht = img.size
        long_edge = max(wd, ht)
        if long_edge == self.size:
            return img, keypoints
        scale = self.size / long_edge
        new_wd = max(int(round(wd * scale)), 1)
        new_ht = max(int(round(ht * scale)), 1)
        return _resize_with_kp(img, keypoints, new_wd, new_ht)


class RandomLongestEdgeResize:
    """Resize so the longer edge is uniformly sampled from `sizes`.

    Decouples "camera resolution" from "camera zoom": sampling resolution
    here exposes the model to a range of bird pixel-sizes, so a checkpoint
    trained this way is robust to deployment-time --test-size choice.
    """

    def __init__(self, sizes):
        self.sizes = list(sizes)
        if not self.sizes:
            raise ValueError("RandomLongestEdgeResize requires at least one size")

    def __call__(self, img, keypoints):
        target = random.choice(self.sizes)
        wd, ht = img.size
        long_edge = max(wd, ht)
        scale = target / long_edge
        new_wd = max(int(round(wd * scale)), 1)
        new_ht = max(int(round(ht * scale)), 1)
        return _resize_with_kp(img, keypoints, new_wd, new_ht)


class RandomSquareCrop:
    """Random square crop. Upscales first if image is smaller than `size`."""

    def __init__(self, size):
        self.size = size

    def __call__(self, img, keypoints):
        wd, ht = img.size
        st = min(wd, ht)
        if st < self.size:
            rr = self.size / st
            new_wd = int(round(wd * rr))
            new_ht = int(round(ht * rr))
            img, keypoints = _resize_with_kp(img, keypoints, new_wd, new_ht)
            wd, ht = img.size

        i = random.randint(0, ht - self.size)
        j = random.randint(0, wd - self.size)
        return _crop_with_kp(img, keypoints, i, j, self.size)


class RandomCropOrPad:
    """Random `size`x`size` crop, padding (not upscaling) when the image is
    smaller than `size` along either axis.

    Padding preserves the apparent bird pixel-size set by an earlier
    `RandomLongestEdgeResize`, which would be destroyed by `RandomSquareCrop`
    (it upscales). Padding fill is black, matching `PadToMultiple` on the
    val/test side.
    """

    def __init__(self, size: int, fill=(0, 0, 0)):
        self.size = size
        self.fill = fill

    def __call__(self, img, keypoints):
        wd, ht = img.size

        pad_w = max(0, self.size - wd)
        pad_h = max(0, self.size - ht)
        if pad_w or pad_h:
            left = random.randint(0, pad_w)
            top = random.randint(0, pad_h)
            new_img = Image.new(img.mode, (wd + pad_w, ht + pad_h), self.fill)
            new_img.paste(img, (left, top))
            img = new_img
            if len(keypoints):
                keypoints = keypoints + np.array([left, top])
            wd, ht = img.size

        if wd > self.size or ht > self.size:
            i = random.randint(0, ht - self.size)
            j = random.randint(0, wd - self.size)
            img, keypoints = _crop_with_kp(img, keypoints, i, j, self.size)
        return img, keypoints


class RandomHFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, keypoints):
        if random.random() < self.p:
            img = F.hflip(img)
            if len(keypoints):
                keypoints = keypoints.copy()
                keypoints[:, 0] = (img.size[0] - 1) - keypoints[:, 0]
        return img, keypoints


class RandomVFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, keypoints):
        if random.random() < self.p:
            img = F.vflip(img)
            if len(keypoints):
                keypoints = keypoints.copy()
                keypoints[:, 1] = (img.size[1] - 1) - keypoints[:, 1]
        return img, keypoints


class RandomRot90:
    """Random k * 90° rotation. Requires a square image (assert)."""

    def __call__(self, img, keypoints):
        k = random.randint(0, 3)
        if k == 0:
            return img, keypoints
        w, h = img.size
        if w != h:
            raise ValueError(f"RandomRot90 requires square input, got {(w, h)}")
        img = F.rotate(img, 90 * k)
        if len(keypoints):
            x = keypoints[:, 0].copy()
            y = keypoints[:, 1].copy()
            keypoints = keypoints.copy()
            # F.rotate is counter-clockwise.
            if k == 1:
                keypoints[:, 0] = y
                keypoints[:, 1] = (w - 1) - x
            elif k == 2:
                keypoints[:, 0] = (w - 1) - x
                keypoints[:, 1] = (h - 1) - y
            else:  # k == 3
                keypoints[:, 0] = (h - 1) - y
                keypoints[:, 1] = x
        return img, keypoints


class ColorJitter:
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05):
        self._cj = T.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, img, keypoints):
        return self._cj(img), keypoints


class RandomGamma:
    def __init__(self, gamma_range=(0.7, 1.3), p=0.5):
        self.lo, self.hi = gamma_range
        self.p = p

    def __call__(self, img, keypoints):
        if random.random() < self.p:
            img = F.adjust_gamma(img, random.uniform(self.lo, self.hi))
        return img, keypoints


class ToTensor:
    def __call__(self, img, keypoints):
        return F.to_tensor(img), keypoints


class Normalize:
    def __init__(self, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.mean = mean
        self.std = std

    def __call__(self, img, keypoints):
        return F.normalize(img, self.mean, self.std), keypoints


class RandomGaussianNoise:
    """Adds Gaussian noise in (already-normalized) tensor space."""

    def __init__(self, std=0.02, p=0.5):
        self.std = std
        self.p = p

    def __call__(self, img, keypoints):
        if self.std > 0 and random.random() < self.p:
            img = img + torch.randn_like(img) * self.std
        return img, keypoints


class PadToMultiple:
    """Right/bottom-pad a (C, H, W) tensor so H and W are multiples of `m`.

    Padding is black, matching the train-time fill of `RandomCropOrPad` so the
    model sees the same kind of border in both phases. The input is already
    normalized, so black is `-mean/std` per channel, not 0. Keypoints are
    unchanged because padding is appended after the original content.
    """

    def __init__(self, multiple: int, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.m = multiple
        self.fill = [-m / s for m, s in zip(mean, std)]

    def __call__(self, img, keypoints):
        h, w = img.shape[-2:]
        pad_w = (self.m - w % self.m) % self.m
        pad_h = (self.m - h % self.m) % self.m
        if pad_h or pad_w:
            # torch.nn.functional.pad: (left, right, top, bottom)
            img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h))
            fill = torch.tensor(self.fill, dtype=img.dtype, device=img.device).view(-1, 1, 1)
            if pad_h:
                img[..., h:, :] = fill
            if pad_w:
                img[..., :, w:] = fill
        return img, keypoints


def build_train_transform(
    crop_size,
    target_sizes=(1280,),  # 1280/720P
    scale_range=(0.85, 1.2),
    color_jitter=(0.4, 0.4, 0.3, 0.1),
    gamma_range=(0.7, 1.3),
    gamma_p=0.5,
    noise_std=0.02,
    noise_p=0.5,
    hflip_p=0.5,
):
    """Build the train-time augmentation pipeline.

    Order: resolution-sample → zoom-jitter → crop-or-pad → photometric → tensor.
    `target_sizes` covers the deployment resolution range so the model is
    robust to --test-size at inference; `scale_range` is now a narrow zoom
    jitter on top of that.
    """
    return Compose(
        [
            RandomLongestEdgeResize(target_sizes),
            RandomScale(scale_range),
            RandomCropOrPad(crop_size),
            RandomHFlip(hflip_p),
            RandomRot90(),
            ColorJitter(*color_jitter),
            RandomGamma(gamma_range, gamma_p),
            ToTensor(),
            Normalize(),
            RandomGaussianNoise(noise_std, noise_p),
        ]
    )


def build_val_transform(downsample_ratio: int = DOWNSAMPLE_RATIO, test_size: int = 1280):
    """Val transform: optional Resize → ToTensor → Normalize → pad to multiple of `downsample_ratio`.

    When `test_size > 0` the image (and keypoints) is rescaled so the longer
    edge equals `test_size`; this is useful for probing model accuracy at
    lower inference resolutions. `test_size = 0` keeps native resolution.

    Padding is necessary because val images may not be divisible by the
    model's output stride; without it `gen_downsampled_density` would raise.
    """
    ops = []
    if test_size > 0:
        ops.append(ResizeLongestEdge(test_size))
    ops.extend([ToTensor(), Normalize(), PadToMultiple(downsample_ratio)])
    return Compose(ops)
