import torch
import numpy as np
import pytest
from cellmap_data.transforms.augment.gaussian_blur import GaussianBlur
from cellmap_data.transforms.augment.random_contrast import RandomContrast
from cellmap_data.transforms.augment.gaussian_noise import GaussianNoise
from cellmap_data.transforms.augment.random_gamma import RandomGamma
from cellmap_data.transforms.augment.binarize import Binarize
from cellmap_data.transforms.augment.nan_to_num import NaNtoNum
from cellmap_data.transforms.augment.normalize import Normalize


def test_gaussian_blur_forward():
    t = GaussianBlur(sigma=1.0)
    x = torch.ones(1, 5, 5)
    y = t.forward(x)
    assert y.shape == x.shape


def test_random_contrast_forward():
    t = RandomContrast()
    x = torch.ones(3, 8, 8)
    y = t.forward(x)
    assert y.shape == x.shape


def test_gaussian_noise_forward():
    t = GaussianNoise(mean=0.0, std=0.1)
    x = torch.zeros(2, 4, 4)
    y = t.forward(x)
    assert y.shape == x.shape
    assert not torch.equal(x, y)


def test_random_gamma_forward():
    t = RandomGamma()
    x = torch.ones(2, 4, 4)
    y = t.forward(x)
    assert y.shape == x.shape


def test_binarize_transform():
    t = Binarize(threshold=0.5)
    x = torch.tensor([0.2, 0.6, 0.8], dtype=torch.float32)
    y = t.transform(x)
    assert torch.all((y == 0) | (y == 1))


def test_nan_to_num_transform():
    t = NaNtoNum(params={"nan": 0})
    x = torch.tensor([1.0, float("nan"), 2.0], dtype=torch.float32)
    y = t.transform(x)
    assert not torch.isnan(y).any()


def test_normalize_transform():
    t = Normalize(shift=0, scale=1)
    x = torch.tensor([0, 128, 255], dtype=torch.float32)
    y = t.transform(x)
    assert y.min() >= 0
    assert y.max() <= 255
