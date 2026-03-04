"""Tests for cellmap_data.transforms.augment — all transform classes."""

from __future__ import annotations

import math

import torch
import pytest

from cellmap_data.transforms.augment import (
    Binarize,
    GaussianBlur,
    GaussianNoise,
    NaNtoNum,
    RandomContrast,
    RandomGamma,
)

# ---------------------------------------------------------------------------
# NaNtoNum
# ---------------------------------------------------------------------------


class TestNaNtoNum:
    def test_import_path(self):
        from cellmap_data.transforms.augment import NaNtoNum  # noqa: F401

    def test_nan_replaced_by_zero(self):
        t = NaNtoNum({"nan": 0, "posinf": None, "neginf": None})
        x = torch.tensor([float("nan"), 1.0, 2.0])
        out = t(x)
        assert not torch.isnan(out).any()
        assert out[0] == 0.0

    def test_nan_replaced_by_custom_value(self):
        t = NaNtoNum({"nan": -1.0})
        x = torch.tensor([float("nan"), 5.0])
        out = t(x)
        assert out[0] == pytest.approx(-1.0)

    def test_posinf_replaced(self):
        t = NaNtoNum({"nan": 0, "posinf": 99.0, "neginf": None})
        x = torch.tensor([float("inf"), 1.0])
        out = t(x)
        assert out[0] == pytest.approx(99.0)

    def test_neginf_replaced(self):
        t = NaNtoNum({"nan": 0, "posinf": None, "neginf": -99.0})
        x = torch.tensor([float("-inf"), 1.0])
        out = t(x)
        assert out[0] == pytest.approx(-99.0)

    def test_no_nans_unchanged(self):
        t = NaNtoNum({"nan": 0, "posinf": None, "neginf": None})
        x = torch.tensor([1.0, 2.0, 3.0])
        out = t(x)
        assert torch.allclose(out, x)

    def test_callable(self):
        t = NaNtoNum({"nan": 0})
        x = torch.full((4, 4), float("nan"))
        out = t(x)
        assert (out == 0.0).all()

    def test_repr(self):
        t = NaNtoNum({"nan": 0})
        assert "NaNtoNum" in repr(t)

    def test_transform_method_alias(self):
        """Both __call__ and .transform() should work."""
        t = NaNtoNum({"nan": 42.0})
        x = torch.tensor([float("nan")])
        assert t.transform(x)[0] == pytest.approx(42.0)

    def test_used_as_in_api(self):
        """Replicates the exact usage from API_TO_PRESERVE.md."""
        t = NaNtoNum({"nan": 0, "posinf": None, "neginf": None})
        x = torch.tensor([float("nan"), float("inf"), float("-inf"), 0.5])
        out = t(x)
        assert out[0] == 0.0
        assert not torch.isnan(out).any()

    def test_3d_tensor(self):
        t = NaNtoNum({"nan": 0})
        x = torch.full((4, 4, 4), float("nan"))
        out = t(x)
        assert not torch.isnan(out).any()
        assert out.shape == torch.Size([4, 4, 4])


# ---------------------------------------------------------------------------
# Binarize
# ---------------------------------------------------------------------------


class TestBinarize:
    def test_import_path(self):
        from cellmap_data.transforms.augment import Binarize  # noqa: F401

    def test_default_threshold_zero(self):
        t = Binarize()
        x = torch.tensor([-1.0, 0.0, 0.5, 1.0])
        out = t(x)
        expected = torch.tensor([0.0, 0.0, 1.0, 1.0])
        assert torch.allclose(out, expected)

    def test_custom_threshold(self):
        t = Binarize(0.5)
        x = torch.tensor([0.0, 0.49, 0.5, 0.51, 1.0])
        out = t(x)
        # > 0.5 → 1
        assert out[0] == 0.0
        assert out[1] == 0.0
        assert out[2] == 0.0  # 0.5 is NOT > 0.5
        assert out[3] == 1.0
        assert out[4] == 1.0

    def test_nan_preserved_after_binarize(self):
        """NaN in input → NaN in output (unknown class, not zero)."""
        t = Binarize()
        x = torch.tensor([float("nan"), 1.0, 0.0])
        out = t(x)
        assert torch.isnan(out[0])
        assert out[1] == 1.0
        assert out[2] == 0.0

    def test_repr(self):
        t = Binarize(0.5)
        assert "Binarize" in repr(t)
        assert "0.5" in repr(t)

    def test_integer_input_binarize(self):
        t = Binarize()
        x = torch.tensor([0, 1, 2, -1], dtype=torch.int32)
        out = t(x)
        # > 0 → 1
        assert out[0] == 0
        assert out[1] == 1
        assert out[2] == 1

    def test_transform_method_alias(self):
        t = Binarize(0.5)
        x = torch.tensor([0.0, 1.0])
        out = t.transform(x)
        assert torch.allclose(out, torch.tensor([0.0, 1.0]))

    def test_used_as_in_api(self):
        """Replicates the exact usage from API_TO_PRESERVE.md / datasplit default."""
        t = Binarize()
        x = torch.tensor([0.0, 0.001, 0.5, 0.9, float("nan")])
        out = t(x)
        assert out[0] == 0.0  # 0.0 not > 0
        assert out[1] == 1.0  # 0.001 > 0
        assert out[2] == 1.0
        assert out[3] == 1.0
        assert torch.isnan(out[4])

    def test_shape_preserved(self):
        t = Binarize()
        x = torch.rand(3, 4, 5)
        out = t(x)
        assert out.shape == x.shape

    def test_with_import_compose(self):
        """Compose(Binarize()) used as target_value_transforms in CellMapDataSplit."""
        import torchvision.transforms.v2 as T

        pipeline = T.Compose([T.ToDtype(torch.float), Binarize()])
        x = torch.tensor([0, 1, 2], dtype=torch.int32)
        out = pipeline(x)
        assert torch.allclose(out, torch.tensor([0.0, 1.0, 1.0]))


# ---------------------------------------------------------------------------
# GaussianNoise
# ---------------------------------------------------------------------------


class TestGaussianNoise:
    def test_output_shape(self):
        t = GaussianNoise(mean=0.0, std=0.1)
        x = torch.zeros(4, 4, 4)
        out = t(x)
        assert out.shape == x.shape

    def test_adds_noise(self):
        """Output should differ from input (with high probability for nonzero std)."""
        t = GaussianNoise(mean=0.0, std=1.0)
        x = torch.zeros(100)
        out = t(x)
        assert not torch.allclose(out, x)

    def test_zero_std_unchanged(self):
        t = GaussianNoise(mean=0.0, std=0.0)
        x = torch.ones(10)
        out = t(x)
        assert torch.allclose(out, x)

    def test_mean_offset(self):
        """High mean with large tensor → output mean close to input_mean + noise_mean."""
        t = GaussianNoise(mean=10.0, std=0.0)
        x = torch.zeros(1000)
        out = t(x)
        assert out.mean().item() == pytest.approx(10.0, abs=0.5)


# ---------------------------------------------------------------------------
# RandomContrast
# ---------------------------------------------------------------------------


class TestRandomContrast:
    def test_output_shape(self):
        t = RandomContrast((0.8, 1.2))
        x = torch.rand(4, 4, 4)
        out = t(x)
        assert out.shape == x.shape

    def test_output_dtype_preserved(self):
        t = RandomContrast((0.9, 1.1))
        x = torch.rand(8, 8).float()
        out = t(x)
        assert out.dtype == x.dtype

    def test_no_nan_output(self):
        t = RandomContrast((0.5, 1.5))
        x = torch.rand(10, 10)
        out = t(x)
        assert not torch.isnan(out).any()

    def test_clamped_to_dtype_max(self):
        """Output should not exceed the max value for the dtype."""
        from cellmap_data.utils import torch_max_value

        t = RandomContrast((1.0, 1.0))  # identity contrast ratio
        x = torch.rand(8, 8).float()
        out = t(x)
        assert (out <= torch_max_value(x.dtype) + 1e-6).all()


# ---------------------------------------------------------------------------
# RandomGamma
# ---------------------------------------------------------------------------


class TestRandomGamma:
    def test_output_shape(self):
        t = RandomGamma((0.5, 1.5))
        x = torch.rand(4, 4, 4)
        out = t(x)
        assert out.shape == x.shape

    def test_output_in_01(self):
        """After gamma, float input in [0,1] → output in [0,1]."""
        t = RandomGamma((0.5, 2.0))
        x = torch.rand(64)
        out = t(x)
        assert (out >= 0.0).all()
        assert (out <= 1.0 + 1e-5).all()

    def test_no_nan_output(self):
        t = RandomGamma((0.8, 1.2))
        x = torch.rand(10, 10)
        out = t(x)
        assert not torch.isnan(out).any()

    def test_integer_input_converted(self):
        """Integer input should be converted to float without error."""
        t = RandomGamma((0.9, 1.1))
        x = torch.randint(0, 256, (10,), dtype=torch.uint8)
        out = t(x)  # should not raise
        assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# GaussianBlur
# ---------------------------------------------------------------------------


class TestGaussianBlur:
    def test_2d_output_shape(self):
        t = GaussianBlur(kernel_size=3, sigma=1.0, dim=2)
        x = torch.rand(8, 8)
        out = t(x)
        assert out.shape == x.shape

    def test_3d_output_shape(self):
        t = GaussianBlur(kernel_size=3, sigma=1.0, dim=3)
        x = torch.rand(8, 8, 8)
        out = t(x)
        assert out.shape == x.shape

    def test_blurred_differs_from_input(self):
        t = GaussianBlur(kernel_size=5, sigma=2.0, dim=2)
        x = torch.rand(16, 16)
        out = t(x)
        assert not torch.allclose(out, x)

    def test_constant_input_unchanged(self):
        """Blurring a constant field should return the same constant (approximately)."""
        t = GaussianBlur(kernel_size=3, sigma=1.0, dim=2)
        x = torch.ones(16, 16)
        out = t(x)
        assert torch.allclose(out, x, atol=1e-4)

    def test_even_kernel_raises(self):
        with pytest.raises(AssertionError):
            GaussianBlur(kernel_size=4, dim=2)

    def test_invalid_dim_raises(self):
        with pytest.raises(AssertionError):
            GaussianBlur(dim=1)


# ---------------------------------------------------------------------------
# Integration: transforms compose with torchvision
# ---------------------------------------------------------------------------


class TestTransformComposition:
    def test_nan_to_num_in_compose(self):
        import torchvision.transforms.v2 as T

        pipeline = T.Compose(
            [
                NaNtoNum({"nan": 0, "posinf": None, "neginf": None}),
            ]
        )
        x = torch.tensor([float("nan"), 1.0])
        out = pipeline(x)
        assert out[0] == 0.0

    def test_binarize_after_dtype_conversion(self):
        import torchvision.transforms.v2 as T

        pipeline = T.Compose([T.ToDtype(torch.float), Binarize()])
        x = torch.tensor([0, 1, 2], dtype=torch.int32)
        out = pipeline(x)
        assert torch.allclose(out, torch.tensor([0.0, 1.0, 1.0]))

    def test_nan_preserved_through_binarize(self):
        """NaN labels must survive Binarize so loss can ignore them."""
        import torchvision.transforms.v2 as T

        pipeline = T.Compose([T.ToDtype(torch.float), Binarize()])
        x = torch.tensor([float("nan"), 1.0, 0.0])
        out = pipeline(x)
        assert torch.isnan(out[0])
        assert out[1] == 1.0
        assert out[2] == 0.0
