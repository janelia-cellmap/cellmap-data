"""Tests for EmptyImage."""

from __future__ import annotations

import torch

from cellmap_data.empty_image import EmptyImage


def test_empty_image_returns_nan():
    img = EmptyImage("fake/path", "mito", [8.0, 8.0, 8.0], [4, 4, 4])
    patch = img[{"z": 0.0, "y": 0.0, "x": 0.0}]
    assert isinstance(patch, torch.Tensor)
    assert patch.shape == torch.Size([4, 4, 4])
    assert torch.isnan(patch).all()


def test_empty_image_bounding_box_none():
    img = EmptyImage("fake/path", "er", [8.0, 8.0, 8.0], [4, 4, 4])
    assert img.bounding_box is None
    assert img.sampling_box is None


def test_empty_image_class_counts_zero():
    img = EmptyImage("fake/path", "nucleus", [8.0, 8.0, 8.0], [4, 4, 4])
    assert img.class_counts == {"nucleus": 0}


def test_empty_image_set_spatial_transforms_noop():
    img = EmptyImage("fake/path", "mito", [8.0, 8.0, 8.0], [4, 4, 4])
    img.set_spatial_transforms({"mirror": {"z": True}})  # should not raise
    patch = img[{"z": 0.0, "y": 0.0, "x": 0.0}]
    assert torch.isnan(patch).all()


def test_empty_image_repr():
    img = EmptyImage("fake/path", "mito", [8.0, 8.0, 8.0], [4, 4, 4])
    r = repr(img)
    assert "EmptyImage" in r
    assert "mito" in r


def test_empty_image_clone():
    """Each call returns a fresh clone (not the same tensor)."""
    img = EmptyImage("fake/path", "mito", [8.0, 8.0, 8.0], [4, 4, 4])
    p1 = img[{"z": 0.0, "y": 0.0, "x": 0.0}]
    p2 = img[{"z": 0.0, "y": 0.0, "x": 0.0}]
    assert p1 is not p2


def test_empty_image_2d_scale_has_all_axes():
    """Regression: 2D scale/shape with default axis_order='zyx' must produce a
    scale dict covering all three axes (z, y, x), not just two."""
    img = EmptyImage("fake/path", "mito", [8.0, 8.0], [4, 4])
    assert set(img.scale.keys()) == {"z", "y", "x"}
    assert set(img.axes) == {"z", "y", "x"}
