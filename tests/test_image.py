"""Tests for CellMapImage."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cellmap_data import CellMapImage

from .test_helpers import create_test_zarr


class TestCellMapImageBasics:
    def test_init(self, tmp_path):
        path = create_test_zarr(tmp_path, shape=(20, 20, 20))
        img = CellMapImage(
            path=path,
            target_class="raw",
            target_scale=[8.0, 8.0, 8.0],
            target_voxel_shape=[4, 4, 4],
        )
        assert img.label_class == "raw"
        assert img.axes == ["z", "y", "x"]

    def test_bounding_box(self, tmp_path):
        path = create_test_zarr(
            tmp_path, shape=(20, 20, 20), voxel_size=[8.0, 8.0, 8.0]
        )
        img = CellMapImage(path, "raw", [8.0, 8.0, 8.0], [4, 4, 4])
        bb = img.bounding_box
        assert set(bb.keys()) == {"z", "y", "x"}
        # 20 voxels * 8 nm = 160 nm
        assert bb["z"] == pytest.approx((0.0, 160.0))
        assert bb["x"] == pytest.approx((0.0, 160.0))

    def test_sampling_box(self, tmp_path):
        path = create_test_zarr(tmp_path, shape=(20, 20, 20))
        img = CellMapImage(path, "raw", [8.0, 8.0, 8.0], [4, 4, 4])
        sb = img.sampling_box
        # output is 4*8=32 nm → half=16 nm shrink on each side
        assert sb["z"][0] == pytest.approx(16.0)
        assert sb["z"][1] == pytest.approx(144.0)

    def test_sampling_box_none_when_too_small_no_pad(self, tmp_path):
        """Array smaller than output patch with pad=False → sampling_box is None."""
        path = create_test_zarr(tmp_path, shape=(10, 10, 10))
        img = CellMapImage(path, "raw", [8.0, 8.0, 8.0], [100, 100, 100], pad=False)
        assert img.sampling_box is None

    def test_sampling_box_single_centre_when_too_small_with_pad(self, tmp_path):
        """Array smaller than output patch with pad=True → single-centre sampling_box."""
        # 10 voxels * 8nm = 80nm array, output 100 voxels * 8nm = 800nm patch
        path = create_test_zarr(tmp_path, shape=(10, 10, 10))
        img = CellMapImage(path, "raw", [8.0, 8.0, 8.0], [100, 100, 100], pad=True)
        sb = img.sampling_box
        assert sb is not None
        # Single centre: box width == scale == 8nm
        for ax in img.axes:
            assert sb[ax][1] - sb[ax][0] == pytest.approx(8.0)
        # Centre of bounding box is midpoint of [0, 80] = 40nm
        # → sampling_box centre = 40nm → lo = 40 - 4 = 36, hi = 40 + 4 = 44
        assert sb["z"][0] == pytest.approx(36.0)
        assert sb["z"][1] == pytest.approx(44.0)

    def test_sampling_box_single_centre_yields_len_one(self, tmp_path):
        """get_center(0) for a single-centre image returns the bounding box midpoint."""
        path = create_test_zarr(tmp_path, shape=(10, 10, 10))
        img = CellMapImage(path, "raw", [8.0, 8.0, 8.0], [100, 100, 100], pad=True)
        center = img.get_center(0)
        # bounding_box midpoint is 40nm in each axis
        for ax in img.axes:
            assert center[ax] == pytest.approx(40.0)

    def test_small_crop_read_shape_and_nan(self, tmp_path):
        """Reading a small crop (pad=True) returns the full output shape with NaN padding."""
        path = create_test_zarr(tmp_path, shape=(10, 10, 10))
        img = CellMapImage(path, "raw", [8.0, 8.0, 8.0], [100, 100, 100], pad=True)
        center = img.get_center(0)
        patch = img[center]
        assert patch.shape == torch.Size([100, 100, 100])
        # 10*10*10 = 1000 valid voxels; rest are NaN
        valid = (~torch.isnan(patch)).sum().item()
        assert valid == 1000

    def test_scale_level_best_match(self, tmp_path):
        path = create_test_zarr(tmp_path, voxel_size=[8.0, 8.0, 8.0])
        img = CellMapImage(path, "raw", [8.0, 8.0, 8.0], [4, 4, 4])
        assert img.scale_level == 0

    def test_getitem_returns_tensor(self, tmp_path):
        path = create_test_zarr(tmp_path, shape=(20, 20, 20))
        img = CellMapImage(path, "raw", [8.0, 8.0, 8.0], [4, 4, 4])
        center = {"z": 80.0, "y": 80.0, "x": 80.0}
        patch = img[center]
        assert isinstance(patch, torch.Tensor)
        assert patch.shape == torch.Size([4, 4, 4])

    def test_getitem_shape_correct(self, tmp_path):
        path = create_test_zarr(tmp_path, shape=(40, 40, 40))
        img = CellMapImage(path, "raw", [8.0, 8.0, 8.0], [8, 8, 8])
        center = {"z": 160.0, "y": 160.0, "x": 160.0}
        patch = img[center]
        assert patch.shape == torch.Size([8, 8, 8])

    def test_padding_with_nan(self, tmp_path):
        """Reading near edge with pad=True → NaN in OOB regions."""
        path = create_test_zarr(tmp_path, shape=(8, 8, 8))
        img = CellMapImage(
            path, "raw", [8.0, 8.0, 8.0], [4, 4, 4], pad=True, pad_value=float("nan")
        )
        # Center near corner: some region will be outside bounds
        center = {"z": 4.0, "y": 4.0, "x": 4.0}  # origin + 0.5 voxel
        patch = img[center]
        assert patch.shape == torch.Size([4, 4, 4])
        # Should have some NaN in the padded region
        assert torch.isnan(patch).any()

    def test_partial_oob_left_correct_shape(self, tmp_path):
        """Partial OOB on the left: output shape must equal target, left region is NaN."""
        path = create_test_zarr(tmp_path, shape=(8, 8, 8))
        img = CellMapImage(path, "raw", [8.0, 8.0, 8.0], [4, 4, 4], pad=True)
        # Centre near edge: 4nm = 0.5 voxel, so half the patch extends before origin
        center = {"z": 4.0, "y": 32.0, "x": 32.0}
        patch = img[center]
        assert patch.shape == torch.Size([4, 4, 4])
        # z-slices before origin should be NaN; interior slices should not be all-NaN
        assert torch.isnan(patch[0]).all()  # first z slice is OOB
        assert not torch.isnan(patch[-1]).all()  # last z slice is in-bounds

    def test_partial_oob_right_correct_shape(self, tmp_path):
        """Partial OOB on the right: output shape must equal target, right region is NaN."""
        path = create_test_zarr(tmp_path, shape=(8, 8, 8))
        img = CellMapImage(path, "raw", [8.0, 8.0, 8.0], [4, 4, 4], pad=True)
        # 8 voxels * 8nm = 64nm; centre at 60nm = 7.5th voxel
        center = {"z": 60.0, "y": 32.0, "x": 32.0}
        patch = img[center]
        assert patch.shape == torch.Size([4, 4, 4])
        assert not torch.isnan(patch[0]).all()  # first z slice is in-bounds
        assert torch.isnan(patch[-1]).all()  # last z slice is OOB

    def test_fully_oob_returns_all_nan_with_warning(self, tmp_path, caplog):
        """Fully OOB read returns all-pad_value tensor and emits a logger warning."""
        import logging

        path = create_test_zarr(tmp_path, shape=(8, 8, 8))
        img = CellMapImage(path, "raw", [8.0, 8.0, 8.0], [4, 4, 4], pad=True)
        # Centre far outside bounding_box [0, 64] nm
        center = {"z": 10000.0, "y": 32.0, "x": 32.0}
        with caplog.at_level(logging.WARNING, logger="cellmap_data.image"):
            patch = img[center]
        assert patch.shape == torch.Size([4, 4, 4])
        assert torch.isnan(patch).all()
        assert any("out-of-bounds" in msg for msg in caplog.messages)

    def test_no_padding_within_sampling_box(self, tmp_path):
        """Reading a centre within sampling_box with pad=False → no NaN, shape == output."""
        path = create_test_zarr(tmp_path, shape=(8, 8, 8))
        img = CellMapImage(path, "raw", [8.0, 8.0, 8.0], [4, 4, 4], pad=False)
        # sampling_box is [16, 48] nm; use centre of the array (32 nm)
        center = {"z": 32.0, "y": 32.0, "x": 32.0}
        patch = img[center]
        assert patch.shape == torch.Size([4, 4, 4])
        assert not torch.isnan(patch).any()

    def test_get_center(self, tmp_path):
        path = create_test_zarr(tmp_path, shape=(20, 20, 20))
        img = CellMapImage(path, "raw", [8.0, 8.0, 8.0], [4, 4, 4])
        center = img.get_center(0)
        assert set(center.keys()) == {"z", "y", "x"}
        # First centre should be at sampling_box lower bound + 0.5*scale
        sb = img.sampling_box
        assert center["z"] == pytest.approx(sb["z"][0] + 0.5 * 8.0)

    def test_value_transform_applied(self, tmp_path):
        """A value_transform that negates values should change the output."""
        path = create_test_zarr(tmp_path, shape=(20, 20, 20))
        img = CellMapImage(
            path,
            "raw",
            [8.0, 8.0, 8.0],
            [4, 4, 4],
            value_transform=lambda x: x * -1.0,
        )
        img_plain = CellMapImage(path, "raw", [8.0, 8.0, 8.0], [4, 4, 4])
        center = {"z": 80.0, "y": 80.0, "x": 80.0}
        assert torch.allclose(img[center], -img_plain[center])

    def test_repr(self, tmp_path):
        path = create_test_zarr(tmp_path)
        img = CellMapImage(path, "raw", [8.0, 8.0, 8.0], [4, 4, 4])
        r = repr(img)
        assert "CellMapImage" in r
        assert "raw" in r


class TestCellMapImageSpatialTransforms:
    def test_mirror_z(self, tmp_path):
        data = np.arange(8 * 8 * 8, dtype=np.float32).reshape(8, 8, 8)
        path = create_test_zarr(tmp_path, shape=(8, 8, 8), data=data)
        img = CellMapImage(path, "raw", [8.0, 8.0, 8.0], [4, 4, 4], pad=True)
        center = {"z": 32.0, "y": 32.0, "x": 32.0}
        img.set_spatial_transforms(None)
        patch_orig = img[center].clone()
        img.set_spatial_transforms({"mirror": {"z": True, "y": False, "x": False}})
        patch_mirrored = img[center].clone()
        img.set_spatial_transforms(None)
        assert not torch.allclose(patch_orig, patch_mirrored)
        # Mirroring z twice should give back original
        assert torch.allclose(patch_orig, patch_mirrored.flip(0))

    def test_set_spatial_transforms_none(self, tmp_path):
        path = create_test_zarr(tmp_path, shape=(20, 20, 20))
        img = CellMapImage(path, "raw", [8.0, 8.0, 8.0], [4, 4, 4])
        img.set_spatial_transforms(None)
        assert img._current_spatial_transforms is None

    def test_rotation_read_shape_larger(self, tmp_path):
        """With rotation, read_shape should be larger than output_shape."""
        path = create_test_zarr(tmp_path, shape=(40, 40, 40))
        img = CellMapImage(path, "raw", [8.0, 8.0, 8.0], [8, 8, 8], pad=True)
        # 45° rotation about z
        theta = np.deg2rad(45)
        R = np.eye(3)
        R[1, 1] = np.cos(theta)
        R[1, 2] = -np.sin(theta)
        R[2, 1] = np.sin(theta)
        R[2, 2] = np.cos(theta)
        img.set_spatial_transforms({"rotation_matrix": R})
        read_shape = img._compute_read_shape()
        # y and x dims should be larger (by cos+sin ≈ 1.41)
        assert read_shape[1] > 8
        assert read_shape[2] > 8
        img.set_spatial_transforms(None)

    def test_rotation_output_shape_preserved(self, tmp_path):
        """After rotation+crop, output shape must equal target_voxel_shape."""
        path = create_test_zarr(tmp_path, shape=(60, 60, 60))
        img = CellMapImage(path, "raw", [8.0, 8.0, 8.0], [8, 8, 8], pad=True)
        theta = np.deg2rad(30)
        R = np.eye(3)
        R[1, 1] = np.cos(theta)
        R[1, 2] = -np.sin(theta)
        R[2, 1] = np.sin(theta)
        R[2, 2] = np.cos(theta)
        center = {"z": 240.0, "y": 240.0, "x": 240.0}
        img.set_spatial_transforms({"rotation_matrix": R})
        patch = img[center]
        img.set_spatial_transforms(None)
        assert patch.shape == torch.Size([8, 8, 8])


class TestCellMapImageClassCounts:
    def test_class_counts_keys(self, tmp_path):
        import zarr as z

        data = np.zeros((10, 10, 10), dtype=np.uint8)
        data[2:5, 2:5, 2:5] = 1  # some foreground
        path = create_test_zarr(tmp_path, shape=(10, 10, 10), data=data)
        img = CellMapImage(path, "mito", [8.0, 8.0, 8.0], [4, 4, 4])
        counts = img.class_counts
        assert "mito" in counts
        assert counts["mito"] >= 0

    def test_total_voxels_equals_array_size(self, tmp_path):
        shape = (10, 10, 10)
        data = np.zeros(shape, dtype=np.uint8)
        data[2:5, 2:5, 2:5] = 1
        path = create_test_zarr(tmp_path, shape=shape, data=data)
        img = CellMapImage(path, "mito", [8.0, 8.0, 8.0], [4, 4, 4])
        assert img.total_voxels == int(np.prod(shape))
