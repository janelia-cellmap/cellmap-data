"""Tests for CellMapImage edge cases and special methods."""

import numpy as np
import pytest
import torch

from cellmap_data import CellMapImage

from .test_helpers import create_test_image_data, create_test_zarr_array


class TestCellMapImageEdgeCases:
    """Test edge cases and special methods in CellMapImage."""

    @pytest.fixture
    def test_zarr_image(self, tmp_path):
        """Create a test Zarr image."""
        data = create_test_image_data((32, 32, 32), pattern="gradient")
        path = tmp_path / "test_image.zarr"
        create_test_zarr_array(path, data, scale=(4.0, 4.0, 4.0))
        return str(path), data

    def test_axis_order_longer_than_scale(self, test_zarr_image):
        """Test handling when axis_order has more axes than target_scale."""
        path, _ = test_zarr_image

        # Provide fewer scale values than axes
        image = CellMapImage(
            path=path,
            target_class="test_class",
            target_scale=(4.0, 4.0),  # Only 2 values for 3 axes
            target_voxel_shape=(16, 16, 16),
            axis_order="zyx",  # 3 axes
        )

        # Should pad scale with first value
        assert len(image.scale) == 3
        assert image.scale["z"] == 4.0  # Padded value
        assert image.scale["y"] == 4.0
        assert image.scale["x"] == 4.0

    def test_axis_order_longer_than_shape(self, test_zarr_image):
        """Test handling when axis_order has more axes than target_voxel_shape."""
        path, _ = test_zarr_image

        # Provide fewer shape values than axes
        image = CellMapImage(
            path=path,
            target_class="test_class",
            target_scale=(4.0, 4.0, 4.0),
            target_voxel_shape=(16, 16),  # Only 2 values for 3 axes
            axis_order="zyx",  # 3 axes
        )

        # Should pad shape with 1s
        assert len(image.output_shape) == 3
        assert image.output_shape["z"] == 1  # Padded value
        assert image.output_shape["y"] == 16
        assert image.output_shape["x"] == 16

    def test_device_auto_selection_cuda(self, test_zarr_image):
        """Test device auto-selection when no device specified."""
        path, _ = test_zarr_image

        image = CellMapImage(
            path=path,
            target_class="test_class",
            target_scale=(4.0, 4.0, 4.0),
            target_voxel_shape=(16, 16, 16),
        )

        # Should select an appropriate device
        assert image.device in ["cuda", "mps", "cpu"]

    def test_explicit_device_selection(self, test_zarr_image):
        """Test explicit device selection."""
        path, _ = test_zarr_image

        image = CellMapImage(
            path=path,
            target_class="test_class",
            target_scale=(4.0, 4.0, 4.0),
            target_voxel_shape=(16, 16, 16),
            device="cpu",
        )

        assert image.device == "cpu"

    def test_to_device_method(self, test_zarr_image):
        """Test moving image to different device."""
        path, _ = test_zarr_image

        image = CellMapImage(
            path=path,
            target_class="test_class",
            target_scale=(4.0, 4.0, 4.0),
            target_voxel_shape=(16, 16, 16),
        )

        # Move to CPU
        image.to("cpu")
        assert image.device == "cpu"

    def test_set_spatial_transforms_none(self, test_zarr_image):
        """Test setting spatial transforms to None."""
        path, _ = test_zarr_image

        image = CellMapImage(
            path=path,
            target_class="test_class",
            target_scale=(4.0, 4.0, 4.0),
            target_voxel_shape=(16, 16, 16),
        )

        # Set to None
        image.set_spatial_transforms(None)
        assert image._current_spatial_transforms is None

    def test_set_spatial_transforms_with_values(self, test_zarr_image):
        """Test setting spatial transforms with actual transform dict."""
        path, _ = test_zarr_image

        image = CellMapImage(
            path=path,
            target_class="test_class",
            target_scale=(4.0, 4.0, 4.0),
            target_voxel_shape=(16, 16, 16),
        )

        # Set transforms
        transforms = {"mirror": {"axes": {"x": 0.5}}}
        image.set_spatial_transforms(transforms)
        assert image._current_spatial_transforms == transforms

    def test_bounding_box_property(self, test_zarr_image):
        """Test the bounding_box property."""
        path, _ = test_zarr_image

        image = CellMapImage(
            path=path,
            target_class="test_class",
            target_scale=(4.0, 4.0, 4.0),
            target_voxel_shape=(16, 16, 16),
        )

        bbox = image.bounding_box

        # Should be a dict with axis keys
        assert isinstance(bbox, dict)
        for axis in "zyx":
            assert axis in bbox
            assert len(bbox[axis]) == 2
            assert bbox[axis][0] <= bbox[axis][1]

    def test_sampling_box_property(self, test_zarr_image):
        """Test the sampling_box property."""
        path, _ = test_zarr_image

        image = CellMapImage(
            path=path,
            target_class="test_class",
            target_scale=(4.0, 4.0, 4.0),
            target_voxel_shape=(16, 16, 16),
        )

        sbox = image.sampling_box

        # Should be a dict with axis keys
        assert isinstance(sbox, dict)
        for axis in "zyx":
            assert axis in sbox
            assert len(sbox[axis]) == 2

    def test_class_counts_property(self, test_zarr_image):
        """Test the class_counts property."""
        path, _ = test_zarr_image

        image = CellMapImage(
            path=path,
            target_class="test_class",
            target_scale=(4.0, 4.0, 4.0),
            target_voxel_shape=(16, 16, 16),
        )

        counts = image.class_counts

        # Should be a numeric value or dict
        assert isinstance(counts, (int, float, dict))

    def test_pad_parameter_true(self, test_zarr_image):
        """Test padding when pad=True."""
        path, _ = test_zarr_image

        image = CellMapImage(
            path=path,
            target_class="test_class",
            target_scale=(4.0, 4.0, 4.0),
            target_voxel_shape=(16, 16, 16),
            pad=True,
            pad_value=0,
        )

        assert image.pad is True
        assert image.pad_value == 0

    def test_pad_parameter_false(self, test_zarr_image):
        """Test when pad=False."""
        path, _ = test_zarr_image

        image = CellMapImage(
            path=path,
            target_class="test_class",
            target_scale=(4.0, 4.0, 4.0),
            target_voxel_shape=(16, 16, 16),
            pad=False,
        )

        assert image.pad is False

    def test_interpolation_nearest(self, test_zarr_image):
        """Test interpolation mode nearest."""
        path, _ = test_zarr_image

        image = CellMapImage(
            path=path,
            target_class="test_class",
            target_scale=(4.0, 4.0, 4.0),
            target_voxel_shape=(16, 16, 16),
            interpolation="nearest",
        )

        assert image.interpolation == "nearest"

    def test_interpolation_linear(self, test_zarr_image):
        """Test interpolation mode linear."""
        path, _ = test_zarr_image

        image = CellMapImage(
            path=path,
            target_class="test_class",
            target_scale=(4.0, 4.0, 4.0),
            target_voxel_shape=(16, 16, 16),
            interpolation="linear",
        )

        assert image.interpolation == "linear"

    def test_value_transform_none(self, test_zarr_image):
        """Test when no value transform is provided."""
        path, _ = test_zarr_image

        image = CellMapImage(
            path=path,
            target_class="test_class",
            target_scale=(4.0, 4.0, 4.0),
            target_voxel_shape=(16, 16, 16),
            value_transform=None,
        )

        assert image.value_transform is None

    def test_value_transform_provided(self, test_zarr_image):
        """Test when value transform is provided."""
        path, _ = test_zarr_image

        transform = lambda x: x * 2
        image = CellMapImage(
            path=path,
            target_class="test_class",
            target_scale=(4.0, 4.0, 4.0),
            target_voxel_shape=(16, 16, 16),
            value_transform=transform,
        )

        assert image.value_transform is transform

    def test_output_size_calculation(self, test_zarr_image):
        """Test that output_size is correctly calculated."""
        path, _ = test_zarr_image

        image = CellMapImage(
            path=path,
            target_class="test_class",
            target_scale=(4.0, 8.0, 2.0),
            target_voxel_shape=(10, 20, 30),
            axis_order="zyx",
        )

        # output_size = voxel_shape * scale
        assert image.output_size["z"] == 10 * 4.0
        assert image.output_size["y"] == 20 * 8.0
        assert image.output_size["x"] == 30 * 2.0

    def test_axes_property(self, test_zarr_image):
        """Test that axes property is correctly set."""
        path, _ = test_zarr_image

        image = CellMapImage(
            path=path,
            target_class="test_class",
            target_scale=(4.0, 4.0, 4.0),
            target_voxel_shape=(16, 16, 16),
            axis_order="zyx",
        )

        assert image.axes == "zyx"

    def test_context_parameter_none(self, test_zarr_image):
        """Test when no context is provided."""
        path, _ = test_zarr_image

        image = CellMapImage(
            path=path,
            target_class="test_class",
            target_scale=(4.0, 4.0, 4.0),
            target_voxel_shape=(16, 16, 16),
            context=None,
        )

        assert image.context is None

    def test_path_attribute(self, test_zarr_image):
        """Test that path attribute is correctly set."""
        path, _ = test_zarr_image

        image = CellMapImage(
            path=path,
            target_class="test_class",
            target_scale=(4.0, 4.0, 4.0),
            target_voxel_shape=(16, 16, 16),
        )

        assert image.path == path

    def test_label_class_attribute(self, test_zarr_image):
        """Test that label_class attribute is correctly set."""
        path, _ = test_zarr_image

        image = CellMapImage(
            path=path,
            target_class="my_class",
            target_scale=(4.0, 4.0, 4.0),
            target_voxel_shape=(16, 16, 16),
        )

        assert image.label_class == "my_class"

    def test_getitem_returns_tensor(self, test_zarr_image):
        """Test that __getitem__ returns a PyTorch tensor."""
        path, _ = test_zarr_image

        image = CellMapImage(
            path=path,
            target_class="test_class",
            target_scale=(4.0, 4.0, 4.0),
            target_voxel_shape=(8, 8, 8),
        )

        center = {"z": 64.0, "y": 64.0, "x": 64.0}
        result = image[center]

        assert isinstance(result, torch.Tensor)
        assert result.ndim >= 3

    def test_nan_pad_value(self, test_zarr_image):
        """Test using NaN as pad value."""
        path, _ = test_zarr_image

        image = CellMapImage(
            path=path,
            target_class="test_class",
            target_scale=(4.0, 4.0, 4.0),
            target_voxel_shape=(16, 16, 16),
            pad=True,
            pad_value=np.nan,
        )

        assert np.isnan(image.pad_value)

    # -----------------------------------------------------------------------
    # coord_offsets caching
    # -----------------------------------------------------------------------

    def test_coord_offsets_is_cached_property(self, test_zarr_image):
        """coord_offsets must use @cached_property, not a manual null-check pattern.

        Verifies: (a) the returned dict has the expected axes, (b) successive
        accesses return the exact same objects (cached, not recomputed), and
        (c) the offsets are symmetric around zero for each axis.
        """
        path, _ = test_zarr_image
        image = CellMapImage(
            path=path,
            target_class="test_class",
            target_scale=(4.0, 4.0, 4.0),
            target_voxel_shape=(8, 8, 8),
        )

        offsets1 = image.coord_offsets
        offsets2 = image.coord_offsets

        # Cached: same object returned on every access
        assert offsets1 is offsets2

        # Stored in __dict__ (cached_property, not regular property)
        assert "coord_offsets" in image.__dict__

        # Correct axes present
        for axis in image.axes:
            assert axis in offsets1
            arr = offsets1[axis]
            assert len(arr) == image.output_shape[axis]
            # Symmetric around zero within float tolerance
            assert abs(arr[0] + arr[-1]) < 1e-9

    def test_coord_offsets_not_cleared_by_array_cache_clear(self, test_zarr_image):
        """_clear_array_cache must only clear 'array', leaving coord_offsets intact."""
        path, _ = test_zarr_image
        image = CellMapImage(
            path=path,
            target_class="test_class",
            target_scale=(4.0, 4.0, 4.0),
            target_voxel_shape=(8, 8, 8),
        )

        offsets_before = image.coord_offsets  # populate cache
        assert "coord_offsets" in image.__dict__

        image._clear_array_cache()

        # coord_offsets must still be cached after cache clear
        assert "coord_offsets" in image.__dict__
        assert image.coord_offsets is offsets_before

    def test_coord_offsets_values_match_output_size_and_scale(self, test_zarr_image):
        """coord_offsets values must span exactly [-output_size/2+scale/2, output_size/2-scale/2]."""
        path, _ = test_zarr_image
        image = CellMapImage(
            path=path,
            target_class="test_class",
            target_scale=(4.0, 4.0, 4.0),
            target_voxel_shape=(8, 8, 8),
        )

        for axis in image.axes:
            arr = image.coord_offsets[axis]
            expected_lo = -image.output_size[axis] / 2 + image.scale[axis] / 2
            expected_hi = image.output_size[axis] / 2 - image.scale[axis] / 2
            assert abs(arr[0] - expected_lo) < 1e-9
            assert abs(arr[-1] - expected_hi) < 1e-9
