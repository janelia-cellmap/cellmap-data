"""
Tests for memory management in CellMapImage.

Specifically tests the array cache clearing mechanism to prevent memory leaks.
"""

import pytest
import numpy as np
from cellmap_data import CellMapImage
from .test_helpers import create_test_image_data, create_test_zarr_array


class TestMemoryManagement:
    """Test memory management features."""

    @pytest.fixture
    def test_zarr_image(self, tmp_path):
        """Create a test Zarr image."""
        data = create_test_image_data((32, 32, 32), pattern="gradient")
        path = tmp_path / "test_image.zarr"
        create_test_zarr_array(path, data, scale=(4.0, 4.0, 4.0))
        return str(path), data

    def test_array_cache_cleared_after_getitem(self, test_zarr_image):
        """Test that array cache is cleared after __getitem__ to prevent memory leaks."""
        path, _ = test_zarr_image

        image = CellMapImage(
            path=path,
            target_class="test",
            target_scale=(4.0, 4.0, 4.0),
            target_voxel_shape=(8, 8, 8),
            axis_order="zyx",
        )

        # Access array to populate cache
        _ = image.array
        assert "array" in image.__dict__, "Array should be cached after first access"

        # Call __getitem__ which should clear the cache
        center = {"z": 64.0, "y": 64.0, "x": 64.0}
        _ = image[center]

        # Check that cache was cleared
        assert (
            "array" not in image.__dict__
        ), "Array cache should be cleared after __getitem__"

    def test_array_cache_repopulates_after_clearing(self, test_zarr_image):
        """Test that array cache can be repopulated after being cleared."""
        path, _ = test_zarr_image

        image = CellMapImage(
            path=path,
            target_class="test",
            target_scale=(4.0, 4.0, 4.0),
            target_voxel_shape=(8, 8, 8),
            axis_order="zyx",
        )

        # First access
        center = {"z": 64.0, "y": 64.0, "x": 64.0}
        data1 = image[center]

        # Array cache should be cleared
        assert "array" not in image.__dict__

        # Second access - should work without errors (cache will be repopulated)
        data2 = image[center]

        # Both should produce valid tensors
        assert data1.shape == data2.shape
        assert data1.dtype == data2.dtype

    def test_clear_array_cache_method(self, test_zarr_image):
        """Test the _clear_array_cache method directly."""
        path, _ = test_zarr_image

        image = CellMapImage(
            path=path,
            target_class="test",
            target_scale=(4.0, 4.0, 4.0),
            target_voxel_shape=(8, 8, 8),
        )

        # Populate cache
        _ = image.array
        assert "array" in image.__dict__

        # Clear cache
        image._clear_array_cache()
        assert "array" not in image.__dict__

        # Clearing when not cached should not raise an error
        image._clear_array_cache()  # Should be a no-op

    def test_multiple_getitem_calls_clear_cache_each_time(self, test_zarr_image):
        """Test that cache is cleared on every __getitem__ call."""
        path, _ = test_zarr_image

        image = CellMapImage(
            path=path,
            target_class="test",
            target_scale=(4.0, 4.0, 4.0),
            target_voxel_shape=(8, 8, 8),
        )

        centers = [
            {"z": 48.0, "y": 48.0, "x": 48.0},
            {"z": 64.0, "y": 64.0, "x": 64.0},
            {"z": 80.0, "y": 80.0, "x": 80.0},
        ]

        for center in centers:
            _ = image[center]
            # Cache should be cleared after each call
            assert (
                "array" not in image.__dict__
            ), f"Array cache should be cleared after accessing center {center}"

    def test_cache_clearing_with_spatial_transforms(self, test_zarr_image):
        """Test that cache is cleared even with spatial transforms."""
        path, _ = test_zarr_image

        image = CellMapImage(
            path=path,
            target_class="test",
            target_scale=(4.0, 4.0, 4.0),
            target_voxel_shape=(8, 8, 8),
        )

        # Set spatial transforms
        image.set_spatial_transforms(
            {"mirror": {"x": True}, "rotate": {"z": 15}}
        )

        center = {"z": 64.0, "y": 64.0, "x": 64.0}
        _ = image[center]

        # Cache should still be cleared
        assert "array" not in image.__dict__

    def test_cache_clearing_with_value_transforms(self, test_zarr_image):
        """Test that cache is cleared when value transforms are applied."""
        path, _ = test_zarr_image

        def normalize(x):
            return x / 255.0

        image = CellMapImage(
            path=path,
            target_class="test",
            target_scale=(4.0, 4.0, 4.0),
            target_voxel_shape=(8, 8, 8),
            value_transform=normalize,
        )

        center = {"z": 64.0, "y": 64.0, "x": 64.0}
        _ = image[center]

        # Cache should be cleared
        assert "array" not in image.__dict__

    def test_simulated_training_loop_memory(self, test_zarr_image):
        """
        Simulate a training loop to verify cache is cleared on each iteration.
        
        This test simulates the memory leak scenario described in the issue:
        repeated calls to __getitem__ should not accumulate memory from cached arrays.
        """
        path, _ = test_zarr_image

        image = CellMapImage(
            path=path,
            target_class="test",
            target_scale=(4.0, 4.0, 4.0),
            target_voxel_shape=(8, 8, 8),
        )

        # Simulate multiple training iterations
        centers = [
            {"z": 48.0 + i * 4.0, "y": 48.0 + i * 4.0, "x": 48.0 + i * 4.0}
            for i in range(10)
        ]

        for i, center in enumerate(centers):
            _ = image[center]
            
            # After each iteration, array cache should be cleared
            assert (
                "array" not in image.__dict__
            ), f"Iteration {i}: Array cache should be cleared to prevent memory leak"

    def test_cache_clearing_with_interpolation(self, tmp_path):
        """
        Test cache clearing when interpolation is used (the main memory leak scenario).
        
        When coords require interpolation (not simple float/int), the array.interp()
        method creates intermediate arrays that could accumulate memory.
        """
        data = create_test_image_data((32, 32, 32), pattern="gradient")
        path = tmp_path / "test_interp.zarr"
        create_test_zarr_array(path, data, scale=(4.0, 4.0, 4.0))

        image = CellMapImage(
            path=str(path),
            target_class="test",
            target_scale=(4.0, 4.0, 4.0),
            target_voxel_shape=(8, 8, 8),
            interpolation="linear",  # Use linear interpolation to trigger interp()
        )

        # Use spatial transforms to trigger the interpolation code path
        image.set_spatial_transforms({"rotate": {"z": 15}})

        center = {"z": 64.0, "y": 64.0, "x": 64.0}
        _ = image[center]

        # Cache should be cleared even after interpolation
        assert "array" not in image.__dict__
