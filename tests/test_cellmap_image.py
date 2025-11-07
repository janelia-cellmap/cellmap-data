"""
Tests for CellMapImage class.

Tests image loading, spatial transformations, and value transformations
using real Zarr data without mocks.
"""

import pytest
import torch
import numpy as np
from pathlib import Path

from cellmap_data import CellMapImage
from .test_helpers import create_test_zarr_array, create_test_image_data


class TestCellMapImage:
    """Test suite for CellMapImage class."""
    
    @pytest.fixture
    def test_zarr_image(self, tmp_path):
        """Create a test Zarr image."""
        data = create_test_image_data((32, 32, 32), pattern="gradient")
        path = tmp_path / "test_image.zarr"
        create_test_zarr_array(path, data, scale=(4.0, 4.0, 4.0))
        return str(path), data
    
    def test_initialization(self, test_zarr_image):
        """Test basic initialization of CellMapImage."""
        path, _ = test_zarr_image
        
        image = CellMapImage(
            path=path,
            target_class="test_class",
            target_scale=(4.0, 4.0, 4.0),
            target_voxel_shape=(16, 16, 16),
            axis_order="zyx",
        )
        
        assert image.path == path
        assert image.label_class == "test_class"
        assert image.scale == {"z": 4.0, "y": 4.0, "x": 4.0}
        assert image.output_shape == {"z": 16, "y": 16, "x": 16}
        assert image.axes == "zyx"
    
    def test_device_selection(self, test_zarr_image):
        """Test device selection logic."""
        path, _ = test_zarr_image
        
        # Test explicit device
        image = CellMapImage(
            path=path,
            target_class="test",
            target_scale=(4.0, 4.0, 4.0),
            target_voxel_shape=(8, 8, 8),
            device="cpu",
        )
        assert image.device == "cpu"
        
        # Test automatic device selection
        image = CellMapImage(
            path=path,
            target_class="test",
            target_scale=(4.0, 4.0, 4.0),
            target_voxel_shape=(8, 8, 8),
        )
        # Should select cuda if available, otherwise mps, otherwise cpu
        assert image.device in ["cuda", "mps", "cpu"]
    
    def test_scale_and_shape_mismatch(self, test_zarr_image):
        """Test handling of mismatched axis order, scale, and shape."""
        path, _ = test_zarr_image
        
        # Test with more axes in axis_order than in scale
        image = CellMapImage(
            path=path,
            target_class="test",
            target_scale=(4.0, 4.0),
            target_voxel_shape=(8, 8),
            axis_order="zyx",
        )
        # Should pad scale with first value
        assert image.scale == {"z": 4.0, "y": 4.0, "x": 4.0}
        
        # Test with more axes in axis_order than in voxel_shape
        image = CellMapImage(
            path=path,
            target_class="test",
            target_scale=(4.0, 4.0, 4.0),
            target_voxel_shape=(8, 8),
            axis_order="zyx",
        )
        # Should pad voxel_shape with 1s
        assert image.output_shape == {"z": 1, "y": 8, "x": 8}
    
    def test_output_size_calculation(self, test_zarr_image):
        """Test that output size is correctly calculated."""
        path, _ = test_zarr_image
        
        image = CellMapImage(
            path=path,
            target_class="test",
            target_scale=(8.0, 8.0, 8.0),
            target_voxel_shape=(16, 16, 16),
        )
        
        # Output size should be voxel_shape * scale
        expected_size = {"z": 128.0, "y": 128.0, "x": 128.0}
        assert image.output_size == expected_size
    
    def test_value_transform(self, test_zarr_image):
        """Test value transform application."""
        path, _ = test_zarr_image
        
        # Create a simple transform that multiplies by 2
        def multiply_by_2(x):
            return x * 2
        
        image = CellMapImage(
            path=path,
            target_class="test",
            target_scale=(4.0, 4.0, 4.0),
            target_voxel_shape=(8, 8, 8),
            value_transform=multiply_by_2,
        )
        
        assert image.value_transform is not None
        # Test the transform works
        test_tensor = torch.tensor([1.0, 2.0, 3.0])
        result = image.value_transform(test_tensor)
        expected = torch.tensor([2.0, 4.0, 6.0])
        assert torch.allclose(result, expected)
    
    def test_2d_image(self, tmp_path):
        """Test handling of 2D images."""
        # Create a 2D image
        data = create_test_image_data((32, 32), pattern="checkerboard")
        path = tmp_path / "test_2d.zarr"
        create_test_zarr_array(path, data, axes=("y", "x"), scale=(4.0, 4.0))
        
        image = CellMapImage(
            path=str(path),
            target_class="test_2d",
            target_scale=(4.0, 4.0),
            target_voxel_shape=(16, 16),
            axis_order="yx",
        )
        
        assert image.axes == "yx"
        assert image.scale == {"y": 4.0, "x": 4.0}
    
    def test_pad_parameter(self, test_zarr_image):
        """Test pad parameter."""
        path, _ = test_zarr_image
        
        image_with_pad = CellMapImage(
            path=path,
            target_class="test",
            target_scale=(4.0, 4.0, 4.0),
            target_voxel_shape=(8, 8, 8),
            pad=True,
        )
        assert image_with_pad.pad is True
        
        image_without_pad = CellMapImage(
            path=path,
            target_class="test",
            target_scale=(4.0, 4.0, 4.0),
            target_voxel_shape=(8, 8, 8),
            pad=False,
        )
        assert image_without_pad.pad is False
    
    def test_pad_value(self, test_zarr_image):
        """Test pad value parameter."""
        path, _ = test_zarr_image
        
        # Test with NaN pad value
        image = CellMapImage(
            path=path,
            target_class="test",
            target_scale=(4.0, 4.0, 4.0),
            target_voxel_shape=(8, 8, 8),
            pad=True,
            pad_value=np.nan,
        )
        assert np.isnan(image.pad_value)
        
        # Test with numeric pad value
        image = CellMapImage(
            path=path,
            target_class="test",
            target_scale=(4.0, 4.0, 4.0),
            target_voxel_shape=(8, 8, 8),
            pad=True,
            pad_value=0.0,
        )
        assert image.pad_value == 0.0
    
    def test_interpolation_modes(self, test_zarr_image):
        """Test different interpolation modes."""
        path, _ = test_zarr_image
        
        for interp in ["nearest", "linear"]:
            image = CellMapImage(
                path=path,
                target_class="test",
                target_scale=(4.0, 4.0, 4.0),
                target_voxel_shape=(8, 8, 8),
                interpolation=interp,
            )
            assert image.interpolation == interp
    
    def test_different_axis_orders(self, tmp_path):
        """Test different axis orderings."""
        for axis_order in ["xyz", "zyx", "yxz"]:
            data = create_test_image_data((16, 16, 16), pattern="random")
            path = tmp_path / f"test_{axis_order}.zarr"
            create_test_zarr_array(
                path, data, axes=tuple(axis_order), scale=(4.0, 4.0, 4.0)
            )
            
            image = CellMapImage(
                path=str(path),
                target_class="test",
                target_scale=(4.0, 4.0, 4.0),
                target_voxel_shape=(8, 8, 8),
                axis_order=axis_order,
            )
            assert image.axes == axis_order
            assert len(image.scale) == 3
    
    def test_different_dtypes(self, tmp_path):
        """Test handling of different data types."""
        dtypes = [np.float32, np.float64, np.uint8, np.uint16, np.int32]
        
        for dtype in dtypes:
            data = create_test_image_data((16, 16, 16), dtype=dtype, pattern="constant")
            path = tmp_path / f"test_{dtype.__name__}.zarr"
            create_test_zarr_array(path, data, scale=(4.0, 4.0, 4.0))
            
            image = CellMapImage(
                path=str(path),
                target_class="test",
                target_scale=(4.0, 4.0, 4.0),
                target_voxel_shape=(8, 8, 8),
            )
            # Image should be created successfully
            assert image.path == str(path)
    
    def test_context_parameter(self, test_zarr_image):
        """Test TensorStore context parameter."""
        import tensorstore as ts
        
        path, _ = test_zarr_image
        
        # Create a custom context
        context = ts.Context()
        
        image = CellMapImage(
            path=path,
            target_class="test",
            target_scale=(4.0, 4.0, 4.0),
            target_voxel_shape=(8, 8, 8),
            context=context,
        )
        
        assert image.context is context
    
    def test_without_context(self, test_zarr_image):
        """Test that image works without explicit context."""
        path, _ = test_zarr_image
        
        image = CellMapImage(
            path=path,
            target_class="test",
            target_scale=(4.0, 4.0, 4.0),
            target_voxel_shape=(8, 8, 8),
            context=None,
        )
        
        assert image.context is None
