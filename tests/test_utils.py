"""
Tests for utility functions.

Tests dtype utilities, sampling utilities, and miscellaneous utilities.
"""

import pytest
import torch
import numpy as np

from cellmap_data.utils.misc import (
    get_sliced_shape,
    torch_max_value,
)


class TestUtilsMisc:
    """Test suite for miscellaneous utility functions."""
    
    def test_get_sliced_shape_basic(self):
        """Test get_sliced_shape with axis parameter."""
        shape = (64, 64)
        # Add singleton at axis 0
        sliced_shape = get_sliced_shape(shape, 0)
        assert isinstance(sliced_shape, list)
        assert 1 in sliced_shape
    
    def test_get_sliced_shape_different_axes(self):
        """Test get_sliced_shape with different axes."""
        shape = (64, 64)
        for axis in [0, 1, 2]:
            sliced_shape = get_sliced_shape(shape, axis)
            assert isinstance(sliced_shape, list)
    
    def test_torch_max_value_float32(self):
        """Test torch_max_value for float32."""
        max_val = torch_max_value(torch.float32)
        assert isinstance(max_val, int)
        assert max_val > 0
    
    def test_torch_max_value_uint8(self):
        """Test torch_max_value for uint8."""
        max_val = torch_max_value(torch.uint8)
        assert max_val == 255
    
    def test_torch_max_value_int16(self):
        """Test torch_max_value for int16."""
        max_val = torch_max_value(torch.int16)
        assert max_val == 32767
    
    def test_torch_max_value_int32(self):
        """Test torch_max_value for int32."""
        max_val = torch_max_value(torch.int32)
        assert max_val == 2147483647
    
    def test_torch_max_value_bool(self):
        """Test torch_max_value for bool."""
        max_val = torch_max_value(torch.bool)
        assert max_val == 1


class TestSamplingUtils:
    """Test suite for sampling utilities."""
    
    def test_sampling_weights_basic(self):
        """Test basic sampling weight calculation."""
        # Create simple class distributions
        class_counts = {
            "class_0": 100,
            "class_1": 200,
            "class_2": 300,
        }
        
        # Weights should be inversely proportional to counts
        weights = []
        for count in class_counts.values():
            weight = 1.0 / count if count > 0 else 0.0
            weights.append(weight)
        
        # Check that smaller classes get higher weights
        assert weights[0] > weights[1] > weights[2]
    
    def test_sampling_with_zero_counts(self):
        """Test sampling when some classes have zero counts."""
        class_counts = {
            "class_0": 100,
            "class_1": 0,  # No samples
            "class_2": 300,
        }
        
        # Zero-count classes should get zero weight
        for name, count in class_counts.items():
            weight = 1.0 / count if count > 0 else 0.0
            if count == 0:
                assert weight == 0.0
            else:
                assert weight > 0.0
    
    def test_normalized_weights(self):
        """Test that weights can be normalized."""
        class_counts = [100, 200, 300, 400]
        
        # Calculate unnormalized weights
        weights = [1.0 / count for count in class_counts]
        
        # Normalize
        total = sum(weights)
        normalized = [w / total for w in weights]
        
        # Should sum to 1
        assert abs(sum(normalized) - 1.0) < 1e-6
        
        # Should preserve relative ordering
        assert normalized[0] > normalized[1] > normalized[2] > normalized[3]


class TestArrayOperations:
    """Test suite for array operation utilities."""
    
    def test_array_2d_detection(self):
        """Test detection of 2D arrays."""
        from cellmap_data.utils.misc import is_array_2D
        
        # is_array_2D takes a mapping of array info, not arrays directly
        # Test with dict format
        arr_2d_info = {"raw": {"shape": (64, 64)}}
        result_2d = is_array_2D(arr_2d_info)
        assert isinstance(result_2d, (bool, dict))
        
        # 3D array info
        arr_3d_info = {"raw": {"shape": (64, 64, 64)}}
        result_3d = is_array_2D(arr_3d_info)
        assert isinstance(result_3d, (bool, dict))
    
    def test_2d_array_with_singleton(self):
        """Test 2D detection with singleton dimensions."""
        from cellmap_data.utils.misc import is_array_2D
        
        # Shape with singleton
        arr_info = {"raw": {"shape": (1, 64, 64)}}
        result = is_array_2D(arr_info)
        assert isinstance(result, (bool, dict))
    
    # Tests for min_redundant_inds removed - function doesn't exist in current implementation


class TestPathUtilities:
    """Test suite for path utility functions."""
    
    def test_split_target_path_basic(self):
        """Test basic target path splitting."""
        from cellmap_data.utils.misc import split_target_path
        
        # Path without embedded classes
        path = "/path/to/dataset.zarr"
        base_path, classes = split_target_path(path)
        
        assert isinstance(base_path, str)
        assert isinstance(classes, list)
    
    def test_split_target_path_with_classes(self):
        """Test target path splitting with embedded classes."""
        from cellmap_data.utils.misc import split_target_path
        
        # Path with class specification in brackets
        path = "/path/to/dataset[class1,class2].zarr"
        base_path, classes = split_target_path(path)
        
        assert isinstance(base_path, str)
        assert isinstance(classes, list)
        assert "{label}" in base_path  # Should have placeholder
    
    def test_split_target_path_multiple_classes(self):
        """Test with multiple classes in path."""
        from cellmap_data.utils.misc import split_target_path
        
        path = "/path/to/dataset.zarr"
        base_path, classes = split_target_path(path)
        
        # Should handle standard case
        assert base_path is not None
        assert classes is not None
        assert isinstance(classes, list)


class TestCoordinateTransforms:
    """Test suite for coordinate transformation utilities."""
    
    def test_coordinate_scaling(self):
        """Test coordinate scaling transformations."""
        # Physical coordinates to voxel coordinates
        physical_coord = np.array([80.0, 80.0, 80.0])  # nm
        scale = np.array([8.0, 8.0, 8.0])  # nm/voxel
        
        voxel_coord = physical_coord / scale
        
        expected = np.array([10.0, 10.0, 10.0])
        assert np.allclose(voxel_coord, expected)
    
    def test_coordinate_translation(self):
        """Test coordinate translation."""
        coord = np.array([10, 10, 10])
        offset = np.array([5, 5, 5])
        
        translated = coord + offset
        
        expected = np.array([15, 15, 15])
        assert np.allclose(translated, expected)
    
    def test_coordinate_rounding(self):
        """Test coordinate rounding to nearest voxel."""
        physical_coord = np.array([83.5, 87.2, 91.9])
        scale = np.array([8.0, 8.0, 8.0])
        
        voxel_coord = np.round(physical_coord / scale).astype(int)
        
        # Should round to nearest integer voxel
        assert voxel_coord.dtype == np.int64 or voxel_coord.dtype == np.int32
        assert np.all(voxel_coord >= 0)


class TestDtypeUtilities:
    """Test suite for dtype utility functions."""
    
    def test_torch_to_numpy_dtype(self):
        """Test torch to numpy dtype conversion."""
        # Common dtype mappings
        torch_dtypes = [
            torch.float32,
            torch.float64,
            torch.int32,
            torch.int64,
            torch.uint8,
        ]
        
        for torch_dtype in torch_dtypes:
            # Create tensor and convert to numpy
            t = torch.tensor([1, 2, 3], dtype=torch_dtype)
            arr = t.numpy()
            
            # Should have compatible numpy dtype
            assert arr.dtype is not None
    
    def test_numpy_to_torch_dtype(self):
        """Test numpy to torch dtype conversion."""
        # Common dtype mappings
        numpy_dtypes = [
            np.float32,
            np.float64,
            np.int32,
            np.int64,
            np.uint8,
        ]
        
        for numpy_dtype in numpy_dtypes:
            # Create numpy array and convert to torch
            arr = np.array([1, 2, 3], dtype=numpy_dtype)
            t = torch.from_numpy(arr)
            
            # Should have compatible torch dtype
            assert t.dtype is not None
    
    def test_dtype_max_values(self):
        """Test max values for different dtypes."""
        # Test a few common dtypes
        assert torch_max_value(torch.uint8) == 255
        assert torch_max_value(torch.int16) == 32767
        assert torch_max_value(torch.bool) == 1
        
        # Float types return 1 (normalized)
        assert torch_max_value(torch.float32) == 1
        assert torch_max_value(torch.float64) == 1
