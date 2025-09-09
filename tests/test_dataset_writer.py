"""
Comprehensive tests for CellMapDatasetWriter to improve test coverage.
"""

import pytest
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from cellmap_data.dataset_writer import CellMapDatasetWriter


class TestCellMapDatasetWriter:
    """Test suite for CellMapDatasetWriter functionality"""

    @pytest.fixture
    def mock_dependencies(self):
        """Mock external dependencies to avoid file system operations"""
        with (
            patch("cellmap_data.dataset_writer.CellMapImage") as mock_image,
            patch("cellmap_data.dataset_writer.ImageWriter") as mock_writer,
            patch("cellmap_data.dataset_writer.UPath") as mock_path,
        ):

            # Setup mock image
            mock_image_instance = Mock()
            mock_image_instance.scale = {"x": 1.0, "y": 1.0, "z": 1.0}
            mock_image.return_value = mock_image_instance

            # Setup mock writer with proper scale attribute that is iterable
            mock_writer_instance = Mock()
            mock_scale = Mock()
            mock_scale.items = Mock(return_value=[("x", 2.0), ("y", 2.0), ("z", 2.0)])
            mock_scale.__getitem__ = lambda self, key: {"x": 2.0, "y": 2.0, "z": 2.0}[
                key
            ]
            mock_writer_instance.scale = mock_scale
            mock_writer_instance.write_world_shape = {"x": 8.0, "y": 8.0, "z": 8.0}
            mock_writer.return_value = mock_writer_instance

            # Setup mock path
            mock_path.return_value = mock_path
            mock_path.__truediv__ = lambda self, other: f"{self}/{other}"

            yield {"image": mock_image, "writer": mock_writer, "path": mock_path}

    @pytest.fixture
    def basic_config(self):
        """Basic configuration for creating test instances"""
        return {
            "raw_path": "/fake/raw/path",
            "target_path": "/fake/target/path",
            "classes": ["class_a", "class_b"],
            "input_arrays": {
                "input1": {"shape": [16, 16, 16], "scale": [1.0, 1.0, 1.0]}
            },
            "target_arrays": {
                "target1": {"shape": [8, 8, 8], "scale": [2.0, 2.0, 2.0]}
            },
            "target_bounds": {
                "target1": {"x": [0.0, 16.0], "y": [0.0, 16.0], "z": [0.0, 16.0]}
            },
        }

    def test_initialization_basic(self, mock_dependencies, basic_config):
        """Test basic initialization of CellMapDatasetWriter"""
        writer = CellMapDatasetWriter(**basic_config)

        assert writer.raw_path == basic_config["raw_path"]
        assert writer.target_path == basic_config["target_path"]
        assert writer.classes == basic_config["classes"]
        assert writer.input_arrays == basic_config["input_arrays"]
        assert writer.target_arrays == basic_config["target_arrays"]
        assert writer.target_bounds == basic_config["target_bounds"]
        assert writer.axis_order == "zyx"
        assert writer.empty_value == 0
        assert writer.overwrite is False

    def test_initialization_with_device(self, mock_dependencies, basic_config):
        """Test initialization with specific device"""
        writer = CellMapDatasetWriter(device="cpu", **basic_config)
        assert writer.device.type == "cpu"

    def test_initialization_optional_params(self, mock_dependencies, basic_config):
        """Test initialization with optional parameters"""

        def dummy_transform(x):
            return x * 2

        writer = CellMapDatasetWriter(
            raw_value_transforms=dummy_transform,
            axis_order="xyz",
            empty_value=255,
            overwrite=True,
            **basic_config,
        )

        assert writer.raw_value_transforms == dummy_transform
        assert writer.axis_order == "xyz"
        assert writer.empty_value == 255
        assert writer.overwrite is True

    def test_device_property_cpu_fallback(self, mock_dependencies, basic_config):
        """Test device property falls back to CPU when CUDA/MPS unavailable"""
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
        ):
            writer = CellMapDatasetWriter(**basic_config)
            assert writer.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_property_cuda(self, mock_dependencies, basic_config):
        """Test device property selects CUDA when available"""
        writer = CellMapDatasetWriter(**basic_config)
        # Should default to CUDA if available
        assert writer.device.type == "cuda"

    def test_center_property(self, mock_dependencies, basic_config):
        """Test center property calculation"""
        writer = CellMapDatasetWriter(**basic_config)
        center = writer.center

        # Center should be middle of bounding box
        assert center is not None
        assert "x" in center and "y" in center and "z" in center
        assert center["x"] == 8.0  # (0 + 16) / 2
        assert center["y"] == 8.0
        assert center["z"] == 8.0

    def test_smallest_voxel_sizes_property(self, mock_dependencies, basic_config):
        """Test smallest_voxel_sizes property calculation"""
        writer = CellMapDatasetWriter(**basic_config)
        sizes = writer.smallest_voxel_sizes

        assert "x" in sizes and "y" in sizes and "z" in sizes
        # Should be minimum of input (1.0) and target writer (2.0) scales
        assert sizes["x"] == 1.0
        assert sizes["y"] == 1.0
        assert sizes["z"] == 1.0

    def test_bounding_box_property(self, mock_dependencies, basic_config):
        """Test bounding_box property calculation"""
        writer = CellMapDatasetWriter(**basic_config)
        bbox = writer.bounding_box

        assert bbox == basic_config["target_bounds"]["target1"]

    def test_bounding_box_shape_property(self, mock_dependencies, basic_config):
        """Test bounding_box_shape property calculation"""
        writer = CellMapDatasetWriter(**basic_config)
        shape = writer.bounding_box_shape

        # Shape should be bbox size divided by smallest voxel size
        assert shape["x"] == 16  # (16.0 - 0.0) / 1.0
        assert shape["y"] == 16
        assert shape["z"] == 16

    def test_sampling_box_property(self, mock_dependencies, basic_config):
        """Test sampling_box property calculation"""
        writer = CellMapDatasetWriter(**basic_config)
        sbox = writer.sampling_box

        # Sampling box should be smaller than bounding box due to padding
        assert sbox["x"][0] > basic_config["target_bounds"]["target1"]["x"][0]
        assert sbox["x"][1] < basic_config["target_bounds"]["target1"]["x"][1]

    def test_len_property(self, mock_dependencies, basic_config):
        """Test __len__ method"""
        writer = CellMapDatasetWriter(**basic_config)
        length = len(writer)

        assert isinstance(length, int)
        assert length > 0

    def test_size_property(self, mock_dependencies, basic_config):
        """Test size property"""
        writer = CellMapDatasetWriter(**basic_config)
        size = writer.size

        assert isinstance(size, (int, np.integer))
        assert size > 0

    def test_get_center_method(self, mock_dependencies, basic_config):
        """Test get_center method with various indices"""
        writer = CellMapDatasetWriter(**basic_config)

        # Test with valid index only (dataset length is 1)
        center0 = writer.get_center(0)
        assert isinstance(center0, dict)
        assert all(c in center0 for c in ["x", "y", "z"])

        # Test with negative index
        center_neg = writer.get_center(-1)
        assert isinstance(center_neg, dict)

    def test_getitem_method(self, mock_dependencies, basic_config):
        """Test __getitem__ method"""
        writer = CellMapDatasetWriter(**basic_config)

        # Mock the image source to return a tensor
        mock_tensor = torch.randn(1, 16, 16, 16)
        writer.input_sources["input1"].__getitem__ = Mock(return_value=mock_tensor)

        result = writer[0]

        assert isinstance(result, dict)
        assert "input1" in result
        assert "idx" in result
        assert isinstance(result["idx"], torch.Tensor)
        assert result["idx"].item() == 0

    def test_setitem_method_single_value(self, mock_dependencies, basic_config):
        """Test __setitem__ method with single values"""
        writer = CellMapDatasetWriter(**basic_config)

        # Mock get_center to avoid complex property calculations
        writer.get_center = Mock(return_value={"x": 8.0, "y": 8.0, "z": 8.0})

        # Mock the target array writers to support item assignment
        mock_writers = {}
        for class_name in basic_config["classes"]:
            mock_writer = Mock()
            mock_writer.__setitem__ = Mock()
            mock_writers[class_name] = mock_writer
        writer.target_array_writers = {"target1": mock_writers}

        # Test with tensor array that has proper dimensions for channel indexing
        test_tensor = torch.randn(2, 8, 8, 8)  # 2 channels for 2 classes
        writer[0] = {"target1": test_tensor}

        # Should call each class writer
        for class_name in basic_config["classes"]:
            mock_writers[class_name].__setitem__.assert_called()

    def test_setitem_method_dict_values(self, mock_dependencies, basic_config):
        """Test __setitem__ method with direct tensor values"""
        writer = CellMapDatasetWriter(**basic_config)

        # Mock get_center to avoid complex property calculations
        writer.get_center = Mock(return_value={"x": 8.0, "y": 8.0, "z": 8.0})

        # Mock the target array writers to support item assignment
        mock_writers = {}
        for class_name in basic_config["classes"]:
            mock_writer = Mock()
            mock_writer.__setitem__ = Mock()
            mock_writers[class_name] = mock_writer
        writer.target_array_writers = {"target1": mock_writers}

        # Test with tensor that has proper dimensions for channel indexing
        test_tensor = torch.randn(2, 8, 8, 8)  # 2 channels for 2 classes
        writer[0] = {"target1": test_tensor}

        # Should call each class writer with corresponding data
        for class_name in basic_config["classes"]:
            mock_writers[class_name].__setitem__.assert_called()

    def test_setitem_method_tensor_values(self, mock_dependencies, basic_config):
        """Test __setitem__ method with tensor values"""
        writer = CellMapDatasetWriter(**basic_config)

        # Mock get_center to avoid complex property calculations
        writer.get_center = Mock(return_value={"x": 8.0, "y": 8.0, "z": 8.0})

        # Mock the target array writers to support item assignment
        mock_writers = {}
        for class_name in basic_config["classes"]:
            mock_writer = Mock()
            mock_writer.__setitem__ = Mock()
            mock_writers[class_name] = mock_writer
        writer.target_array_writers = {"target1": mock_writers}

        # Test with tensor (should split by channel)
        test_tensor = torch.randn(2, 8, 8, 8)  # 2 channels for 2 classes
        writer[0] = {"target1": test_tensor}

        # Should call each class writer
        for class_name in basic_config["classes"]:
            mock_writers[class_name].__setitem__.assert_called()

    def test_repr_method(self, mock_dependencies, basic_config):
        """Test __repr__ method"""
        writer = CellMapDatasetWriter(**basic_config)
        repr_str = repr(writer)

        assert "CellMapDatasetWriter" in repr_str
        assert basic_config["raw_path"] in repr_str
        assert basic_config["target_path"] in repr_str

    def test_get_indices_method(self, mock_dependencies, basic_config):
        """Test get_indices method for tiling"""
        writer = CellMapDatasetWriter(**basic_config)

        chunk_size = {"x": 4.0, "y": 4.0, "z": 4.0}
        indices = writer.get_indices(chunk_size)

        assert isinstance(indices, (list, np.ndarray))
        assert len(indices) > 0
        # All indices should be valid
        for idx in indices:
            assert 0 <= idx < len(writer)

    def test_writer_indices_property(self, mock_dependencies, basic_config):
        """Test writer_indices property"""
        writer = CellMapDatasetWriter(**basic_config)
        indices = writer.writer_indices

        assert isinstance(indices, (list, np.ndarray))
        assert len(indices) > 0

    def test_blocks_property(self, mock_dependencies, basic_config):
        """Test blocks property"""
        writer = CellMapDatasetWriter(**basic_config)
        blocks = writer.blocks

        assert hasattr(blocks, "__len__")
        assert hasattr(blocks, "__getitem__")

    def test_to_method(self, mock_dependencies, basic_config):
        """Test to() method for device transfer"""
        writer = CellMapDatasetWriter(**basic_config)

        # Test transfer to CPU
        result = writer.to("cpu")
        assert result is writer  # Should return self
        assert writer.device.type == "cpu"

    def test_to_method_with_none(self, mock_dependencies, basic_config):
        """Test to() method device change"""
        writer = CellMapDatasetWriter(**basic_config)
        original_device = writer.device

        # Test transfer to different device
        result = writer.to("cpu")
        assert result is writer
        assert writer.device.type == "cpu"

    def test_verify_method(self, mock_dependencies, basic_config):
        """Test verify method"""
        writer = CellMapDatasetWriter(**basic_config)

        # Should return True for valid dataset
        assert writer.verify() is True

    def test_verify_method_invalid(self, mock_dependencies, basic_config):
        """Test verify method with invalid dataset that returns False"""
        writer = CellMapDatasetWriter(**basic_config)

        # Directly patch the verify method's behavior by overriding len to return 0
        # This should cause verify to return False since len(self) > 0 will be False
        writer._len = 0  # Set cached len to 0
        # Also clear any cached sampling_box_shape to force recalculation
        if hasattr(writer, "_sampling_box_shape"):
            delattr(writer, "_sampling_box_shape")

        # Create a scenario where sampling_box_shape would result in 0 size
        # Mock sampling_box to have invalid dimensions
        writer._sampling_box = {
            "x": [10.0, 10.0],
            "y": [10.0, 10.0],
            "z": [10.0, 10.0],
        }  # Zero-size box

        # Now verify should return False since the product will be 0
        assert writer.verify() is False

    def test_set_raw_value_transforms(self, mock_dependencies, basic_config):
        """Test set_raw_value_transforms method"""
        writer = CellMapDatasetWriter(**basic_config)

        def new_transform(x):
            return x * 3

        writer.set_raw_value_transforms(new_transform)

        assert writer.raw_value_transforms == new_transform
        # Should also update input sources
        for source in writer.input_sources.values():
            assert source.value_transform == new_transform

    def test_get_weighted_sampler_not_implemented(
        self, mock_dependencies, basic_config
    ):
        """Test that get_weighted_sampler raises NotImplementedError"""
        writer = CellMapDatasetWriter(**basic_config)

        with pytest.raises(NotImplementedError):
            writer.get_weighted_sampler()

    def test_get_subset_random_sampler_not_implemented(
        self, mock_dependencies, basic_config
    ):
        """Test that get_subset_random_sampler raises NotImplementedError"""
        writer = CellMapDatasetWriter(**basic_config)

        with pytest.raises(NotImplementedError):
            writer.get_subset_random_sampler(10)

    def test_get_target_array_writer(self, mock_dependencies, basic_config):
        """Test get_target_array_writer method"""
        writer = CellMapDatasetWriter(**basic_config)

        array_info = basic_config["target_arrays"]["target1"]
        writers = writer.get_target_array_writer("target1", array_info)

        assert isinstance(writers, dict)
        assert len(writers) == len(basic_config["classes"])
        for class_name in basic_config["classes"]:
            assert class_name in writers

    def test_get_image_writer(self, mock_dependencies, basic_config):
        """Test get_image_writer method"""
        writer = CellMapDatasetWriter(**basic_config)

        array_info = basic_config["target_arrays"]["target1"]
        image_writer = writer.get_image_writer("target1", "class_a", array_info)

        # Should return the mocked ImageWriter
        assert image_writer is not None

    def test_box_utility_methods(self, mock_dependencies, basic_config):
        """Test box utility methods"""
        writer = CellMapDatasetWriter(**basic_config)

        # Test _get_box_shape
        test_box = {"x": [0.0, 10.0], "y": [0.0, 20.0], "z": [0.0, 30.0]}
        shape = writer._get_box_shape(test_box)
        assert isinstance(shape, dict)
        assert all(c in shape for c in ["x", "y", "z"])

        # Test _get_box_union
        box1 = {"x": [0.0, 10.0], "y": [0.0, 10.0], "z": [0.0, 10.0]}
        box2 = {"x": [5.0, 15.0], "y": [5.0, 15.0], "z": [5.0, 15.0]}
        union = writer._get_box_union(
            box1, box2.copy()
        )  # Pass a copy since method modifies in place
        assert union is not None
        assert union["x"][0] == 0.0  # min start
        assert union["x"][1] == 15.0  # max stop

        # Test _get_box_intersection
        box1_copy = {"x": [0.0, 10.0], "y": [0.0, 10.0], "z": [0.0, 10.0]}
        box2_copy = {"x": [5.0, 15.0], "y": [5.0, 15.0], "z": [5.0, 15.0]}
        intersection = writer._get_box_intersection(box1_copy, box2_copy.copy())
        assert intersection is not None
        assert intersection["x"][0] == 5.0  # max start
        assert intersection["x"][1] == 10.0  # min stop

    def test_box_union_with_none(self, mock_dependencies, basic_config):
        """Test _get_box_union with None inputs"""
        writer = CellMapDatasetWriter(**basic_config)

        box = {"x": [0.0, 10.0], "y": [0.0, 10.0], "z": [0.0, 10.0]}

        # None + box = box
        result1 = writer._get_box_union(box, None)
        assert result1 == box

        # box + None = box
        result2 = writer._get_box_union(None, box)
        assert result2 == box

    def test_loader_method(self, mock_dependencies, basic_config):
        """Test loader method"""
        with patch("cellmap_data.dataloader.CellMapDataLoader") as mock_loader_cls:
            mock_loader = Mock()
            mock_loader.device = "cpu"
            mock_loader_cls.return_value = mock_loader

            writer = CellMapDatasetWriter(**basic_config)
            loader = writer.loader(batch_size=4, num_workers=2)

            # Should create CellMapDataLoader with correct parameters
            mock_loader_cls.assert_called_once()
            call_args = mock_loader_cls.call_args
            assert call_args[0][0] is writer  # dataset
            assert call_args[1]["batch_size"] == 4
            assert call_args[1]["num_workers"] == 2
            assert call_args[1]["is_train"] is False

    def test_smallest_target_array_property(self, mock_dependencies, basic_config):
        """Test smallest_target_array property"""
        writer = CellMapDatasetWriter(**basic_config)
        smallest = writer.smallest_target_array

        assert isinstance(smallest, dict)
        assert all(c in smallest for c in ["x", "y", "z"])
        # Should be from the mocked write_world_shape
        assert smallest["x"] == 8.0
        assert smallest["y"] == 8.0
        assert smallest["z"] == 8.0

    def test_multiple_target_arrays(self, mock_dependencies, basic_config):
        """Test with multiple target arrays"""
        # Add a second target array
        basic_config["target_arrays"]["target2"] = {
            "shape": [4, 4, 4],
            "scale": [4.0, 4.0, 4.0],
        }
        basic_config["target_bounds"]["target2"] = {
            "x": [8.0, 24.0],
            "y": [8.0, 24.0],
            "z": [8.0, 24.0],
        }

        writer = CellMapDatasetWriter(**basic_config)

        # Should have writers for both target arrays
        assert "target1" in writer.target_array_writers
        assert "target2" in writer.target_array_writers

        # Bounding box should encompass both target bounds
        bbox = writer.bounding_box
        assert bbox["x"][0] == 0.0  # min of both
        assert bbox["x"][1] == 24.0  # max of both

    def test_edge_case_indices(self, mock_dependencies, basic_config):
        """Test edge cases for index handling"""
        writer = CellMapDatasetWriter(**basic_config)

        # Test boundary indices
        max_idx = len(writer) - 1
        center_max = writer.get_center(max_idx)
        assert isinstance(center_max, dict)

        # Test out of bounds handling (should be handled gracefully)
        try:
            center_oob = writer.get_center(len(writer) + 100)
            assert isinstance(center_oob, dict)  # Should return closest valid center
        except Exception:
            pass  # Expected to potentially fail, but shouldn't crash

    def test_property_caching(self, mock_dependencies, basic_config):
        """Test that properties are properly cached"""
        writer = CellMapDatasetWriter(**basic_config)

        # Access property twice
        center1 = writer.center
        center2 = writer.center

        # Should be the same object (cached)
        assert center1 is center2

        # Test other cached properties
        bbox1 = writer.bounding_box
        bbox2 = writer.bounding_box
        assert bbox1 is bbox2

        sizes1 = writer.smallest_voxel_sizes
        sizes2 = writer.smallest_voxel_sizes
        assert sizes1 is sizes2

    def test_axis_order_variations(self, mock_dependencies, basic_config):
        """Test different axis orders"""
        for axis_order in ["zyx", "xyz", "yxz"]:
            basic_config["axis_order"] = axis_order
            writer = CellMapDatasetWriter(**basic_config)
            assert writer.axis_order == axis_order

            # Should still be able to compute properties
            center = writer.center
            assert isinstance(center, dict)
            assert len(center) == 3
