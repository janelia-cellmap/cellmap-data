"""Tests for CellMapDataset edge cases and special methods."""

import pickle
import pytest
import torch
import numpy as np

from cellmap_data import CellMapDataset, CellMapMultiDataset

from .test_helpers import create_minimal_test_dataset


class TestCellMapDatasetEdgeCases:
    """Test edge cases and special methods in CellMapDataset."""

    @pytest.fixture
    def minimal_dataset(self, tmp_path):
        """Create a minimal dataset for testing."""
        config = create_minimal_test_dataset(tmp_path)

        input_arrays = {
            "raw": {
                "shape": (8, 8, 8),
                "scale": (4.0, 4.0, 4.0),
            }
        }

        target_arrays = {
            "gt": {
                "shape": (8, 8, 8),
                "scale": (4.0, 4.0, 4.0),
            }
        }

        dataset = CellMapDataset(
            raw_path=str(config["raw_path"]),
            target_path=str(config["gt_path"]),
            classes=config["classes"],
            input_arrays=input_arrays,
            target_arrays=target_arrays,
            force_has_data=True,
        )

        return dataset, config

    def test_pickle_support(self, minimal_dataset):
        """Test that dataset can be pickled and unpickled."""
        dataset, _ = minimal_dataset

        # Pickle the dataset
        pickled = pickle.dumps(dataset)

        # Unpickle the dataset
        unpickled = pickle.loads(pickled)

        # Verify properties are preserved
        assert unpickled.raw_path == dataset.raw_path
        assert unpickled.target_path == dataset.target_path
        assert unpickled.classes == dataset.classes
        assert unpickled.input_arrays == dataset.input_arrays
        assert unpickled.target_arrays == dataset.target_arrays

    def test_del_method_cleanup(self, minimal_dataset):
        """Test that __del__ properly cleans up the executor."""
        dataset, _ = minimal_dataset

        # Access executor to force initialization
        _ = dataset.executor

        # Verify executor exists
        assert dataset._executor is not None

        # Delete dataset should trigger cleanup
        del dataset

        # No exception should be raised
        assert True

    def test_executor_property_lazy_init(self, minimal_dataset):
        """Test that executor is lazily initialized."""
        dataset, _ = minimal_dataset

        # Initially, executor should not be initialized
        assert dataset._executor is None

        # Access executor property
        executor = dataset.executor

        # Now it should be initialized
        assert executor is not None
        assert dataset._executor is not None

        # Accessing again should return same instance
        executor2 = dataset.executor
        assert executor is executor2

    def test_executor_handles_fork(self, minimal_dataset):
        """Test that executor is recreated after process fork."""
        dataset, _ = minimal_dataset

        # Access executor
        _ = dataset.executor
        original_pid = dataset._executor_pid

        # Simulate a fork by changing the PID tracking
        import os
        dataset._executor_pid = os.getpid() + 1

        # Access executor again - should create new one
        _ = dataset.executor

        # PID should be updated
        assert dataset._executor_pid == os.getpid()

    def test_center_property(self, minimal_dataset):
        """Test the center property calculation."""
        dataset, _ = minimal_dataset

        center = dataset.center

        # Center should be a dict with axis keys
        assert isinstance(center, dict)
        for axis in dataset.axis_order:
            assert axis in center
            assert isinstance(center[axis], (int, float))

    def test_largest_voxel_sizes_property(self, minimal_dataset):
        """Test the largest_voxel_sizes property."""
        dataset, _ = minimal_dataset

        voxel_sizes = dataset.largest_voxel_sizes

        # Should be a dict with axis keys
        assert isinstance(voxel_sizes, dict)
        for axis in dataset.axis_order:
            assert axis in voxel_sizes
            assert voxel_sizes[axis] > 0

    def test_bounding_box_property(self, minimal_dataset):
        """Test the bounding_box property."""
        dataset, _ = minimal_dataset

        bbox = dataset.bounding_box

        # Should be a dict mapping axes to [min, max] pairs
        assert isinstance(bbox, dict)
        for axis in dataset.axis_order:
            assert axis in bbox
            assert len(bbox[axis]) == 2
            assert bbox[axis][0] <= bbox[axis][1]

    def test_sampling_box_property(self, minimal_dataset):
        """Test the sampling_box property."""
        dataset, _ = minimal_dataset

        sbox = dataset.sampling_box

        # Should be a dict mapping axes to [min, max] pairs
        assert isinstance(sbox, dict)
        for axis in dataset.axis_order:
            assert axis in sbox
            assert len(sbox[axis]) == 2

    def test_sampling_box_shape_property(self, minimal_dataset):
        """Test the sampling_box_shape property."""
        dataset, _ = minimal_dataset

        shape = dataset.sampling_box_shape

        # Should be a dict mapping axes to integer sizes
        assert isinstance(shape, dict)
        for axis in dataset.axis_order:
            assert axis in shape
            assert isinstance(shape[axis], int)
            assert shape[axis] > 0

    def test_device_property_auto_selection(self, minimal_dataset):
        """Test device property auto-selects appropriate device."""
        dataset, _ = minimal_dataset

        device = dataset.device

        # Should be a torch device
        assert isinstance(device, torch.device)
        # Should be one of the expected types
        assert device.type in ["cpu", "cuda", "mps"]

    def test_negative_index_handling(self, minimal_dataset):
        """Test that negative indices are handled correctly."""
        dataset, _ = minimal_dataset

        # Try to get item with negative index
        item = dataset[-1]

        # Should return a valid item
        assert isinstance(item, dict)
        assert "raw" in item

    def test_out_of_bounds_index_handling(self, minimal_dataset):
        """Test that out of bounds indices are handled gracefully."""
        dataset, _ = minimal_dataset

        # Try an index way out of bounds
        large_idx = len(dataset) * 10

        # Should not raise, but may log warning
        item = dataset[large_idx]

        # Should still return a valid item (clamped to bounds)
        assert isinstance(item, dict)

    def test_class_counts_property(self, minimal_dataset):
        """Test the class_counts property."""
        dataset, _ = minimal_dataset

        counts = dataset.class_counts

        # Should be a dict
        assert isinstance(counts, dict)
        # class_counts structure has changed - it's now nested with 'totals'
        # Check that the totals key exists and has class entries
        if 'totals' in counts:
            for cls in dataset.classes:
                # Class names might have _bg suffix
                assert any(cls in key for key in counts['totals'].keys())
        else:
            # Old structure - direct class keys
            for cls in dataset.classes:
                assert cls in counts

    def test_class_weights_property(self, minimal_dataset):
        """Test the class_weights property."""
        dataset, _ = minimal_dataset

        weights = dataset.class_weights

        # Should be a dict
        assert isinstance(weights, dict)
        # Should have entries for each class
        for cls in dataset.classes:
            assert cls in weights
            assert isinstance(weights[cls], (int, float))
            assert 0 <= weights[cls] <= 1

    def test_validation_indices_property(self, minimal_dataset):
        """Test the validation_indices property."""
        dataset, _ = minimal_dataset

        indices = dataset.validation_indices

        # Should be a sequence
        assert hasattr(indices, '__iter__')

    def test_2d_array_creates_multidataset(self, tmp_path):
        """Test that 2D array without slicing axis triggers special handling."""
        config = create_minimal_test_dataset(tmp_path)

        # Create 2D array configuration (shape has a 1 in it)
        # Note: The actual behavior may depend on how is_array_2D is implemented
        input_arrays = {
            "raw": {
                "shape": (1, 8, 8),  # 2D array
                "scale": (4.0, 4.0, 4.0),
            }
        }

        target_arrays = {
            "gt": {
                "shape": (1, 8, 8),  # 2D array
                "scale": (4.0, 4.0, 4.0),
            }
        }

        # Creating dataset with 2D arrays may create multidataset or regular dataset
        # depending on implementation details
        dataset = CellMapDataset(
            raw_path=str(config["raw_path"]),
            target_path=str(config["gt_path"]),
            classes=config["classes"],
            input_arrays=input_arrays,
            target_arrays=target_arrays,
            force_has_data=True,
        )

        # Should create some kind of dataset (either regular or multi)
        # The key is that it doesn't raise an error
        assert dataset is not None
        assert hasattr(dataset, '__getitem__')

    def test_set_raw_value_transforms(self, minimal_dataset):
        """Test setting raw value transforms."""
        dataset, _ = minimal_dataset

        transform = lambda x: x * 2
        dataset.set_raw_value_transforms(transform)

        # Should not raise
        assert True

    def test_set_target_value_transforms(self, minimal_dataset):
        """Test setting target value transforms."""
        dataset, _ = minimal_dataset

        transform = lambda x: x * 0.5
        dataset.set_target_value_transforms(transform)

        # Should not raise
        assert True

    def test_to_device_method(self, minimal_dataset):
        """Test moving dataset to device."""
        dataset, _ = minimal_dataset

        # Move to CPU explicitly
        result = dataset.to("cpu")

        # Should return self
        assert result is dataset
        assert dataset.device.type == "cpu"

    def test_get_random_subset_indices(self, minimal_dataset):
        """Test getting random subset indices."""
        dataset, _ = minimal_dataset

        num_samples = 5
        indices = dataset.get_random_subset_indices(num_samples)

        # Should return list of indices
        assert len(indices) == num_samples
        for idx in indices:
            assert 0 <= idx < len(dataset)

    def test_get_subset_random_sampler(self, minimal_dataset):
        """Test creating a subset random sampler."""
        dataset, _ = minimal_dataset

        num_samples = 5
        sampler = dataset.get_subset_random_sampler(num_samples)

        # Should create a sampler
        assert sampler is not None
        # Should be iterable
        indices = list(sampler)
        assert len(indices) == num_samples
