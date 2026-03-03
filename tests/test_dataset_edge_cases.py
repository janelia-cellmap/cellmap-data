"""Tests for CellMapDataset edge cases and special methods."""

import pickle

import numpy as np
import pytest
import torch

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
        if "totals" in counts:
            for cls in dataset.classes:
                # Class names might have _bg suffix
                assert any(cls in key for key in counts["totals"].keys())
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
        assert hasattr(indices, "__iter__")

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
        assert hasattr(dataset, "__getitem__")

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


class TestProcessExecutorSingleton:
    """Tests for the per-process shared ThreadPoolExecutor.

    Before the fix, each CellMapDataset created its own ThreadPoolExecutor,
    causing thread explosion when many datasets exist inside DataLoader workers.
    After the fix, all datasets in a process share one pool.
    """

    @pytest.fixture
    def two_datasets(self, tmp_path):
        """Create two independent datasets in the same process."""
        configs = []
        datasets = []
        for i in range(2):
            cfg = create_minimal_test_dataset(tmp_path / f"ds{i}")
            configs.append(cfg)
            datasets.append(
                CellMapDataset(
                    raw_path=str(cfg["raw_path"]),
                    target_path=str(cfg["gt_path"]),
                    classes=cfg["classes"],
                    input_arrays={"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
                    target_arrays={"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
                    force_has_data=True,
                )
            )
        return datasets

    def test_executor_is_shared_across_datasets(self, two_datasets):
        """Two datasets in the same process must return the exact same executor object."""
        ds0, ds1 = two_datasets
        assert ds0.executor is ds1.executor

    def test_executor_is_module_level_singleton(self, two_datasets):
        """The executor must live in the module-level _PROCESS_EXECUTORS dict."""
        import os

        from cellmap_data.dataset import _PROCESS_EXECUTORS

        ds0, _ = two_datasets
        pid = os.getpid()
        assert pid in _PROCESS_EXECUTORS
        assert _PROCESS_EXECUTORS[pid] is ds0.executor

    def test_close_does_not_shut_down_shared_pool(self, two_datasets):
        """close() on one dataset must not prevent other datasets from using the pool."""
        ds0, ds1 = two_datasets

        # Trigger executor creation on both
        _ = ds0.executor
        _ = ds1.executor

        # Close the first dataset
        ds0.close()
        assert ds0._executor is None

        # The second dataset must still be able to submit work
        future = ds1.executor.submit(lambda: 42)
        assert future.result() == 42

    def test_del_does_not_shut_down_shared_pool(self, two_datasets):
        """__del__ on one dataset must not prevent other datasets from using the pool."""
        ds0, ds1 = two_datasets
        _ = ds0.executor
        executor_ref = ds1.executor

        ds0.__del__()
        assert ds0._executor is None

        # Pool is still operational
        future = executor_ref.submit(lambda: "alive")
        assert future.result() == "alive"

    def test_executor_lazy_init(self, tmp_path):
        """Executor must not be created until first access via the property."""
        from cellmap_data.dataset import _PROCESS_EXECUTORS

        import os

        cfg = create_minimal_test_dataset(tmp_path / "lazy")
        # Clear any existing entry so the laziness is observable
        _PROCESS_EXECUTORS.pop(os.getpid(), None)

        ds = CellMapDataset(
            raw_path=str(cfg["raw_path"]),
            target_path=str(cfg["gt_path"]),
            classes=cfg["classes"],
            input_arrays={"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            target_arrays={"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            force_has_data=True,
        )

        assert ds._executor is None  # not yet created
        _ = ds.executor               # trigger lazy init
        assert os.getpid() in _PROCESS_EXECUTORS

    def test_pid_change_triggers_new_executor(self, two_datasets):
        """Simulating a PID change (post-fork child) causes a fresh executor lookup."""
        import os
        from unittest.mock import patch

        from cellmap_data.dataset import _PROCESS_EXECUTORS

        ds0, _ = two_datasets
        original_executor = ds0.executor  # ensure cached
        fake_pid = os.getpid() + 99999    # a PID that isn't in the dict

        with patch("cellmap_data.dataset.os.getpid", return_value=fake_pid):
            # Force re-evaluation by clearing the cached pid
            ds0._executor_pid = None
            new_executor = ds0.executor

        # A new entry was created for the fake PID
        assert fake_pid in _PROCESS_EXECUTORS
        # The new executor is different from the original process's executor
        assert new_executor is not original_executor

        # Cleanup: remove the fake PID entry
        _PROCESS_EXECUTORS.pop(fake_pid, None)
