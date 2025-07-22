"""
Test coverage improvements for low-hanging fruit files.

This module focuses on achieving high coverage for small, testable files:
1. MutableSubsetRandomSampler (70% → 100%)
2. EmptyImage (95% → 100%)
3. CellMapSubset (64% → ~90%)
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock

from cellmap_data.mutable_sampler import MutableSubsetRandomSampler
from cellmap_data.empty_image import EmptyImage
from cellmap_data.subdataset import CellMapSubset


class TestMutableSubsetRandomSampler:
    """Test the MutableSubsetRandomSampler class for 100% coverage."""

    def test_initialization(self):
        """Test basic initialization of MutableSubsetRandomSampler."""

        def indices_gen():
            return [0, 1, 2, 3, 4]

        sampler = MutableSubsetRandomSampler(indices_gen)

        assert sampler.indices == [0, 1, 2, 3, 4]
        assert sampler.indices_generator is indices_gen
        assert sampler.rng is None
        assert len(sampler) == 5

    def test_initialization_with_rng(self):
        """Test initialization with custom random number generator."""

        def indices_gen():
            return [10, 20, 30]

        rng = torch.Generator()
        rng.manual_seed(42)

        sampler = MutableSubsetRandomSampler(indices_gen, rng=rng)

        assert sampler.indices == [10, 20, 30]
        assert sampler.rng is rng
        assert len(sampler) == 3

    def test_iter_deterministic(self):
        """Test that __iter__ produces deterministic results with seeded RNG."""

        def indices_gen():
            return [0, 1, 2, 3, 4]

        rng = torch.Generator()
        rng.manual_seed(42)

        sampler = MutableSubsetRandomSampler(indices_gen, rng=rng)

        # Get first iteration
        first_iteration = list(sampler)

        # Reset RNG and get second iteration
        rng.manual_seed(42)
        sampler.rng = rng
        second_iteration = list(sampler)

        assert first_iteration == second_iteration
        assert len(first_iteration) == 5
        assert set(first_iteration) == {0, 1, 2, 3, 4}

    def test_iter_random_without_seed(self):
        """Test that __iter__ produces random permutations when no seed is set."""

        def indices_gen():
            return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        sampler = MutableSubsetRandomSampler(indices_gen)

        # Get multiple iterations
        iterations = [list(sampler) for _ in range(5)]

        # All should have same length and same elements
        for iteration in iterations:
            assert len(iteration) == 10
            assert set(iteration) == set(range(10))

        # At least some should be different (very unlikely to be all identical)
        unique_iterations = [tuple(it) for it in iterations]
        assert len(set(unique_iterations)) > 1, "Expected some randomness in iterations"

    def test_refresh_updates_indices(self):
        """Test that refresh() updates indices by calling the generator."""
        call_count = 0

        def dynamic_indices_gen():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [0, 1, 2]
            else:
                return [10, 20, 30, 40]

        sampler = MutableSubsetRandomSampler(dynamic_indices_gen)

        # Initial state
        assert sampler.indices == [0, 1, 2]
        assert len(sampler) == 3

        # After refresh
        sampler.refresh()
        assert sampler.indices == [10, 20, 30, 40]
        assert len(sampler) == 4

    def test_empty_indices(self):
        """Test behavior with empty indices."""

        def empty_indices_gen():
            return []

        sampler = MutableSubsetRandomSampler(empty_indices_gen)

        assert sampler.indices == []
        assert len(sampler) == 0
        assert list(sampler) == []

    def test_single_index(self):
        """Test behavior with single index."""

        def single_index_gen():
            return [42]

        sampler = MutableSubsetRandomSampler(single_index_gen)

        assert sampler.indices == [42]
        assert len(sampler) == 1
        assert list(sampler) == [42]


class TestEmptyImage:
    """Test the EmptyImage class for 100% coverage."""

    def test_basic_initialization(self):
        """Test basic EmptyImage initialization."""
        empty_img = EmptyImage(
            target_class="test_class",
            target_scale=[1.0, 1.0, 1.0],
            target_voxel_shape=[32, 32, 32],
        )

        assert empty_img.label_class == "test_class"
        assert empty_img.target_scale == [1.0, 1.0, 1.0]
        assert empty_img.axes == "zyx"
        assert empty_img.output_shape == {"z": 32, "y": 32, "x": 32}
        assert empty_img.output_size == {"z": 32.0, "y": 32.0, "x": 32.0}
        assert empty_img.scale == {"z": 1.0, "y": 1.0, "x": 1.0}
        assert empty_img.empty_value == -100

    def test_initialization_with_custom_empty_value(self):
        """Test initialization with custom empty value."""
        empty_img = EmptyImage(
            target_class="test",
            target_scale=[2.0, 2.0, 2.0],
            target_voxel_shape=[16, 16, 16],
            empty_value=999.0,
        )

        assert empty_img.empty_value == 999.0
        assert torch.all(empty_img.store == 999.0)

    def test_initialization_with_custom_store(self):
        """Test initialization with pre-provided store tensor."""
        custom_store = torch.ones((16, 16, 16)) * 42.0

        empty_img = EmptyImage(
            target_class="test",
            target_scale=[1.0, 1.0, 1.0],
            target_voxel_shape=[16, 16, 16],
            store=custom_store,
        )

        assert torch.equal(empty_img.store, custom_store)
        assert torch.all(empty_img.store == 42.0)

    def test_custom_axis_order(self):
        """Test initialization with custom axis order."""
        empty_img = EmptyImage(
            target_class="test",
            target_scale=[1.0, 1.0],
            target_voxel_shape=[64, 32],
            axis_order="yx",
        )

        assert empty_img.axes == "yx"
        assert empty_img.output_shape == {"y": 64, "x": 32}
        assert empty_img.output_size == {"y": 64.0, "x": 32.0}

    def test_axis_order_truncation(self):
        """Test that axis order is truncated when longer than voxel shape."""
        empty_img = EmptyImage(
            target_class="test",
            target_scale=[2.0, 2.0],
            target_voxel_shape=[16, 32],
            axis_order="zyxabc",  # Longer than voxel shape
        )

        assert empty_img.axes == "bc"  # Should be truncated from the end
        assert empty_img.output_shape == {"b": 16, "c": 32}

    def test_getitem_returns_store(self):
        """Test that __getitem__ returns the store tensor."""
        empty_img = EmptyImage(
            target_class="test",
            target_scale=[1.0, 1.0, 1.0],
            target_voxel_shape=[8, 8, 8],
        )

        center = {"x": 0.0, "y": 0.0, "z": 0.0}
        result = empty_img[center]

        assert torch.equal(result, empty_img.store)
        assert result.shape == (8, 8, 8)

    def test_properties(self):
        """Test all property methods."""
        empty_img = EmptyImage(
            target_class="test",
            target_scale=[1.0, 1.0, 1.0],
            target_voxel_shape=[16, 16, 16],
        )

        assert empty_img.bounding_box is None
        assert empty_img.sampling_box is None
        assert empty_img.bg_count == 0.0
        assert empty_img.class_counts == 0.0

    def test_to_device(self):
        """Test moving EmptyImage to different device."""
        empty_img = EmptyImage(
            target_class="test",
            target_scale=[1.0, 1.0, 1.0],
            target_voxel_shape=[8, 8, 8],
        )

        # Test CPU (should work everywhere)
        empty_img.to("cpu")
        assert empty_img.store.device.type == "cpu"

        # Test CUDA if available
        if torch.cuda.is_available():
            empty_img.to("cuda")
            assert empty_img.store.device.type == "cuda"

    def test_to_device_non_blocking(self):
        """Test non_blocking parameter in to() method."""
        empty_img = EmptyImage(
            target_class="test",
            target_scale=[1.0, 1.0, 1.0],
            target_voxel_shape=[4, 4, 4],
        )

        # Test with non_blocking=False
        empty_img.to("cpu", non_blocking=False)
        assert empty_img.store.device.type == "cpu"

    def test_set_spatial_transforms_no_op(self):
        """Test that set_spatial_transforms does nothing (no-op)."""
        empty_img = EmptyImage(
            target_class="test",
            target_scale=[1.0, 1.0, 1.0],
            target_voxel_shape=[8, 8, 8],
        )

        # Should not raise any errors and not change anything
        empty_img.set_spatial_transforms({"rotation": 45})
        empty_img.set_spatial_transforms(None)

        # Store should be unchanged
        assert empty_img.store.shape == (8, 8, 8)


class TestCellMapSubset:
    """Test the CellMapSubset class for improved coverage."""

    def test_initialization(self):
        """Test CellMapSubset initialization with mock dataset."""
        # Create a mock dataset
        mock_dataset = MagicMock()
        mock_dataset.classes = ["class1", "class2", "class3"]
        mock_dataset.class_counts = {"class1": 100.0, "class2": 200.0, "class3": 150.0}
        mock_dataset.__len__ = MagicMock(return_value=1000)

        indices = [0, 1, 2, 5, 10, 100]

        subset = CellMapSubset(mock_dataset, indices)

        assert subset.dataset is mock_dataset
        assert subset.indices == indices
        assert len(subset) == len(indices)

    def test_classes_property(self):
        """Test that classes property delegates to dataset."""
        mock_dataset = MagicMock()
        mock_dataset.classes = ["neuron", "mitochondria", "endoplasmic_reticulum"]

        subset = CellMapSubset(mock_dataset, [0, 1, 2])

        assert subset.classes == ["neuron", "mitochondria", "endoplasmic_reticulum"]

    def test_class_counts_property(self):
        """Test that class_counts property delegates to dataset."""
        mock_dataset = MagicMock()
        mock_dataset.class_counts = {
            "neurons": 500.5,
            "mitochondria": 1200.2,
            "vesicles": 75.8,
        }

        subset = CellMapSubset(mock_dataset, [10, 20, 30, 40])

        assert subset.class_counts == {
            "neurons": 500.5,
            "mitochondria": 1200.2,
            "vesicles": 75.8,
        }

    def test_getitem_delegates_to_dataset(self):
        """Test that __getitem__ correctly delegates to the underlying dataset."""
        mock_dataset = MagicMock()
        mock_dataset.__getitem__ = MagicMock(return_value="mock_item")

        indices = [5, 10, 15, 20]
        subset = CellMapSubset(mock_dataset, indices)

        # Access subset index 2, which should map to dataset index 15
        result = subset[2]

        mock_dataset.__getitem__.assert_called_once_with(15)
        assert result == "mock_item"

    def test_empty_subset(self):
        """Test CellMapSubset with empty indices."""
        mock_dataset = MagicMock()
        mock_dataset.classes = ["class1"]
        mock_dataset.class_counts = {"class1": 50.0}

        subset = CellMapSubset(mock_dataset, [])

        assert len(subset) == 0
        assert subset.classes == ["class1"]
        assert subset.class_counts == {"class1": 50.0}

    def test_single_index_subset(self):
        """Test CellMapSubset with single index."""
        mock_dataset = MagicMock()
        mock_dataset.classes = ["test_class"]
        mock_dataset.class_counts = {"test_class": 25.0}
        mock_dataset.__getitem__ = MagicMock(return_value="single_item")

        subset = CellMapSubset(mock_dataset, [42])

        assert len(subset) == 1
        result = subset[0]

        mock_dataset.__getitem__.assert_called_once_with(42)
        assert result == "single_item"


def test_integration_mutable_sampler_with_cellmap_subset():
    """Test integration between MutableSubsetRandomSampler and CellMapSubset."""
    # Create a mock dataset
    mock_dataset = MagicMock()
    mock_dataset.classes = ["class1", "class2"]
    mock_dataset.class_counts = {"class1": 100.0, "class2": 200.0}
    mock_dataset.__len__ = MagicMock(return_value=1000)

    # Create subset
    subset = CellMapSubset(mock_dataset, list(range(100)))

    # Create sampler that generates indices for the subset
    def subset_indices_gen():
        return list(range(0, 100, 10))  # Every 10th element from subset

    sampler = MutableSubsetRandomSampler(subset_indices_gen)

    # Test that the sampler works with subset length
    assert len(sampler) == 10
    assert all(0 <= idx < len(subset) for idx in sampler)

    # Test refresh
    sampler.refresh()
    assert len(sampler) == 10


if __name__ == "__main__":
    pytest.main([__file__])
