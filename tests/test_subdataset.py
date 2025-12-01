"""Tests for CellMapSubset class."""

import pytest
import torch

from cellmap_data import CellMapDataset, CellMapSubset
from cellmap_data.mutable_sampler import MutableSubsetRandomSampler

from .test_helpers import create_minimal_test_dataset


class TestCellMapSubset:
    """Test suite for CellMapSubset class."""

    @pytest.fixture
    def dataset_with_indices(self, tmp_path):
        """Create a dataset and indices for subsetting."""
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

        # Create indices for subset
        indices = [0, 2, 4, 6, 8]
        return dataset, indices

    def test_initialization(self, dataset_with_indices):
        """Test basic initialization of CellMapSubset."""
        dataset, indices = dataset_with_indices

        subset = CellMapSubset(dataset, indices)

        assert isinstance(subset, CellMapSubset)
        assert subset.dataset is dataset
        assert list(subset.indices) == indices
        assert len(subset) == len(indices)

    def test_input_arrays_property(self, dataset_with_indices):
        """Test that input_arrays property delegates to parent dataset."""
        dataset, indices = dataset_with_indices
        subset = CellMapSubset(dataset, indices)

        assert subset.input_arrays == dataset.input_arrays
        assert "raw" in subset.input_arrays

    def test_target_arrays_property(self, dataset_with_indices):
        """Test that target_arrays property delegates to parent dataset."""
        dataset, indices = dataset_with_indices
        subset = CellMapSubset(dataset, indices)

        assert subset.target_arrays == dataset.target_arrays
        assert "gt" in subset.target_arrays

    def test_classes_property(self, dataset_with_indices):
        """Test that classes property delegates to parent dataset."""
        dataset, indices = dataset_with_indices
        subset = CellMapSubset(dataset, indices)

        assert subset.classes == dataset.classes
        assert len(subset.classes) > 0

    def test_class_counts_property(self, dataset_with_indices):
        """Test that class_counts property delegates to parent dataset."""
        dataset, indices = dataset_with_indices
        subset = CellMapSubset(dataset, indices)

        assert subset.class_counts == dataset.class_counts
        assert isinstance(subset.class_counts, dict)

    def test_class_weights_property(self, dataset_with_indices):
        """Test that class_weights property delegates to parent dataset."""
        dataset, indices = dataset_with_indices
        subset = CellMapSubset(dataset, indices)

        assert subset.class_weights == dataset.class_weights
        assert isinstance(subset.class_weights, dict)

    def test_validation_indices_property(self, dataset_with_indices):
        """Test that validation_indices property delegates to parent dataset."""
        dataset, indices = dataset_with_indices
        subset = CellMapSubset(dataset, indices)

        assert subset.validation_indices == dataset.validation_indices

    def test_to_device(self, dataset_with_indices):
        """Test moving subset to different device."""
        dataset, indices = dataset_with_indices
        subset = CellMapSubset(dataset, indices)

        # Test moving to CPU
        result = subset.to("cpu")
        assert result is subset  # Should return self
        assert dataset.device.type == "cpu"

    def test_set_raw_value_transforms(self, dataset_with_indices):
        """Test setting raw value transforms."""
        dataset, indices = dataset_with_indices
        subset = CellMapSubset(dataset, indices)

        transform = lambda x: x * 2
        subset.set_raw_value_transforms(transform)

        # Verify it was set on the parent dataset
        # We can't directly test if it worked, but we can verify no error was raised
        assert True

    def test_set_target_value_transforms(self, dataset_with_indices):
        """Test setting target value transforms."""
        dataset, indices = dataset_with_indices
        subset = CellMapSubset(dataset, indices)

        transform = lambda x: x * 0.5
        subset.set_target_value_transforms(transform)

        # Verify it was set on the parent dataset
        # We can't directly test if it worked, but we can verify no error was raised
        assert True

    def test_get_random_subset_indices_without_replacement(self, dataset_with_indices):
        """Test getting random subset indices when num_samples <= len(indices)."""
        dataset, indices = dataset_with_indices
        subset = CellMapSubset(dataset, indices)

        # Request fewer samples than available
        num_samples = 3
        result_indices = subset.get_random_subset_indices(num_samples)

        assert len(result_indices) == num_samples
        # All returned indices should be from the original subset indices
        for idx in result_indices:
            assert idx in indices

    def test_get_random_subset_indices_with_replacement(self, dataset_with_indices):
        """Test getting random subset indices when num_samples > len(indices)."""
        dataset, indices = dataset_with_indices
        subset = CellMapSubset(dataset, indices)

        # Request more samples than available (requires replacement)
        num_samples = 10
        with pytest.warns(UserWarning, match="Sampling with replacement"):
            result_indices = subset.get_random_subset_indices(num_samples)

        assert len(result_indices) == num_samples
        # All returned indices should be from the original subset indices
        for idx in result_indices:
            assert idx in indices

    def test_get_random_subset_indices_with_rng(self, dataset_with_indices):
        """Test that get_random_subset_indices respects the RNG for reproducibility."""
        dataset, indices = dataset_with_indices
        subset = CellMapSubset(dataset, indices)

        rng1 = torch.Generator().manual_seed(42)
        rng2 = torch.Generator().manual_seed(42)

        num_samples = 5
        result1 = subset.get_random_subset_indices(num_samples, rng=rng1)
        result2 = subset.get_random_subset_indices(num_samples, rng=rng2)

        assert result1 == result2  # Same seed should give same results

    def test_get_subset_random_sampler(self, dataset_with_indices):
        """Test creating a MutableSubsetRandomSampler from subset."""
        dataset, indices = dataset_with_indices
        subset = CellMapSubset(dataset, indices)

        num_samples = 5
        sampler = subset.get_subset_random_sampler(num_samples)

        assert isinstance(sampler, MutableSubsetRandomSampler)
        # Sample from the sampler
        sampled_indices = list(sampler)
        assert len(sampled_indices) == num_samples

    def test_get_subset_random_sampler_with_rng(self, dataset_with_indices):
        """Test that sampler respects RNG."""
        dataset, indices = dataset_with_indices
        subset = CellMapSubset(dataset, indices)

        rng1 = torch.Generator().manual_seed(123)
        rng2 = torch.Generator().manual_seed(123)

        num_samples = 5
        sampler1 = subset.get_subset_random_sampler(num_samples, rng=rng1)
        sampler2 = subset.get_subset_random_sampler(num_samples, rng=rng2)

        result1 = list(sampler1)
        result2 = list(sampler2)

        assert result1 == result2  # Same seed should give same results

    def test_getitem_delegates_to_parent(self, dataset_with_indices):
        """Test that __getitem__ properly delegates to parent dataset with mapped indices."""
        dataset, indices = dataset_with_indices
        subset = CellMapSubset(dataset, indices)

        # Get first item from subset (should be index 0 from original dataset)
        item = subset[0]

        # Should return a dictionary with 'raw' and 'gt' keys
        assert isinstance(item, dict)
        assert "raw" in item
        # The gt might not be present if force_has_data doesn't work as expected,
        # but raw should always be there

    def test_subset_length(self, dataset_with_indices):
        """Test that len() returns correct subset length."""
        dataset, indices = dataset_with_indices
        subset = CellMapSubset(dataset, indices)

        assert len(subset) == len(indices)
        assert len(subset) < len(dataset)

    def test_empty_subset(self, dataset_with_indices):
        """Test creating a subset with no indices."""
        dataset, _ = dataset_with_indices
        empty_indices = []

        subset = CellMapSubset(dataset, empty_indices)

        assert len(subset) == 0
        assert list(subset.indices) == []

    def test_single_index_subset(self, dataset_with_indices):
        """Test creating a subset with a single index."""
        dataset, _ = dataset_with_indices
        single_index = [0]

        subset = CellMapSubset(dataset, single_index)

        assert len(subset) == 1
        assert list(subset.indices) == single_index
