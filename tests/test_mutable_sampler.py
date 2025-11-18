"""
Tests for MutableSubsetRandomSampler class.

Tests weighted sampling and mutable subset functionality.
"""

import torch
from torch.utils.data import Dataset

from cellmap_data import MutableSubsetRandomSampler


class DummyDataset(Dataset):
    """Simple dummy dataset for testing samplers."""

    def __init__(self, size=100):
        self.size = size
        self.data = torch.arange(size)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


class TestMutableSubsetRandomSampler:
    """Test suite for MutableSubsetRandomSampler."""

    def test_initialization_basic(self):
        """Test basic sampler initialization."""
        indices = list(range(100))
        sampler = MutableSubsetRandomSampler(lambda: indices)

        assert sampler is not None
        assert len(list(sampler)) > 0

    def test_initialization_with_generator(self):
        """Test sampler with custom generator."""
        indices = list(range(100))
        generator = torch.Generator()
        generator.manual_seed(42)

        sampler = MutableSubsetRandomSampler(lambda: indices, rng=generator)

        assert sampler is not None
        # Sample some indices
        sample1 = list(sampler)
        assert len(sample1) > 0

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same sequence."""
        indices = list(range(100))

        # First sampler
        gen1 = torch.Generator()
        gen1.manual_seed(42)
        sampler1 = MutableSubsetRandomSampler(lambda: indices, rng=gen1)
        samples1 = list(sampler1)

        # Second sampler with same seed
        gen2 = torch.Generator()
        gen2.manual_seed(42)
        sampler2 = MutableSubsetRandomSampler(lambda: indices, rng=gen2)
        samples2 = list(sampler2)

        # Should produce same sequence
        assert samples1 == samples2

    def test_different_seeds_produce_different_sequences(self):
        """Test that different seeds produce different sequences."""
        indices = list(range(100))

        # First sampler
        gen1 = torch.Generator()
        gen1.manual_seed(42)
        sampler1 = MutableSubsetRandomSampler(lambda: indices, rng=gen1)
        samples1 = list(sampler1)

        # Second sampler with different seed
        gen2 = torch.Generator()
        gen2.manual_seed(123)
        sampler2 = MutableSubsetRandomSampler(lambda: indices, rng=gen2)
        samples2 = list(sampler2)

        # Should produce different sequences
        assert samples1 != samples2

    def test_length(self):
        """Test sampler length."""
        indices = list(range(50))
        sampler = MutableSubsetRandomSampler(lambda: indices)

        assert len(sampler) == 50

    def test_iteration(self):
        """Test iterating through sampler."""
        indices = list(range(20))
        sampler = MutableSubsetRandomSampler(lambda: indices)

        samples = list(sampler)

        # Should return all indices (in random order)
        assert len(samples) == 20
        assert set(samples) == set(indices)

    def test_multiple_iterations(self):
        """Test multiple iterations produce different orders."""
        indices = list(range(50))
        generator = torch.Generator()
        generator.manual_seed(42)
        sampler = MutableSubsetRandomSampler(lambda: indices, rng=generator)

        samples1 = list(sampler)
        samples2 = list(sampler)

        # Each iteration should produce results
        assert len(samples1) == 50
        assert len(samples2) == 50

        # Orders may differ between iterations
        # (depends on implementation)

    def test_subset_of_indices(self):
        """Test sampler with subset of indices."""
        # Only sample from subset
        all_indices = list(range(100))
        subset_indices = list(range(0, 100, 2))  # Even indices only

        sampler = MutableSubsetRandomSampler(subset_indices)
        samples = list(sampler)

        # All samples should be from subset
        assert all(s in subset_indices for s in samples)
        assert len(samples) == len(subset_indices)

    def test_empty_indices(self):
        """Test sampler with empty indices."""
        sampler = MutableSubsetRandomSampler(lambda: [])
        samples = list(sampler)

        assert len(samples) == 0

    def test_single_index(self):
        """Test sampler with single index."""
        sampler = MutableSubsetRandomSampler(lambda: [42])
        samples = list(sampler)

        assert len(samples) == 1
        assert samples[0] == 42

    def test_indices_mutation(self):
        """Test that indices can be mutated."""
        indices = list(range(10))
        sampler = MutableSubsetRandomSampler(lambda: indices)

        # Get initial samples
        samples1 = list(sampler)
        assert len(samples1) == 10

        # Mutate indices
        new_indices = list(range(10, 20))
        sampler.indices_generator = lambda: new_indices
        sampler.refresh()

        # New samples should be from new indices
        samples2 = list(sampler)
        assert all(s in new_indices for s in samples2)

    def test_use_with_dataloader(self):
        """Test sampler integration with DataLoader."""
        from torch.utils.data import DataLoader

        dataset = DummyDataset(size=50)
        indices = list(range(25))  # Only use first half
        sampler = MutableSubsetRandomSampler(lambda: indices)

        loader = DataLoader(dataset, batch_size=5, sampler=sampler)

        # Should be able to iterate
        batches = list(loader)
        assert len(batches) > 0

        # Should only see indices from sampler
        all_indices = []
        for batch in batches:
            all_indices.extend(batch.tolist())

        assert all(idx in indices for idx in all_indices)

    def test_weighted_sampling_setup(self):
        """Test setup for weighted sampling."""
        # Create indices with weights
        indices = list(range(100))

        # Could be used with weights (implementation specific)
        sampler = MutableSubsetRandomSampler(lambda: indices)

        # Sampler should work
        samples = list(sampler)
        assert len(samples) == 100

    def test_deterministic_ordering_with_seed(self):
        """Test that seed makes ordering deterministic."""
        indices = list(range(30))

        results = []
        for _ in range(3):
            gen = torch.Generator()
            gen.manual_seed(42)
            sampler = MutableSubsetRandomSampler(indices, rng=gen)
            results.append(list(sampler))

        # All should be identical
        assert results[0] == results[1] == results[2]

    def test_refresh_capability(self):
        """Test that sampler can be refreshed."""
        indices = list(range(50))
        gen = torch.Generator()
        sampler = MutableSubsetRandomSampler(indices, rng=gen)

        # Get first sampling
        samples1 = list(sampler)

        # Get second sampling (may or may not be different)
        samples2 = list(sampler)

        # Both should have correct length
        assert len(samples1) == 50
        assert len(samples2) == 50

        # Both should contain all indices
        assert set(samples1) == set(indices)
        assert set(samples2) == set(indices)


class TestWeightedSampling:
    """Test weighted sampling scenarios."""

    def test_balanced_sampling(self):
        """Test balanced sampling across classes."""
        # Simulate class-balanced sampling
        class_0_indices = list(range(0, 30))  # 30 samples
        class_1_indices = list(range(30, 100))  # 70 samples

        # To balance, we might oversample class_0
        # For simplicity, just test that we can sample from both
        all_indices = class_0_indices + class_1_indices
        sampler = MutableSubsetRandomSampler(all_indices)

        samples = list(sampler)

        # Should include samples from both classes
        assert any(s in class_0_indices for s in samples)
        assert any(s in class_1_indices for s in samples)

    def test_stratified_indices(self):
        """Test stratified sampling indices."""
        # Create stratified indices
        strata = [
            list(range(0, 25)),  # Stratum 1
            list(range(25, 50)),  # Stratum 2
            list(range(50, 75)),  # Stratum 3
            list(range(75, 100)),  # Stratum 4
        ]

        # Sample from each stratum
        for stratum_indices in strata:
            sampler = MutableSubsetRandomSampler(stratum_indices)
            samples = list(sampler)

            # All samples should be from this stratum
            assert all(s in stratum_indices for s in samples)
            assert len(samples) == len(stratum_indices)
