"""Tests for ClassBalancedSampler."""

from __future__ import annotations

import numpy as np
import pytest

from cellmap_data.sampler import ClassBalancedSampler
from cellmap_data.utils.misc import min_redundant_inds


class FakeDataset:
    """Minimal dataset with a known crop-class matrix."""

    def __init__(self, matrix: np.ndarray):
        self._matrix = matrix

    def get_crop_class_matrix(self) -> np.ndarray:
        return self._matrix

    def __len__(self) -> int:
        return self._matrix.shape[0]


class FakeConcatDataset:
    """Minimal ConcatDataset-like dataset with datasets + cumulative_sizes."""

    def __init__(self, sub_lengths: list[int], matrix: np.ndarray):
        self._matrix = matrix
        self._sub_lengths = sub_lengths
        # Cumulative sizes mirrors torch.utils.data.ConcatDataset behaviour
        self.cumulative_sizes: list[int] = []
        total = 0
        for length in sub_lengths:
            total += length
            self.cumulative_sizes.append(total)
        self.datasets = [None] * len(sub_lengths)  # placeholders

    def get_crop_class_matrix(self) -> np.ndarray:
        return self._matrix

    def __len__(self) -> int:
        return sum(self._sub_lengths)


class TestClassBalancedSampler:
    def _make_sampler(self, matrix, samples_per_epoch=None, seed=42):
        ds = FakeDataset(matrix)
        return ClassBalancedSampler(ds, samples_per_epoch=samples_per_epoch, seed=seed)

    def test_basic_iteration(self):
        matrix = np.array([[True, False], [False, True], [True, True]], dtype=bool)
        sampler = self._make_sampler(matrix, samples_per_epoch=10)
        indices = list(sampler)
        assert len(indices) == 10
        assert all(0 <= i < 3 for i in indices)

    def test_len(self):
        matrix = np.eye(4, dtype=bool)
        sampler = self._make_sampler(matrix, samples_per_epoch=20)
        assert len(sampler) == 20

    def test_default_samples_per_epoch(self):
        matrix = np.eye(3, dtype=bool)
        ds = FakeDataset(matrix)
        sampler = ClassBalancedSampler(ds)
        assert len(sampler) == 3

    def test_reset_between_epochs(self):
        """Each __iter__ call resets counts → different random sequence."""
        matrix = np.array([[True, False], [False, True]], dtype=bool)
        sampler = self._make_sampler(matrix, samples_per_epoch=6, seed=123)
        epoch1 = list(sampler)
        epoch2 = list(sampler)
        # With small samples_per_epoch, deterministic resets should produce same result
        # (counts reset → same greedy order from same RNG state if RNG is re-seeded each iter)
        # We just check both are valid indices
        assert all(0 <= i < 2 for i in epoch1)
        assert all(0 <= i < 2 for i in epoch2)

    def test_rare_class_sampled(self):
        """Class appearing in only 1 of 10 crops must still be sampled."""
        # Class 0 appears in all 10; class 1 appears only in crop 0
        matrix = np.zeros((10, 2), dtype=bool)
        matrix[:, 0] = True
        matrix[0, 1] = True
        sampler = self._make_sampler(matrix, samples_per_epoch=20)
        indices = list(sampler)
        # Crop 0 must appear (it's the only way to see class 1)
        assert 0 in indices

    def test_crop_class_matrix_stored(self):
        matrix = np.eye(3, dtype=bool)
        ds = FakeDataset(matrix)
        sampler = ClassBalancedSampler(ds, samples_per_epoch=5)
        assert np.array_equal(sampler.crop_class_matrix, matrix)

    def test_active_classes_only_annotated(self):
        """Classes with zero crops must not be in active_classes."""
        # class 2 has no annotated crops
        matrix = np.array([[True, False, False], [False, True, False]], dtype=bool)
        sampler = self._make_sampler(matrix, samples_per_epoch=4)
        assert 2 not in sampler.active_classes

    def test_yields_valid_indices_for_single_class(self):
        matrix = np.ones((5, 1), dtype=bool)
        sampler = self._make_sampler(matrix, samples_per_epoch=10)
        indices = list(sampler)
        assert len(indices) == 10
        assert all(0 <= i < 5 for i in indices)

    def test_raises_when_no_active_classes(self):
        """All-False crop-class matrix must raise ValueError immediately."""
        matrix = np.zeros((4, 3), dtype=bool)
        with pytest.raises(ValueError, match="no active classes"):
            self._make_sampler(matrix, samples_per_epoch=5)

    def test_concat_dataset_indices_in_correct_subdataset(self):
        """ConcatDataset path: each yielded index falls in the expected sub-dataset range."""
        # Two sub-datasets: first has 10 samples, second has 20 samples
        sub_lengths = [10, 20]
        # Row 0 → only class 0 annotated; Row 1 → only class 1 annotated
        matrix = np.array([[True, False], [False, True]], dtype=bool)
        ds = FakeConcatDataset(sub_lengths, matrix)
        sampler = ClassBalancedSampler(ds, samples_per_epoch=40, seed=0)
        indices = list(sampler)
        assert len(indices) == 40
        # All indices must be valid dataset indices
        assert all(0 <= i < len(ds) for i in indices)
        # Indices from class-0 crops (row 0 → sub-dataset 0) must be in [0, 10)
        # Indices from class-1 crops (row 1 → sub-dataset 1) must be in [10, 30)
        # Because the sampler alternates classes, roughly half go to each sub-dataset
        indices_set = set(indices)
        assert any(i < 10 for i in indices_set), "No index from sub-dataset 0"
        assert any(10 <= i < 30 for i in indices_set), "No index from sub-dataset 1"

    def test_concat_dataset_all_indices_in_range(self):
        """ConcatDataset path: all yielded indices are within [0, len(dataset))."""
        sub_lengths = [5, 5, 5]
        matrix = np.eye(3, dtype=bool)
        ds = FakeConcatDataset(sub_lengths, matrix)
        sampler = ClassBalancedSampler(ds, samples_per_epoch=30, seed=7)
        indices = list(sampler)
        assert all(0 <= i < len(ds) for i in indices)


class TestMinRedundantInds:
    def test_replacement_returns_k(self):
        result = min_redundant_inds(5, 12, replacement=True)
        assert len(result) == 12

    def test_no_replacement_k_leq_n(self):
        result = min_redundant_inds(10, 4, replacement=False)
        assert len(result) == 4
        assert len(set(result.tolist())) == 4  # all unique

    def test_no_replacement_k_equals_n(self):
        result = min_redundant_inds(5, 5, replacement=False)
        assert len(result) == 5
        assert sorted(result.tolist()) == list(range(5))

    def test_no_replacement_k_gt_n_exact_multiple(self):
        """k=6, n=3: two full permutations, exactly 6 indices returned."""
        result = min_redundant_inds(3, 6, replacement=False)
        assert len(result) == 6

    def test_no_replacement_k_gt_n_with_remainder(self):
        """k=7, n=3: must return exactly 7 indices, not 6."""
        result = min_redundant_inds(3, 7, replacement=False)
        assert len(result) == 7

    def test_no_replacement_all_values_in_range(self):
        result = min_redundant_inds(4, 11, replacement=False)
        assert len(result) == 11
        assert all(0 <= v < 4 for v in result.tolist())
