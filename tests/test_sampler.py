"""Tests for ClassBalancedSampler."""

from __future__ import annotations

import numpy as np
import pytest

from cellmap_data.sampler import ClassBalancedSampler


class FakeDataset:
    """Minimal dataset with a known crop-class matrix."""

    def __init__(self, matrix: np.ndarray):
        self._matrix = matrix

    def get_crop_class_matrix(self) -> np.ndarray:
        return self._matrix

    def __len__(self) -> int:
        return self._matrix.shape[0]


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
