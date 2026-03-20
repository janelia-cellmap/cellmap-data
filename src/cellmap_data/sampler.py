"""Class-balanced sampler for CellMap datasets.

Problem: Some classes appear in 200+ crops, others in <10.
Uniform crop sampling means rare classes barely appear during training.

Solution: At each step, pick the least-seen class so far, then sample
a crop that annotates it.  All active classes receive roughly equal
representation over an epoch regardless of annotation frequency.
"""

from __future__ import annotations

from typing import Iterator, Optional

import numpy as np
from torch.utils.data import Sampler


class ClassBalancedSampler(Sampler):
    """Greedy class-balanced sampler.

    Algorithm:
    1. Build a crop-class matrix from the dataset:
       ``dataset.get_crop_class_matrix()`` → ``bool[n_crops, n_classes]``.
    2. Maintain running counts of how many times each class has been seen.
    3. At each step: pick the class with the lowest count (ties broken
       randomly), sample a matrix row (crop) annotating it, map that row to
       an actual dataset sample index, and yield the sample index.  Then
       increment counts for *all* classes that row annotates.

    This guarantees rare classes get sampled as often as common ones.

    The sampler resets ``class_counts`` to zero at the start of each
    ``__iter__`` call, so no ``refresh()`` is needed between epochs.

    Parameters
    ----------
    dataset:
        Must implement ``get_crop_class_matrix() -> np.ndarray``.
    samples_per_epoch:
        Number of samples to yield per epoch.  Defaults to ``len(dataset)``.
    seed:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        dataset,
        samples_per_epoch: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch or len(dataset)
        self.rng = np.random.default_rng(seed)

        # [n_crops × n_classes] boolean matrix
        self.crop_class_matrix: np.ndarray = dataset.get_crop_class_matrix()
        self.n_crops, self.n_classes = self.crop_class_matrix.shape

        # Pre-compute per-class crop lists
        self.class_to_crops: dict[int, np.ndarray] = {}
        for c in range(self.n_classes):
            indices = np.where(self.crop_class_matrix[:, c])[0]
            if len(indices) > 0:
                self.class_to_crops[c] = indices

        self.active_classes: list[int] = sorted(self.class_to_crops.keys())
        if not self.active_classes:
            raise ValueError(
                "ClassBalancedSampler: no active classes found in crop-class "
                "matrix. This can occur when all requested classes are only "
                "represented by empty crops (e.g., EmptyImage)."
            )

    def __iter__(self) -> Iterator[int]:
        class_counts = np.zeros(self.n_classes, dtype=np.float64)

        for _ in range(self.samples_per_epoch):
            # Find least-seen active class; break ties randomly
            active_counts = np.array([class_counts[c] for c in self.active_classes])
            min_count = active_counts.min()
            tied = [
                self.active_classes[i]
                for i, v in enumerate(active_counts)
                if v == min_count
            ]
            target_class = int(self.rng.choice(tied))

            # Sample a matrix row (crop) that annotates this class
            row_idx = int(self.rng.choice(self.class_to_crops[target_class]))

            # Increment counts for all classes this row annotates
            annotated = np.where(self.crop_class_matrix[row_idx])[0]
            class_counts[annotated] += 1.0

            # Map matrix row (dataset-level row) to an actual sample index.
            # If n_crops equals len(dataset), the row index IS the sample index.
            if self.n_crops == len(self.dataset):
                sample_idx = row_idx
            elif hasattr(self.dataset, "datasets") and hasattr(
                self.dataset, "cumulative_sizes"
            ):
                # ConcatDataset / CellMapMultiDataset: each row corresponds
                # to one sub-dataset; pick a random sample within that sub-dataset.
                cumulative_sizes = self.dataset.cumulative_sizes
                if row_idx < len(cumulative_sizes):
                    start = int(cumulative_sizes[row_idx - 1]) if row_idx > 0 else 0
                    end = int(cumulative_sizes[row_idx])
                else:
                    raise ValueError(
                        "ClassBalancedSampler: crop index out of range for "
                        "ConcatDataset/CellMapMultiDataset mapping. "
                        f"row_idx={row_idx}, n_subdatasets={len(cumulative_sizes)}"
                    )
                if start >= end or end > len(self.dataset):
                    raise ValueError(
                        "ClassBalancedSampler: invalid sub-dataset slice computed "
                        "from cumulative_sizes for row index "
                        f"{row_idx}: start={start}, end={end}, "
                        f"len(dataset)={len(self.dataset)}"
                    )
                sample_idx = int(self.rng.integers(start, end))
            else:
                # Generic fallback: partition [0, len(dataset)) into n_crops
                # contiguous segments and sample within this row's segment.
                total = len(self.dataset)
                if self.n_crops <= 1 or total <= 0:
                    start, end = 0, max(total, 1)
                else:
                    base = total // self.n_crops
                    remainder = total % self.n_crops
                    if row_idx < remainder:
                        start = row_idx * (base + 1)
                        end = start + (base + 1)
                    else:
                        start = remainder * (base + 1) + (row_idx - remainder) * base
                        end = start + base
                    if start >= end or end > total:
                        start, end = 0, total
                sample_idx = int(self.rng.integers(start, end))

            yield sample_idx

    def __len__(self) -> int:
        return self.samples_per_epoch
