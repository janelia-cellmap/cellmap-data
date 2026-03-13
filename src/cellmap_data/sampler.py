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
       randomly), sample a crop annotating it, yield that crop index, then
       increment counts for *all* classes that crop annotates.

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

            # Sample a crop that annotates this class
            crop_idx = int(self.rng.choice(self.class_to_crops[target_class]))

            # Increment counts for all classes this crop annotates
            annotated = np.where(self.crop_class_matrix[crop_idx])[0]
            class_counts[annotated] += 1.0

            yield crop_idx

    def __len__(self) -> int:
        return self.samples_per_epoch
