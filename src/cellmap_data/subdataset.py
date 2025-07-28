import functools
from typing import Any, Callable, Optional, Sequence
import torch
from torch.utils.data import Subset

from .mutable_sampler import MutableSubsetRandomSampler
from .utils.sampling import min_redundant_inds
from .dataset import CellMapDataset

from .multidataset import CellMapMultiDataset


class CellMapSubset(Subset):
    """A PyTorch Subset wrapper for CellMap datasets with consistent API.

    This class subclasses PyTorch Subset to provide a unified interface for working
    with subsets of CellMapDataset or CellMapMultiDataset objects. It maintains
    compatibility with PyTorch's Subset API while preserving access to CellMap-specific
    properties and methods like class information and class counts.

    The subset allows efficient access to a selected portion of a larger dataset
    without duplicating data in memory, making it suitable for train/validation
    splits and other data partitioning workflows.

    Args:
        dataset: The parent dataset to create a subset from.
            Must be an instance of CellMapDataset or CellMapMultiDataset.
        indices: Indices from the parent dataset to include in this subset.
            Must be valid indices within the range of the parent dataset.
    """

    def __init__(
        self,
        dataset: CellMapDataset | CellMapMultiDataset,
        indices: Sequence[int],
    ) -> None:
        """Initialize a CellMapSubset with specified dataset and indices.

        Args:
            dataset: The parent dataset from which to create the subset.
                Must implement the standard dataset interface.
            indices: List or array of indices to include in the subset.
                All indices must be valid for the parent dataset.

        Raises:
            IndexError: If any index in indices is out of range for the parent dataset.
            TypeError: If dataset is not a CellMapDataset or CellMapMultiDataset instance.
        """
        super().__init__(dataset, indices)

    @property
    def classes(self) -> Sequence[str]:
        return self.dataset.classes

    @property
    def class_counts(self) -> dict[str, float]:
        return self.dataset.class_counts

    @property
    def class_weights(self) -> dict[str, float]:
        return self.dataset.class_weights

    @property
    def validation_indices(self) -> Sequence[int]:
        return self.dataset.validation_indices

    def to(self, device, non_blocking: bool = True) -> "CellMapSubset":
        self.dataset.to(device, non_blocking=non_blocking)
        return self

    def set_raw_value_transforms(self, transforms: Callable) -> None:
        """Sets the raw value transforms for the subset dataset."""
        self.dataset.set_raw_value_transforms(transforms)

    def set_target_value_transforms(self, transforms: Callable) -> None:
        """Sets the target value transforms for the subset dataset."""
        self.dataset.set_target_value_transforms(transforms)

    def get_random_subset_indices(
        self, num_samples: int, rng: Optional[torch.Generator] = None, **kwargs: Any
    ) -> Sequence[int]:
        inds = min_redundant_inds(len(self.indices), num_samples, rng=rng)
        return torch.tensor(self.indices, dtype=torch.long)[inds].tolist()

    def get_subset_random_sampler(
        self,
        num_samples: int,
        rng: Optional[torch.Generator] = None,
        **kwargs: Any,
    ) -> MutableSubsetRandomSampler:
        """
        Returns a random sampler that yields exactly `num_samples` indices from this subset.
        - If `num_samples` ≤ total number of available indices, samples without replacement.
        - If `num_samples` > total number of available indices, samples with replacement using repeated shuffles to minimize duplicates.
        """

        indices_generator = functools.partial(
            self.get_random_subset_indices, num_samples, rng, **kwargs
        )

        return MutableSubsetRandomSampler(
            indices_generator,
            rng=rng,
        )
