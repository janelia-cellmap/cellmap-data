import functools
from typing import Any, Callable, Optional, Sequence
import torch
from torch.utils.data import Subset

from .mutable_sampler import MutableSubsetRandomSampler
from .utils.sampling import min_redundant_inds
from .dataset import CellMapDataset

from .multidataset import CellMapMultiDataset


class CellMapSubset(Subset):
    """
    This subclasses PyTorch Subset to wrap a CellMapDataset or CellMapMultiDataset object under a common API, which can be used for dataloading. It maintains the same API as the Subset class. It retrieves raw and groundtruth data from a CellMapDataset or CellMapMultiDataset object.
    """

    def __init__(
        self,
        dataset: CellMapDataset | CellMapMultiDataset,
        indices: Sequence[int],
    ) -> None:
        """
        Args:
            dataset: CellMapDataset | CellMapMultiDataset
                The dataset to be subsetted.
            indices: Sequence[int]
                The indices of the dataset to be used as the subset.
        """
        super().__init__(dataset, indices)

    @property
    def classes(self) -> Sequence[str]:
        """The classes in the dataset."""
        return self.dataset.classes

    @property
    def class_counts(self) -> dict[str, float]:
        """The number of samples in each class in the dataset normalized by resolution."""
        return self.dataset.class_counts

    @property
    def class_weights(self) -> dict[str, float]:
        """The class weights for the dataset based on the number of samples in each class."""
        return self.dataset.class_weights

    @property
    def validation_indices(self) -> Sequence[int]:
        """The indices of the validation set."""
        return self.dataset.validation_indices

    def to(self, device, non_blocking: bool = True) -> "CellMapSubset":
        """Move the dataset to the specified device."""
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
