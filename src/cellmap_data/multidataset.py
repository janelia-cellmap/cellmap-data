from typing import Any, Callable, Iterable, Optional, Sequence
import numpy as np
import torch
from torch.utils.data import ConcatDataset, WeightedRandomSampler

from .dataset import CellMapDataset


class CellMapMultiDataset(ConcatDataset):
    """
    This subclasses PyTorch Dataset to wrap multiple CellMapDataset objects under a common API, which can be used for dataloading. It maintains the same API as the Dataset class. It retrieves raw and groundtruth data from CellMapDataset objects.
    """

    classes: Sequence[str]
    input_arrays: dict[str, dict[str, Sequence[int | float]]]
    target_arrays: dict[str, dict[str, Sequence[int | float]]]
    datasets: Iterable[CellMapDataset]

    def __init__(
        self,
        classes: Sequence[str],
        input_arrays: dict[str, dict[str, Sequence[int | float]]],
        target_arrays: dict[str, dict[str, Sequence[int | float]]],
        datasets: Sequence[CellMapDataset],
    ):
        super().__init__(datasets)
        self.input_arrays = input_arrays
        self.target_arrays = target_arrays
        self.classes = classes
        self.datasets = datasets

    def __repr__(self) -> str:
        out_string = f"CellMapMultiDataset(["
        for dataset in self.datasets:
            out_string += f"\n\t{dataset},"
        out_string += "\n])"
        return out_string

    @property
    def class_counts(self):
        if (
            not hasattr(self, "_class_counts")
            or self._class_counts is None  # This should be overkill...
            or len(self._class_counts) == 0
        ):
            class_counts = {}
            for c in self.classes:
                class_counts[c] = {}
                total = 0.0
                for ds in self.datasets:
                    total += ds.class_counts["totals"][c]
                    class_counts[c][ds] = ds.class_counts["totals"][c]
                class_counts[c]["total"] = total
            self._class_counts = class_counts
        return self._class_counts

    @property
    def class_weights(self):
        """
        Returns the class weights for the multi-dataset based on the number of samples in each class.
        """
        if not hasattr(self, "_class_weights"):
            if len(self.classes) > 1:
                class_counts = {}
                class_count_sum = 0
                for c in self.classes:
                    class_counts[c] = self.class_counts[c]["total"]
                    class_count_sum += class_counts[c]

                class_weights = {
                    c: (
                        1 - (class_counts[c] / class_count_sum)
                        if class_counts[c] != class_count_sum
                        else 0.1
                    )
                    for c in self.classes
                }
            else:
                class_weights = {self.classes[0]: 0.1}  # less than 1 to avoid overflow
            self._class_weights = class_weights
        return self._class_weights

    @property
    def dataset_weights(self):
        """
        Returns the weights for each dataset in the multi-dataset based on the number of samples in each dataset.
        """
        if not hasattr(self, "_dataset_weights"):
            class_weights = self.class_weights

            dataset_weights = {}
            for dataset in self.datasets:
                dataset_weight = np.sum(
                    [
                        dataset.class_counts["totals"][c] * class_weights[c]
                        for c in self.classes
                    ]
                )
                dataset_weights[dataset] = dataset_weight
            self._dataset_weights = dataset_weights
        return self._dataset_weights

    @property
    def sample_weights(self):
        """
        Returns the weights for each sample in the multi-dataset based on the number of samples in each dataset.
        """
        if not hasattr(self, "_sample_weights"):
            dataset_weights = self.dataset_weights
            sample_weights = []
            for dataset, dataset_weight in dataset_weights.items():
                sample_weights += [dataset_weight] * len(dataset)
            self._sample_weights = sample_weights
        return self._sample_weights

    @property
    def validation_indices(self) -> Sequence[int]:
        """
        Returns the indices of the validation set for each dataset in the multi-dataset.
        """
        if not hasattr(self, "_validation_indices"):
            indices = []
            for i, dataset in enumerate(self.datasets):
                try:
                    if i == 0:
                        offset = 0
                    else:
                        offset = self.cummulative_sizes[i - 1]
                    sample_indices = np.array(dataset.validation_indices) + offset
                    indices.extend(list(sample_indices))
                except AttributeError:
                    UserWarning(
                        f"Unable to get validation indices for dataset {dataset}\n skipping"
                    )
            self._validation_indices = indices
        return self._validation_indices

    def to(self, device: str):
        for dataset in self.datasets:
            dataset.to(device)
        return self

    def get_weighted_sampler(
        self, batch_size: int = 1, rng: Optional[torch.Generator] = None
    ):
        return WeightedRandomSampler(
            self.sample_weights, batch_size, replacement=False, generator=rng
        )

    def get_subset_random_sampler(
        self,
        num_samples: int,
        weighted: bool = True,
        rng: Optional[torch.Generator] = None,
    ):
        """
        Returns a random sampler that samples num_samples from the dataset.
        """
        if not weighted:
            return torch.utils.data.SubsetRandomSampler(
                torch.randint(0, len(self) - 1, [num_samples], generator=rng),
                generator=rng,
            )
        else:
            dataset_weights = list(self.dataset_weights.values())

            datasets_sampled = torch.multinomial(
                torch.tensor(dataset_weights), num_samples, replacement=True
            )
            indices = []
            index_offset = 0
            for i, dataset in enumerate(self.datasets):
                if len(dataset) == 0:
                    continue
                count = (datasets_sampled == i).sum().item()
                dataset_indices = torch.randint(
                    0, len(dataset) - 1, [count], generator=rng
                )
                indices.append(dataset_indices + index_offset)
                index_offset += len(dataset)
            indices = torch.cat(indices).flatten()
            indices = indices[torch.randperm(len(indices), generator=rng)]
            return torch.utils.data.SubsetRandomSampler(indices, generator=rng)

    def get_indices(self, chunk_size: dict[str, int]) -> Sequence[int]:
        """Returns the indices of the dataset that will tile all of the datasets according to the chunk_size."""
        indices = []
        for i, dataset in enumerate(self.datasets):
            if i == 0:
                offset = 0
            else:
                offset = self.cummulative_sizes[i - 1]
            sample_indices = np.array(dataset.get_indices(chunk_size)) + offset
            indices.extend(list(sample_indices))
        return indices

    def set_raw_value_transforms(self, transforms: Callable):
        """Sets the raw value transforms for each dataset in the multi-dataset."""
        for dataset in self.datasets:
            dataset.set_raw_value_transforms(transforms)

    def set_target_value_transforms(self, transforms: Callable):
        """Sets the target value transforms for each dataset in the multi-dataset."""
        for dataset in self.datasets:
            dataset.set_target_value_transforms(transforms)

    def set_spatial_transforms(self, spatial_transforms: dict[str, Any] | None):
        """Sets the raw value transforms for each dataset in the training multi-dataset."""
        for dataset in self.datasets:
            dataset.spatial_transforms = spatial_transforms


# TODO: make "last" and "current" variable names consistent
