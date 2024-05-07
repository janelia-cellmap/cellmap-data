from typing import Callable, Iterable, Optional, Sequence
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
    _weighted_sampler: Optional[WeightedRandomSampler]
    _class_counts: dict[str, int] = {}

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
        self._weighted_sampler = None

    def __repr__(self) -> str:
        out_string = f"CellMapMultiDataset(["
        for dataset in self.datasets:
            out_string += f"\n\t{dataset},"
        out_string += "\n])"
        return out_string

    @property
    def class_counts(self):
        if not hasattr(self, "_class_counts"):
            class_counts = {}
            for c in self.classes:
                class_counts[c] = {}
                total: int = 0
                for ds in self.datasets:
                    total += ds.class_counts["totals"][c]
                    class_counts[c][ds] = ds.class_counts["totals"][c]
                class_counts[c]["total"] = total
            self._class_counts = class_counts
        return self._class_counts

    def to(self, device: str):
        for dataset in self.datasets:
            dataset.to(device)
        return self

    def weighted_sampler(
        self, batch_size: int = 1, rng: Optional[torch.Generator] = None
    ):
        if self._weighted_sampler is None:
            # TODO: calculate weights for each sample
            sample_weights = self.get_sample_weights()

            self._weighted_sampler = WeightedRandomSampler(
                sample_weights, batch_size, replacement=False, generator=rng
            )
        return self._weighted_sampler

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
            dataset_weights = list(self.get_dataset_weights().values())

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

    def get_class_weights(self):
        """
        Returns the class weights for the multi-dataset based on the number of samples in each class.
        """
        if len(self.classes) > 1:
            class_counts = {c: 0 for c in self.classes}
            class_count_sum = 0
            for dataset in self.datasets:
                for c in self.classes:
                    class_counts[c] += dataset.class_counts["totals"][c]
                    class_count_sum += dataset.class_counts["totals"][c]

            class_weights = {
                c: 1 - (class_counts[c] / class_count_sum) for c in self.classes
            }
        else:
            class_weights = {self.classes[0]: 0.1}  # less than 1 to avoid overflow
        return class_weights

    def get_dataset_weights(self):
        """
        Returns the weights for each dataset in the multi-dataset based on the number of samples in each dataset.
        """
        class_weights = self.get_class_weights()

        dataset_weights = {}
        for dataset in self.datasets:
            dataset_weight = np.sum(
                [
                    dataset.class_counts["totals"][c] * class_weights[c]
                    for c in self.classes
                ]
            )
            dataset_weights[dataset] = dataset_weight
        return dataset_weights

    def get_sample_weights(self):
        """
        Returns the weights for each sample in the multi-dataset based on the number of samples in each dataset.
        """

        dataset_weights = self.get_dataset_weights()
        sample_weights = []
        for dataset, dataset_weight in dataset_weights.items():
            sample_weights += [dataset_weight] * len(dataset)
        return sample_weights

    def get_validation_indices(self) -> Sequence[int]:
        """
        Returns the indices of the validation set for each dataset in the multi-dataset.
        """
        validation_indices = []
        index_offset = 0
        for dataset in self.datasets:
            validation_indices.extend(dataset.get_validation_indices())
            index_offset += len(dataset)
        return validation_indices

    def get_indices(self, chunk_size: dict[str, int]) -> Sequence[int]:
        """Returns the indices of the dataset that will tile the dataset according to the chunk_size."""
        indices = []
        index_offset = 0
        for dataset in self.datasets:
            indices.append(dataset.get_indices(chunk_size))
            index_offset += len(dataset)
        return indices

    def set_raw_value_transforms(self, transforms: Callable):
        """Sets the raw value transforms for each dataset in the multi-dataset."""
        for dataset in self.datasets:
            dataset.set_raw_value_transforms(transforms)

    def set_target_value_transforms(self, transforms: Callable):
        """Sets the target value transforms for each dataset in the multi-dataset."""
        for dataset in self.datasets:
            dataset.set_target_value_transforms(transforms)


# TODO: make "last" and "current" variable names consistent
