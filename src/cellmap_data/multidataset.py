from typing import Any, Callable, Mapping, Optional, Sequence
import numpy as np
import torch
from torch.utils.data import ConcatDataset, WeightedRandomSampler
from tqdm import tqdm

from .dataset import CellMapDataset


class CellMapMultiDataset(ConcatDataset):
    """
    This class is used to combine multiple datasets into a single dataset. It is a subclass of PyTorch's ConcatDataset. It maintains the same API as the ConcatDataset class. It retrieves raw and groundtruth data from multiple CellMapDataset objects. See the CellMapDataset class for more information on the dataset object.

    Attributes:
        classes: Sequence[str]
            The classes in the dataset.
        input_arrays: Mapping[str, Mapping[str, Sequence[int | float]]]
            The input arrays for each dataset in the multi-dataset.
        target_arrays: Mapping[str, Mapping[str, Sequence[int | float]]]
            The target arrays for each dataset in the multi-dataset.
        datasets: Sequence[CellMapDataset]
            The datasets to be combined into the multi-dataset.

    Methods:
        to(device: str | torch.device) -> "CellMapMultiDataset":
            Moves the multi-dataset to the specified device.
        get_weighted_sampler(batch_size: int = 1, rng: Optional[torch.Generator] = None) -> WeightedRandomSampler:
            Returns a weighted random sampler for the multi-dataset.
        get_subset_random_sampler(num_samples: int, weighted: bool = True, rng: Optional[torch.Generator] = None) -> torch.utils.data.SubsetRandomSampler:
            Returns a random sampler that samples num_samples from the multi-dataset.
        get_indices(chunk_size: Mapping[str, int]) -> Sequence[int]:
            Returns the indices of the multi-dataset that will tile all of the datasets according to the requested chunk_size.
        set_raw_value_transforms(transforms: Callable) -> None:
            Sets the raw value transforms for each dataset in the multi-dataset.
        set_target_value_transforms(transforms: Callable) -> None:
            Sets the target value transforms for each dataset in the multi-dataset.
        set_spatial_transforms(spatial_transforms: Mapping[str, Any] | None) -> None:
            Sets the spatial transforms for each dataset in the multi-dataset.

    Properties:
        class_counts: Mapping[str, float]
            Returns the number of samples in each class for each dataset in the multi-dataset, as well as the total number of samples in each class.
        class_weights: Mapping[str, float]
            Returns the class weights for the multi-dataset based on the number of samples in each class.
        dataset_weights: Mapping[CellMapDataset, float]
            Returns the weights for each dataset in the multi-dataset based on the number of samples of each class in each dataset.
        sample_weights: Sequence[float]
            Returns the weights for each sample in the multi-dataset based on the number of samples in each dataset.
        validation_indices: Sequence[int]
            Returns the indices of the validation set for each dataset in the multi-dataset.

    """

    def __init__(
        self,
        classes: Sequence[str] | None,
        input_arrays: Mapping[str, Mapping[str, Sequence[int | float]]],
        target_arrays: Mapping[str, Mapping[str, Sequence[int | float]]] | None,
        datasets: Sequence[CellMapDataset],
    ) -> None:
        super().__init__(datasets)
        self.input_arrays = input_arrays
        self.target_arrays = target_arrays if target_arrays is not None else {}
        self.classes = classes if classes is not None else []
        self.datasets = datasets

    def __repr__(self) -> str:
        out_string = f"CellMapMultiDataset(["
        for dataset in self.datasets:
            out_string += f"\n\t{dataset},"
        out_string += "\n])"
        return out_string

    @property
    def class_counts(self) -> dict[str, float]:
        """
        Returns the number of samples in each class for each dataset in the multi-dataset, as well as the total number of samples in each class.
        """
        try:
            return self._class_counts
        except AttributeError:
            class_counts = {c: 0.0 for c in self.classes}
            class_counts.update({c + "_bg": 0.0 for c in self.classes})
            print("Gathering class counts...")
            for ds in tqdm(self.datasets):
                for c in self.classes:
                    if c in ds.class_counts["totals"]:
                        class_counts[c] += ds.class_counts["totals"][c]
                        class_counts[c + "_bg"] += ds.class_counts["totals"][c + "_bg"]
            self._class_counts = class_counts
            return self._class_counts

    @property
    def class_weights(self) -> Mapping[str, float]:
        """
        Returns the class weights for the multi-dataset based on the number of samples in each class.
        """
        # TODO: review this implementation
        try:
            return self._class_weights
        except AttributeError:
            class_weights = {
                c: (
                    self.class_counts[c + "_bg"] / self.class_counts[c]
                    if self.class_counts[c] != 0
                    else 1
                )
                for c in self.classes
            }
            self._class_weights = class_weights
            return self._class_weights

    @property
    def dataset_weights(self) -> Mapping[CellMapDataset, float]:
        """
        Returns the weights for each dataset in the multi-dataset based on the number of samples in each dataset.
        """
        try:
            return self._dataset_weights
        except AttributeError:
            dataset_weights = {}
            for dataset in self.datasets:
                if len(self.classes) == 0:
                    dataset_weight = len(dataset)
                else:
                    dataset_weight = np.sum(
                        [
                            dataset.class_counts["totals"][c] * self.class_weights[c]
                            for c in self.classes
                        ]
                    )
                dataset_weights[dataset] = dataset_weight
            self._dataset_weights = dataset_weights
            return self._dataset_weights

    @property
    def sample_weights(self) -> Sequence[float]:
        """
        Returns the weights for each sample in the multi-dataset based on the number of samples in each dataset.
        """
        try:
            return self._sample_weights
        except AttributeError:
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
        try:
            return self._validation_indices
        except AttributeError:
            indices = []
            for i, dataset in enumerate(self.datasets):
                try:
                    if i == 0:
                        offset = 0
                    else:
                        offset = self.cumulative_sizes[i - 1]
                    sample_indices = np.array(dataset.validation_indices) + offset
                    indices.extend(list(sample_indices))
                except AttributeError:
                    UserWarning(
                        f"Unable to get validation indices for dataset {dataset}\n skipping"
                    )
            self._validation_indices = indices
            return self._validation_indices

    def to(
        self, device: str | torch.device, non_blocking: bool = True
    ) -> "CellMapMultiDataset":
        for dataset in self.datasets:
            dataset.to(device, non_blocking=non_blocking)
        return self

    def get_weighted_sampler(
        self, batch_size: int = 1, rng: Optional[torch.Generator] = None
    ) -> WeightedRandomSampler:
        return WeightedRandomSampler(
            self.sample_weights, batch_size, replacement=False, generator=rng
        )

    def get_subset_random_sampler(
        self,
        num_samples: int,
        weighted: bool = True,
        rng: Optional[torch.Generator] = None,
    ) -> torch.utils.data.SubsetRandomSampler:
        """
        Returns a random sampler that samples num_samples from the dataset.
        """
        if not weighted:
            return torch.utils.data.SubsetRandomSampler(
                torch.randint(0, len(self) - 1, [num_samples], generator=rng),
                generator=rng,
            )
        else:
            # TODO: Add cacpacity for curriculum learning
            dataset_weights = torch.tensor(
                [self.dataset_weights[ds] for ds in self.datasets]
            )
            dataset_weights[dataset_weights < 0.1] = 0.1

            datasets_sampled = torch.multinomial(
                torch.as_tensor(dataset_weights, dtype=float),
                num_samples,
                replacement=True,
            )
            indices = []
            index_offset = 0
            for i, dataset in enumerate(self.datasets):
                if len(dataset) == 0:
                    RuntimeWarning(f"Dataset {dataset} has no samples, skipping")
                    continue
                count = (datasets_sampled == i).sum().item()
                if count == 0:
                    continue
                if len(dataset) == 1:
                    dataset_indices = torch.tensor([0] * count)
                else:
                    dataset_indices = torch.randint(
                        0, len(dataset) - 1, [count], generator=rng
                    )
                indices.append(dataset_indices + index_offset)
                index_offset += len(dataset)
            indices = torch.cat(indices).flatten()
            indices = indices[torch.randperm(len(indices), generator=rng)]
            return torch.utils.data.SubsetRandomSampler(indices, generator=rng)

    def get_indices(self, chunk_size: Mapping[str, int]) -> Sequence[int]:
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

    def set_raw_value_transforms(self, transforms: Callable) -> None:
        """Sets the raw value transforms for each dataset in the multi-dataset."""
        for dataset in self.datasets:
            dataset.set_raw_value_transforms(transforms)

    def set_target_value_transforms(self, transforms: Callable) -> None:
        """Sets the target value transforms for each dataset in the multi-dataset."""
        for dataset in self.datasets:
            dataset.set_target_value_transforms(transforms)

    def set_spatial_transforms(
        self, spatial_transforms: Mapping[str, Any] | None
    ) -> None:
        """Sets the raw value transforms for each dataset in the training multi-dataset."""
        for dataset in self.datasets:
            dataset.spatial_transforms = spatial_transforms

    @staticmethod
    def empty() -> "CellMapMultiDataset":
        """Creates an empty dataset."""
        empty_dataset = CellMapMultiDataset([], {}, {}, [CellMapDataset.empty()])
        empty_dataset.classes = []
        empty_dataset._class_counts = {}
        empty_dataset._class_weights = {}
        empty_dataset._validation_indices = []

        return empty_dataset
