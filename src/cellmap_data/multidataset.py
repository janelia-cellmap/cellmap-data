import functools
from typing import Any, Callable, Mapping, Optional, Sequence
import numpy as np
import torch
from torch.utils.data import ConcatDataset, WeightedRandomSampler
from tqdm import tqdm
import logging

from .mutable_sampler import MutableSubsetRandomSampler
from .utils.sampling import min_redundant_inds
from .dataset import CellMapDataset

logger = logging.getLogger(__name__)


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
            Returns a nested dictionary containing the number of samples in each class for each dataset in the multi-dataset, with class-specific counts nested under a 'totals' key.
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

    def __reduce__(self):
        """
        Support pickling for multiprocessing DataLoader and spawned processes.
        """
        # These are the args __init__ needs:
        args = (self.classes, self.input_arrays, self.target_arrays, self.datasets)
        # Return: (callable, args_for_constructor, state_dict)
        return (self.__class__, args, self.__dict__)

    @property
    def has_data(self) -> bool:
        """
        Returns True if the multi-dataset has data, i.e., if it contains any datasets.
        """
        return len(self) > 0

    @property
    def class_counts(self) -> dict[str, dict[str, float]]:
        """
        Returns the number of samples in each class for each dataset in the multi-dataset, as well as the total number of samples in each class.
        """
        try:
            return self._class_counts
        except AttributeError:
            class_counts = {"totals": {c: 0.0 for c in self.classes}}
            class_counts["totals"].update({c + "_bg": 0.0 for c in self.classes})
            logger.info("Gathering class counts...")
            for ds in tqdm(self.datasets):
                for c in self.classes:
                    if c in ds.class_counts["totals"]:
                        class_counts["totals"][c] += ds.class_counts["totals"][c]
                        class_counts["totals"][c + "_bg"] += ds.class_counts["totals"][
                            c + "_bg"
                        ]
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
                    self.class_counts["totals"][c + "_bg"]
                    / self.class_counts["totals"][c]
                    if self.class_counts["totals"][c] != 0
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
                    # If no classes are defined, assign equal weight to all datasets
                    dataset_weight = 1.0
                else:
                    dataset_weight = np.sum(
                        [
                            dataset.class_counts["totals"][c] * self.class_weights[c]
                            for c in self.classes
                        ]
                    )
                    dataset_weight *= (1 / len(dataset)) if len(dataset) > 0 else 0
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

    def verify(self) -> bool:
        """
        Verifies that all datasets in the multi-dataset have the same classes and input/target array keys.
        """
        if len(self.datasets) == 0:
            return False

        n_verified_datasets = 0
        for dataset in self.datasets:
            n_verified_datasets += int(dataset.verify())
            try:
                assert (
                    dataset.classes == self.classes
                ), "All datasets must have the same classes."
                assert set(dataset.input_arrays.keys()) == set(
                    self.input_arrays.keys()
                ), "All datasets must have the same input arrays."
                if self.target_arrays is not None:
                    assert set(dataset.target_arrays.keys()) == set(
                        self.target_arrays.keys()
                    ), "All datasets must have the same target arrays."
            except AssertionError as e:
                logger.error(
                    f"Dataset {dataset} does not match the expected structure: {e}"
                )
                return False
        return n_verified_datasets > 0

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

    def get_random_subset_indices(
        self,
        num_samples: int,
        weighted: bool = True,
        rng: Optional[torch.Generator] = None,
    ) -> Sequence[int]:
        if not weighted:
            return min_redundant_inds(len(self), num_samples, rng=rng).tolist()
        else:
            # 1) Draw raw counts per dataset
            dataset_weights = torch.tensor(
                [self.dataset_weights[ds] for ds in self.datasets], dtype=torch.double
            )
            dataset_weights[dataset_weights < 0.1] = 0.1

            raw_choice = torch.multinomial(
                dataset_weights,
                num_samples,
                replacement=num_samples > len(dataset_weights),
                generator=rng,
            )
            raw_counts = [
                (raw_choice == i).sum().item() for i in range(len(self.datasets))
            ]

            # 2) Clamp counts at each dataset's size and accumulate overflow
            final_counts = []
            overflow = 0
            for i, ds in enumerate(self.datasets):
                size_i = len(ds)
                c = raw_counts[i]
                if c > size_i:
                    overflow += c - size_i
                    c = size_i
                final_counts.append(c)

            # 3) Distribute overflow via recursion, using dataset_weights
            capacity = [len(ds) - final_counts[i] for i, ds in enumerate(self.datasets)]
            weights = dataset_weights.clone()

            def redistribute(counts, caps, free_weights, over):
                """
                Recursively assign `over` extra samples to datasets in proportion to `free_weights`,
                but never exceed capacities in `caps`.

                Args:
                    counts       (List[int]): current final_counts per dataset
                    caps         (List[int]): remaining capacity per dataset
                    free_weights (torch.Tensor): clone of dataset_weights
                    over         (int): number of overflow samples to distribute

                Returns:
                    (new_counts, new_caps) after assigning as many as possible;
                    any leftover overflow will be handled by deeper recursion.
                """
                if over <= 0:
                    return counts, caps

                # Zero out weights where capacity == 0
                prob = free_weights.clone()
                for idx, cap_i in enumerate(caps):
                    if cap_i <= 0:
                        prob[idx] = 0.0

                total = prob.sum().item()
                if total <= 0:
                    # no capacity left to assign any overflow
                    return counts, caps

                prob = prob / total

                # Draw all `over` picks at once
                picks = torch.multinomial(
                    prob,
                    over,
                    replacement=True,
                    generator=rng,
                )
                freq = torch.bincount(picks, minlength=len(self.datasets)).tolist()

                new_counts = []
                new_caps = []
                leftover = 0
                for j, f_j in enumerate(freq):
                    cap_j = caps[j]
                    if f_j <= cap_j:
                        assigned = f_j
                        rem = 0
                    else:
                        assigned = cap_j
                        rem = f_j - cap_j

                    new_counts.append(counts[j] + assigned)
                    new_caps.append(cap_j - assigned)
                    leftover += rem

                # Recurse only if there’s leftover overflow
                return redistribute(new_counts, new_caps, free_weights, leftover)

            # Call the recursive allocator once
            final_counts, capacity = redistribute(
                final_counts, capacity, weights, overflow
            )

            # 4) Now that final_counts sums to num_samples (and each ≤ its dataset size),
            #    draw without replacement from each dataset:
            indices = []
            index_offset = 0
            for i, ds in enumerate(self.datasets):
                c = final_counts[i]
                size_i = len(ds)
                if c == 0:
                    index_offset += size_i
                    continue
                ds_indices = min_redundant_inds(size_i, c, rng=rng)
                indices.append(ds_indices + index_offset)
                index_offset += size_i

            all_indices = torch.cat(indices).flatten()
            all_indices = all_indices[
                min_redundant_inds(len(all_indices), num_samples, rng)
            ].tolist()
            return all_indices

    def get_subset_random_sampler(
        self,
        num_samples: int,
        weighted: bool = True,
        rng: Optional[torch.Generator] = None,
    ) -> MutableSubsetRandomSampler:
        indices_generator = functools.partial(
            self.get_random_subset_indices, num_samples, weighted, rng
        )

        return MutableSubsetRandomSampler(
            indices_generator,
            rng=rng,
        )

    def get_indices(self, chunk_size: Mapping[str, int]) -> Sequence[int]:
        """Returns the indices of the dataset that will tile all of the datasets according to the chunk_size."""
        indices = []
        for i, dataset in enumerate(self.datasets):
            if i == 0:
                offset = 0
            else:
                offset = self.cumulative_sizes[i - 1]
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
