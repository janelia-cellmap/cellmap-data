from typing import Iterable, Optional, Sequence
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
        datasets: Iterable[CellMapDataset],
    ):
        super().__init__(datasets)
        self.input_arrays = input_arrays
        self.target_arrays = target_arrays
        self.classes = classes
        self.datasets = datasets
        self.construct()

    def __repr__(self) -> str:
        out_string = f"CellMapMultiDataset(["
        for dataset in self.datasets:
            out_string += f"\n\t{dataset},"
        out_string += "\n])"
        return out_string

    def to(self, device: str):
        for dataset in self.datasets:
            dataset.to(device)
        return self

    def construct(self):
        self._weighted_sampler = None

    def weighted_sampler(
        self, batch_size: int = 1, rng: Optional[torch.Generator] = None
    ):
        if self._weighted_sampler is None:
            # TODO: calculate weights for each sample
            class_counts = {c: 0 for c in self.classes}
            for dataset in self.datasets:
                for c in self.classes:
                    class_counts[c] += dataset.class_counts["totals"][c]
            class_weights = {c: 1 / class_counts[c] for c in self.classes}
            sample_weights = []
            for dataset in self.datasets:
                dataset_weight = np.sum(
                    [
                        dataset.class_counts["totals"][c] / class_weights[c]
                        for c in self.classes
                    ]
                )
                sample_weights += [dataset_weight] * len(dataset)

            self._weighted_sampler = WeightedRandomSampler(
                sample_weights, batch_size, replacement=False, generator=rng
            )
        return self._weighted_sampler

    @property
    def class_counts(self):
        if not self._class_counts:
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


# TODO: make "last" and "current" variable names consistent
