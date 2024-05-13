from typing import Callable, Sequence
from torch.utils.data import Subset
from .dataset import CellMapDataset
from .multidataset import CellMapMultiDataset


class CellMapSubset(Subset):

    dataset: CellMapDataset | CellMapMultiDataset
    indices: Sequence[int]

    def __init__(
        self, dataset: CellMapDataset | CellMapMultiDataset, indices: Sequence[int]
    ) -> None:
        super().__init__(dataset, indices)

    @property
    def classes(self):
        return self.dataset.classes

    @property
    def class_counts(self):
        return self.dataset.class_counts

    @property
    def class_weights(self):
        return self.dataset.class_weights

    @property
    def validation_indices(self):
        return self.dataset.validation_indices

    def to(self, device):
        self.dataset.to(device)
        return self

    def set_raw_value_transforms(self, transforms: Callable):
        """Sets the raw value transforms for the subset dataset."""
        self.dataset.set_raw_value_transforms(transforms)

    def set_target_value_transforms(self, transforms: Callable):
        """Sets the target value transforms for the subset dataset."""
        self.dataset.set_target_value_transforms(transforms)
