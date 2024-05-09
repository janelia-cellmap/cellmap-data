from typing import Callable
from torch.utils.data import Dataset


class CellMapSubset(Dataset):
    def __init__(self, dataset, indices):
        super().__init__()
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

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
