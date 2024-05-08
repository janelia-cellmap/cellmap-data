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

    def to(self, device):
        self.dataset.to(device)
        return self

    def set_raw_value_transforms(self, transforms: Callable):
        """Sets the raw value transforms for the subset dataset."""
        self.dataset.set_raw_value_transforms(transforms)

    def set_target_value_transforms(self, transforms: Callable):
        """Sets the target value transforms for the subset dataset."""
        self.dataset.set_target_value_transforms(transforms)

    def get_class_weights(self):
        """
        Returns the class weights for the multi-dataset based on the number of samples in each class.
        """
        if len(self.dataset.classes) > 1:
            class_counts = {c: 0 for c in self.dataset.classes}
            class_count_sum = 0
            for c in self.dataset.classes:
                class_counts[c] += self.dataset.class_counts["totals"][c]
                class_count_sum += self.dataset.class_counts["totals"][c]

            class_weights = {
                c: 1 - (class_counts[c] / class_count_sum) for c in self.dataset.classes
            }
        else:
            class_weights = {
                self.dataset.classes[0]: 0.1
            }  # less than 1 to avoid overflow
        return class_weights
