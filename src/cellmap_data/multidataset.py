from typing import Iterable, Sequence
from torch.utils.data import ConcatDataset

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
        datasets: Iterable[CellMapDataset],
    ):
        self.input_arrays = input_arrays
        self.target_arrays = target_arrays
        self.classes = classes
        self.datasets = datasets
        self.construct()
        # TODO: SHOULD BE REPLACEABLE BY torch.utils.data.ConcatDataset

    def __len__(self):
        # TODO
        ...

    def __getitem__(self, idx: int):
        # TODO
        ...

    def __iter__(self):
        # TODO
        ...

    def construct(self):
        # TODO
        ...


# TODO: make "last" and "current" variable names consistent
