from torch.utils.data import DataLoader

from .multidataset import CellMapMultiDataset
from .datasplit import CellMapDataSplit
from typing import Callable, Iterable, Optional


class CellMapDataLoader:
    # TODO: This class may be unnecessary
    # TODO: docstring corrections
    """This subclasses PyTorch DataLoader to load CellMap data for training. It maintains the same API as the DataLoader class. This includes applying augmentations to the data and returning the data in the correct format for training, such as generating the target arrays (e.g. signed distance transform of labels). It retrieves raw and groundtruth data from a CellMapDataSplit object, which is a subclass of PyTorch Dataset. Training and validation data are split using the CellMapDataSplit object, and separate dataloaders are maintained as `train_loader` and `validate_loader` respectively."""

    datasplit: CellMapDataSplit
    train_datasets: CellMapMultiDataset
    validate_datasets: CellMapMultiDataset
    train_loader: DataLoader
    validate_loader: DataLoader
    batch_size: int
    num_workers: int
    is_train: bool

    def __init__(
        self,
        datasplit: CellMapDataSplit,
        batch_size: int,
        num_workers: int,
        is_train: bool,
    ):
        self.datasplit = datasplit
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.is_train = is_train
        self.construct()

        # TODO: could keep dataloaders separate

    def construct(self):
        self.train_datasets = self.datasplit.train_datasets_combined
        self.validate_datasets = self.datasplit.validate_datasets_combined
        self.train_loader = DataLoader(
            self.train_datasets,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.is_train,
        )
        self.validate_loader = DataLoader(
            self.validate_datasets,
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False,
        )
