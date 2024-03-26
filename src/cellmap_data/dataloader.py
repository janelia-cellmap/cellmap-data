from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _BaseDataLoaderIter
from .dataset import CellMapDataset
from .datasplit import CellMapDataSplit
from typing import Callable, Iterable


class CellMapDataLoader(DataLoader):
    """This subclasses PyTorch DataLoader to load CellMap data for training. It maintains the same API as the DataLoader class. This includes applying augmentations to the data and returning the data in the correct format for training, such as generating the target arrays (e.g. signed distance transform of labels). It retrieves raw and groundtruth data from a CellMapDataSplit object, which is a subclass of PyTorch Dataset. Training and validation data are split using the CellMapDataSplit object, and separate dataloaders are maintained as `train_loader` and `val_loader` respectively."""

    input_arrays: dict[str, dict[str, tuple[int | float]]]
    target_arrays: dict[str, dict[str, tuple[int | float]]]
    classes: list[str]
    datasplit: CellMapDataSplit
    train_loader: DataLoader
    val_loader: DataLoader
    is_train: bool
    augmentations: list[dict[str, any]]
    to_target: Callable

    def __init__(
            ...)
    
    def __getitem__(self, idx: int) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Returns the input and target data for the given index."""
        ...
    
    def __len__(self) -> int:
        """Returns the length of the dataset."""
        ...

    def __iter__(self):
        
    
    def _apply_augmentations(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Applies augmentations to the data."""
        ...

    def _to_target(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Converts the input data to the target data."""
        ...
