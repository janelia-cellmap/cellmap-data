import torch
from torch.utils.data import DataLoader, Sampler, Subset
from .dataset import CellMapDataset
from .multidataset import CellMapMultiDataset
from .dataset_writer import CellMapDatasetWriter
from typing import Callable, Optional, Sequence


class CellMapDataLoader:
    """
    Utility class to create a DataLoader for a CellMapDataset or CellMapMultiDataset.

    Attributes:
        dataset (CellMapMultiDataset | CellMapDataset | CellMapSubset): The dataset to load.
        classes (Iterable[str]): The classes to load.
        batch_size (int): The batch size.
        num_workers (int): The number of workers to use.
        weighted_sampler (bool): Whether to use a weighted sampler.
        sampler (Sampler | Callable | None): The sampler to use.
        is_train (bool): Whether the data is for training and thus should be shuffled.
        rng (Optional[torch.Generator]): The random number generator to use.
        loader (DataLoader): The PyTorch DataLoader.
        default_kwargs (dict): The default arguments to pass to the PyTorch DataLoader.

    Methods:
        refresh: If the sampler is a Callable, refresh the DataLoader with the current sampler.
        collate_fn: Combine a list of dictionaries from different sources into a single dictionary for output.

    """

    def __init__(
        self,
        dataset: CellMapMultiDataset | CellMapDataset | Subset | CellMapDatasetWriter,
        classes: Sequence[str] | None = None,
        batch_size: int = 1,
        num_workers: int = 0,
        weighted_sampler: bool = False,
        sampler: Sampler | Callable | None = None,
        is_train: bool = True,
        rng: Optional[torch.Generator] = None,
        device: Optional[str | torch.device] = None,
        **kwargs,
    ):
        """
        Initialize the CellMapDataLoader

        Args:
            dataset (CellMapMultiDataset | CellMapDataset | CellMapSubset): The dataset to load.
            classes (Iterable[str]): The classes to load.
            batch_size (int): The batch size.
            num_workers (int): The number of workers to use.
            weighted_sampler (bool): Whether to use a weighted sampler. Defaults to False.
            sampler (Sampler | Callable | None): The sampler to use.
            is_train (bool): Whether the data is for training and thus should be shuffled.
            rng (Optional[torch.Generator]): The random number generator to use.
            device (Optional[str | torch.device]): The device to use. Defaults to "cuda" or "mps" if available, else "cpu".
            `**kwargs`: Additional arguments to pass to the DataLoader.

        """
        self.dataset = dataset
        self.classes = classes if classes is not None else dataset.classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.weighted_sampler = weighted_sampler
        self.sampler = sampler
        self.is_train = is_train
        self.rng = rng
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        if num_workers == 0:
            self.dataset.to(device, non_blocking=True)
        if self.sampler is None and self.weighted_sampler:
            assert isinstance(
                self.dataset, CellMapMultiDataset
            ), "Weighted sampler only relevant for CellMapMultiDataset"
            self.sampler = self.dataset.get_weighted_sampler(self.batch_size, self.rng)
        self.default_kwargs = kwargs.copy()
        kwargs.update(
            {
                "dataset": self.dataset,
                "batch_size": self.batch_size,
                "num_workers": self.num_workers,
                "collate_fn": self.collate_fn,
            }
        )
        if self.sampler is not None:
            if isinstance(self.sampler, Callable):
                kwargs["sampler"] = self.sampler()
            else:
                kwargs["sampler"] = self.sampler
        elif self.is_train:
            kwargs["shuffle"] = True
        else:
            kwargs["shuffle"] = False
        self.loader = DataLoader(**kwargs)

    def __getitem__(self, indices: Sequence[int]) -> dict:
        """Get an item from the DataLoader."""
        if isinstance(indices, int):
            indices = [indices]
        return self.collate_fn([self.loader.dataset[index] for index in indices])

    def to(self, device: str | torch.device, non_blocking: bool = True):
        """Move the dataset to the specified device."""
        self.dataset.to(device, non_blocking=non_blocking)
        self.device = device

    def refresh(self):
        """If the sampler is a Callable, refresh the DataLoader with the current sampler."""
        kwargs = self.default_kwargs.copy()
        kwargs.update(
            {
                "dataset": self.dataset,
                "batch_size": self.batch_size,
                "num_workers": self.num_workers,
                "collate_fn": self.collate_fn,
            }
        )
        if self.sampler is not None:
            if isinstance(self.sampler, Callable):
                kwargs["sampler"] = self.sampler()
            else:
                kwargs["sampler"] = self.sampler
        elif self.is_train:
            kwargs["shuffle"] = True
        else:
            kwargs["shuffle"] = False
        self.loader = DataLoader(**kwargs)

    def collate_fn(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        """Combine a list of dictionaries from different sources into a single dictionary for output."""
        outputs = {}
        for b in batch:
            for key, value in b.items():
                if key not in outputs:
                    outputs[key] = []
                outputs[key].append(value)
        for key, value in outputs.items():
            outputs[key] = torch.stack(value).to(self.device, non_blocking=True)
        return outputs
