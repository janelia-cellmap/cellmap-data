import torch
from torch.utils.data import DataLoader, Sampler
from .dataset import CellMapDataset
from .multidataset import CellMapMultiDataset
from .subdataset import CellMapSubset
from typing import Callable, Iterable, Optional, Sequence


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
        dataset: CellMapMultiDataset | CellMapDataset | CellMapSubset,
        classes: Iterable[str],
        batch_size: int = 1,
        num_workers: int = 0,
        weighted_sampler: bool = False,
        sampler: Sampler | Callable | None = None,
        is_train: bool = True,
        rng: Optional[torch.Generator] = None,
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
            `**kwargs`: Additional arguments to pass to the DataLoader.

        """
        self.dataset = dataset
        self.classes = classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.weighted_sampler = weighted_sampler
        self.sampler = sampler
        self.is_train = is_train
        self.rng = rng
        if self.sampler is None and self.weighted_sampler:
            assert isinstance(
                self.dataset, CellMapMultiDataset
            ), "Weighted sampler only relevant for CellMapMultiDataset"
            self.sampler = self.dataset.get_weighted_sampler(self.batch_size, self.rng)
        if torch.cuda.is_available():
            self.dataset.to("cuda")
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
        # TODO: Try persistent workers
        self.loader = DataLoader(**kwargs)

    def __getitem__(self, indices: Sequence[int]) -> dict:
        """Get an item from the DataLoader."""
        return self.collate_fn([self.loader.dataset[index] for index in indices])

    def refresh(self):
        """If the sampler is a Callable, refresh the DataLoader with the current sampler."""
        if isinstance(self.sampler, Callable):
            kwargs = self.default_kwargs.copy()
            kwargs.update(
                {
                    "dataset": self.dataset,
                    "batch_size": self.batch_size,
                    "num_workers": self.num_workers,
                    "collate_fn": self.collate_fn,
                    "shuffle": False,
                }
            )
            kwargs["sampler"] = self.sampler()
            self.loader = DataLoader(**kwargs)

    def collate_fn(self, batch: list[dict]) -> dict:
        """Combine a list of dictionaries from different sources into a single dictionary for output."""
        outputs = {}
        for b in batch:
            for key, value in b.items():
                if key not in outputs:
                    outputs[key] = []
                outputs[key].append(value)
        for key, value in outputs.items():
            outputs[key] = torch.stack(value)
        return outputs
