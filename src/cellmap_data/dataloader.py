import torch
from torch.utils.data import DataLoader, Sampler, Subset
from .dataset import CellMapDataset
from .multidataset import CellMapMultiDataset

from typing import Iterable, Optional


class CellMapDataLoader:
    # TODO: docstring corrections
    """This subclasses PyTorch DataLoader to load CellMap data for training. It maintains the same API as the DataLoader class. This includes applying augmentations to the data and returning the data in the correct format for training, such as generating the target arrays (e.g. signed distance transform of labels). It retrieves raw and groundtruth data from a CellMapDataSplit object, which is a subclass of PyTorch Dataset. Training and validation data are split using the CellMapDataSplit object, and separate dataloaders are maintained as `train_loader` and `validate_loader` respectively."""

    dataset: (
        CellMapMultiDataset
        | CellMapDataset
        | Subset[CellMapDataset | CellMapMultiDataset]
    )
    classes: Iterable[str]
    loader = DataLoader
    batch_size: int
    num_workers: int
    weighted_sampler: bool
    sampler: Sampler | None
    is_train: bool
    rng: Optional[torch.Generator] = None

    def __init__(
        self,
        dataset: (
            CellMapMultiDataset
            | CellMapDataset
            | Subset[CellMapDataset | CellMapMultiDataset]
        ),
        classes: Iterable[str],
        batch_size: int = 1,
        num_workers: int = 0,
        weighted_sampler: bool = False,
        sampler: Sampler | None = None,
        is_train: bool = True,
        rng: Optional[torch.Generator] = None,
    ):
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
            self.sampler = self.dataset.weighted_sampler(self.batch_size, self.rng)
        if torch.cuda.is_available():
            if isinstance(self.dataset, Subset):
                self.dataset.dataset.to("cuda")  # type: ignore
            else:
                self.dataset.to("cuda")
        kwargs = {
            "dataset": self.dataset,
            "dataset": self.dataset,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "collate_fn": self.collate_fn,
        }
        if self.sampler is not None:
            kwargs["sampler"] = self.sampler
        elif self.is_train:
            kwargs["shuffle"] = True
        else:
            kwargs["shuffle"] = False
        self.loader = DataLoader(**kwargs)

    def collate_fn(self, batch):
        outputs = {}
        for b in batch:
            for key, value in b.items():
                if key not in outputs:
                    outputs[key] = []
                outputs[key].append(value)
        for key, value in outputs.items():
            outputs[key] = torch.stack(value)
        return outputs
