import pytest
import torch
from cellmap_data.dataloader import CellMapDataLoader
from cellmap_data.subdataset import CellMapSubset
from cellmap_data.dataset import CellMapDataset
from cellmap_data.multidataset import CellMapMultiDataset


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, length=10, num_features=3):
        self.length = length
        self.num_features = num_features
        self.classes = ["a", "b"]
        self.class_counts = {"a": 5, "b": 5}
        self.class_weights = {"a": 0.5, "b": 0.5}
        self.validation_indices = list(range(length // 2))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            "x": torch.tensor([idx] * self.num_features, dtype=torch.float32),
            "y": torch.tensor(idx % 2),
        }

    def to(self, device, non_blocking=True):
        return self


def test_dataloader_basic():
    dataset = DummyDataset()
    loader = CellMapDataLoader(dataset, batch_size=2, num_workers=0)
    batch = next(iter(loader.loader))
    assert "x" in batch and "y" in batch
    assert batch["x"].shape[0] == 2
    assert batch["x"].device.type == loader.device


def test_dataloader_to_device():
    dataset = DummyDataset()
    loader = CellMapDataLoader(dataset, batch_size=2, num_workers=0)
    loader.to("cpu")
    assert loader.device == "cpu"


def test_dataloader_getitem():
    dataset = DummyDataset()
    loader = CellMapDataLoader(dataset, batch_size=2, num_workers=0)
    item = loader[[0, 1]]
    assert "x" in item and item["x"].shape[0] == 2


def test_dataloader_refresh():
    dataset = DummyDataset()
    loader = CellMapDataLoader(dataset, batch_size=2, num_workers=0)
    loader.refresh()
    batch = next(iter(loader.loader))
    assert batch["x"].shape[0] == 2
