import os
import torch


def pytest_configure():
    # Force torch to avoid MPS (failing in GitHub CI)
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    torch.backends.mps.is_available = lambda: False


import pytest
from unittest.mock import MagicMock
from cellmap_data.dataset import CellMapDataset
from cellmap_data.dataset_writer import CellMapDatasetWriter
from cellmap_data.multidataset import CellMapMultiDataset


@pytest.fixture
def mock_dataset():
    """
    Provides a generic, MagicMock-based dataset.
    """
    ds = MagicMock()
    ds.classes = ["a", "b"]
    ds.input_arrays = {"in": {}}
    ds.target_arrays = {"out": {}}
    ds.class_counts = {"totals": {"a": 10, "a_bg": 90, "b": 20, "b_bg": 80}}
    ds.validation_indices = [0, 1]
    ds.verify.return_value = True
    ds.__len__.return_value = 5
    ds.get_indices.return_value = [0, 1, 2]
    ds.to.return_value = ds
    return ds


@pytest.fixture
def dummy_dataset():
    """
    Provides a simple, functional torch.utils.data.Dataset.
    """

    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, length=10, num_features=3):
            self.length = length
            self.num_features = num_features
            self.classes = ["a", "b"]
            self.class_counts = {"a": 5, "b": 5}
            self.class_weights = {"a": 0.5, "b": 0.5}
            self.validation_indices = list(range(length // 2))
            self.input_arrays = {"x": {"shape": (num_features,)}}
            self.target_arrays = {"y": {"shape": (1,)}}

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            return {
                "x": torch.tensor([idx] * self.num_features, dtype=torch.float32),
                "y": torch.tensor(idx % 2),
            }

        def to(self, device, non_blocking=True):
            # This mock doesn't need to do anything for device transfer
            return self

    return DummyDataset()


@pytest.fixture
def mocked_cellmap_dataset(monkeypatch):
    """
    Provides a CellMapDataset instance with mocked file access.
    """
    from pathlib import Path

    monkeypatch.setattr("zarr.open_group", lambda path, mode="r": MagicMock())
    monkeypatch.setattr("tensorstore.open", lambda spec: MagicMock())
    monkeypatch.setattr(Path, "exists", lambda self: True)

    dataset = CellMapDataset(
        raw_path="/fake/path",
        target_path="/fake/path",
        classes=["test"],
        input_arrays={"em": {"shape": (64, 64, 64), "scale": (1.0, 1.0, 1.0)}},
        target_arrays={"labels": {"shape": (64, 64, 64), "scale": (1.0, 1.0, 1.0)}},
    )
    return dataset


@pytest.fixture
def empty_mock_dataset(mocker):
    """A mock dataset that is empty (has a length of 0)."""
    mock = mocker.MagicMock(spec=CellMapDataset)
    mock.__len__.return_value = 0
    mock.classes = []
    # Add other attributes that might be accessed
    mock.batch_size = 1
    mock.sampler = None
    mock.pin_memory = False
    mock.num_workers = 0
    mock.collate_fn = None
    mock.prefetch_factor = 2
    mock.persistent_workers = False
    mock.timeout = 0
    return mock


@pytest.fixture
def empty_cellmap_dataset(mocker):
    """A completely empty CellMapDataset that has no data."""
    return CellMapDataset.empty()


@pytest.fixture
def empty_cellmap_multidataset(mocker):
    """A completely empty CellMapMultiDataset that has no data."""
    return CellMapMultiDataset.empty()
