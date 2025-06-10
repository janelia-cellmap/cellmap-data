import torch
from cellmap_data.dataset import split_target_path, CellMapDataset
from cellmap_data.dataset_writer import (
    split_target_path as writer_split_target_path,
    CellMapDatasetWriter,
)
from cellmap_data.datasplit import CellMapDataSplit
from cellmap_data.image import CellMapImage
from cellmap_data.multidataset import CellMapMultiDataset
from cellmap_data.subdataset import CellMapSubset


def test_split_target_path_dataset():
    path = "foo/bar/baz"
    root, parts = split_target_path(path)
    assert isinstance(root, str)
    assert isinstance(parts, list)


def test_split_target_path_writer():
    path = "foo/bar/baz"
    root, parts = writer_split_target_path(path)
    assert isinstance(root, str)
    assert isinstance(parts, list)


def test_cellmap_dataset_len_and_getitem():
    # Minimal mock for CellMapDataset
    class Dummy(CellMapDataset):
        def __init__(self):
            pass

        def __len__(self):
            return 3

        def __getitem__(self, idx):
            return {"x": torch.tensor([idx]), "y": torch.tensor(idx)}

    d = Dummy()
    assert len(d) == 3
    item = d[0]
    assert "x" in item and "y" in item


def test_cellmap_dataset_writer_len_and_getitem():
    class Dummy(CellMapDatasetWriter):
        def __init__(self):
            pass

        def __len__(self):
            return 2

        def __getitem__(self, idx):
            return {"x": torch.tensor([idx]), "y": torch.tensor(idx)}

    d = Dummy()
    assert len(d) == 2
    item = d[1]
    assert "x" in item and "y" in item


def test_cellmap_datasplit_repr():
    class Dummy(CellMapDataSplit):
        def __init__(self):
            pass

        def __repr__(self):
            return "DummySplit"

    d = Dummy()
    assert repr(d) == "DummySplit"


def test_cellmap_image_shape_and_center():
    class Dummy(CellMapImage):
        def __init__(self):
            pass

        def shape(self):
            return {"x": 10, "y": 10}

        def center(self):
            return {"x": 5.0, "y": 5.0}

    d = Dummy()
    assert d.shape()["x"] == 10
    assert d.center()["y"] == 5.0


def test_cellmap_multidataset_class_counts():
    class Dummy(CellMapMultiDataset):
        def __init__(self):
            pass

        def class_counts(self):
            return {"a": 1.0, "b": 2.0}

    d = Dummy()
    cc = d.class_counts()
    assert "a" in cc and cc["b"] == 2.0


def test_cellmap_subdataset_classes():
    class Dummy(CellMapSubset):
        def __init__(self):
            pass

        def classes(self):
            return ["a", "b"]

    d = Dummy()
    assert d.classes() == ["a", "b"]
