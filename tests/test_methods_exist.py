import torch
from cellmap_data.dataset import CellMapDataset
from cellmap_data.dataset_writer import CellMapDatasetWriter
from cellmap_data.datasplit import CellMapDataSplit
from cellmap_data.image import CellMapImage
from cellmap_data.multidataset import CellMapMultiDataset
from cellmap_data.subdataset import CellMapSubset


def test_dataset_methods_exist():
    class Dummy(CellMapDataset):
        def __init__(self):
            pass

        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return {"x": torch.tensor([idx]), "y": torch.tensor(idx)}

        def center(self):
            return {"x": 0.0}

        def bounding_box(self):
            return {"x": [0.0, 1.0]}

    d = Dummy()
    assert hasattr(d, "center")
    assert hasattr(d, "bounding_box")


def test_dataset_writer_methods_exist():
    class Dummy(CellMapDatasetWriter):
        def __init__(self):
            pass

        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return {"x": torch.tensor([idx]), "y": torch.tensor(idx)}

        def center(self):
            return {"x": 0.0}

        def bounding_box(self):
            return {"x": [0.0, 1.0]}

    d = Dummy()
    assert hasattr(d, "center")
    assert hasattr(d, "bounding_box")


def test_datasplit_methods_exist():
    class Dummy(CellMapDataSplit):
        def __init__(self):
            pass

        def __repr__(self):
            return "DummySplit"

        def class_counts(self):
            return {"a": 1.0}

    d = Dummy()
    assert hasattr(d, "class_counts")
    assert repr(d) == "DummySplit"


def test_image_methods_exist():
    class Dummy(CellMapImage):
        def __init__(self):
            pass

        def shape(self):
            return {"x": 10}

        def center(self):
            return {"x": 5.0}

    d = Dummy()
    assert d.shape()["x"] == 10
    assert d.center()["x"] == 5.0


def test_multidataset_methods_exist():
    class Dummy(CellMapMultiDataset):
        def __init__(self):
            pass

        def class_counts(self):
            return {"a": 1.0}

    d = Dummy()
    assert d.class_counts()["a"] == 1.0


def test_subdataset_methods_exist():
    class Dummy(CellMapSubset):
        def __init__(self):
            pass

        def classes(self):
            return ["a", "b"]

    d = Dummy()
    assert d.classes() == ["a", "b"]
