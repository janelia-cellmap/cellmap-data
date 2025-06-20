import torch
import numpy as np
import pytest
from unittest.mock import MagicMock

from cellmap_data.dataset import CellMapDataset
from cellmap_data.dataset_writer import CellMapDatasetWriter
from cellmap_data.utils.misc import split_target_path
from cellmap_data.datasplit import CellMapDataSplit
from cellmap_data.image import CellMapImage
from cellmap_data.multidataset import CellMapMultiDataset
from cellmap_data.subdataset import CellMapSubset


def test_split_target_path_dataset():
    path = "foo/[bar,baz]"
    root, parts = split_target_path(path)
    assert isinstance(root, str)
    assert isinstance(parts, list)
    assert root == "foo/{label}"
    assert parts == ["bar", "baz"]


@pytest.fixture
def mock_dataset():
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


def test_has_data(mock_dataset):
    mds = CellMapMultiDataset(["a", "b"], {"in": {}}, {"out": {}}, [mock_dataset])
    assert mds.has_data is True
    mds_empty = CellMapMultiDataset.empty()
    assert mds_empty.has_data is False


def test_class_counts_and_weights(mock_dataset):
    mds = CellMapMultiDataset(["a", "b"], {"in": {}}, {"out": {}}, [mock_dataset])
    cc = mds.class_counts
    assert "totals" in cc
    assert cc["totals"]["a"] == 10
    assert cc["totals"]["b"] == 20
    cw = mds.class_weights
    assert set(cw.keys()) == {"a", "b"}
    assert cw["a"] == 90 / 10
    assert cw["b"] == 80 / 20


def test_dataset_weights_and_sample_weights(mock_dataset):
    mds = CellMapMultiDataset(["a", "b"], {"in": {}}, {"out": {}}, [mock_dataset])
    dw = mds.dataset_weights
    assert mock_dataset in dw
    sw = mds.sample_weights
    assert len(sw) == len(mock_dataset)


def test_validation_indices(mock_dataset):
    mds = CellMapMultiDataset(["a", "b"], {"in": {}}, {"out": {}}, [mock_dataset])
    indices = mds.validation_indices
    assert indices == [0, 1]


def test_verify(mock_dataset):
    mds = CellMapMultiDataset(["a", "b"], {"in": {}}, {"out": {}}, [mock_dataset])
    assert mds.verify() is True
    mds_empty = CellMapMultiDataset.empty()
    assert mds_empty.verify() is False
    ds_empty = CellMapDataset(
        raw_path="dummy_raw_path",
        target_path="dummy_path",
        classes=["a", "b"],
        input_arrays={"in": {"shape": (1, 1, 1), "scale": (1.0, 1.0, 1.0)}},
        target_arrays={"out": {"shape": (1, 1, 1), "scale": (1.0, 1.0, 1.0)}},
    )
    assert ds_empty.verify() is False


def test_empty():
    mds = CellMapMultiDataset.empty()
    assert isinstance(mds, CellMapMultiDataset)
    assert mds.has_data is False
    ds = CellMapDataset.empty()
    assert isinstance(ds, CellMapDataset)
    assert ds.has_data is False


def test_repr(mock_dataset):
    mds = CellMapMultiDataset(["a", "b"], {"in": {}}, {"out": {}}, [mock_dataset])
    s = repr(mds)
    assert "CellMapMultiDataset" in s


def test_to_device(mock_dataset):
    mds = CellMapMultiDataset(["a", "b"], {"in": {}}, {"out": {}}, [mock_dataset])
    result = mds.to("cpu")
    assert result is mds


def test_get_weighted_sampler(mock_dataset):
    mds = CellMapMultiDataset(["a", "b"], {"in": {}}, {"out": {}}, [mock_dataset])
    sampler = mds.get_weighted_sampler(batch_size=2)
    assert hasattr(sampler, "__iter__")


def test_get_subset_random_sampler(mock_dataset):
    mds = CellMapMultiDataset(["a", "b"], {"in": {}}, {"out": {}}, [mock_dataset])
    sampler = mds.get_subset_random_sampler(num_samples=2)
    assert hasattr(sampler, "__iter__")


def test_multidataset_2d_shape_triggers_axis_slicing(monkeypatch):
    """Test that requesting a 2D shape triggers creation of 3 datasets, one for each axis."""
    from cellmap_data.dataset import CellMapDataset
    from cellmap_data.multidataset import CellMapMultiDataset

    # Patch CellMapDataset.__init__ to record calls and not do real work
    created = []
    orig_init = CellMapDataset.__init__

    def fake_init(self, *args, **kwargs):
        created.append((args, kwargs))
        orig_init(self, *args, **kwargs)

    monkeypatch.setattr(CellMapDataset, "__init__", fake_init)

    # Patch CellMapMultiDataset to record datasets passed to it
    multi_created = {}
    orig_multi_init = CellMapMultiDataset.__init__

    def fake_multi_init(self, classes, input_arrays, target_arrays, datasets):
        multi_created["datasets"] = datasets
        orig_multi_init(self, classes, input_arrays, target_arrays, datasets)

    monkeypatch.setattr(CellMapMultiDataset, "__init__", fake_multi_init)

    # 2D shape triggers slicing
    input_arrays = {"in": {"shape": (32, 32), "scale": (1.0, 1.0, 1.0)}}
    target_arrays = {"out": {"shape": (32, 32), "scale": (1.0, 1.0, 1.0)}}
    classes = ["a", "b"]

    # Use __new__ directly to trigger the logic
    ds = CellMapDataset.__new__(
        CellMapDataset,
        raw_path="dummy_raw_path",
        target_path="dummy_path",
        classes=classes,
        input_arrays=input_arrays,
        target_arrays=target_arrays,
        spatial_transforms=None,
        raw_value_transforms=None,
        target_value_transforms=None,
        class_relation_dict=None,
        is_train=False,
        axis_order="zyx",
        context=None,
        rng=None,
        force_has_data=False,
        empty_value=torch.nan,
        pad=True,
        device=None,
    )

    # Should return a CellMapMultiDataset
    assert isinstance(ds, CellMapMultiDataset)
    # Should have created 3 datasets (one per axis)
    assert "datasets" in multi_created
    assert len(multi_created["datasets"]) == 3

    # Each dataset should have 2D shape in its input_arrays
    for d in multi_created["datasets"]:
        arr = d.input_arrays["in"]["shape"]
        assert len(arr) == 2


def test_multidataset_3d_shape_does_not_trigger_axis_slicing(monkeypatch):
    """Test that requesting a 3D shape does not trigger axis slicing."""
    from cellmap_data.dataset import CellMapDataset
    from cellmap_data.multidataset import CellMapMultiDataset

    # Patch CellMapMultiDataset to raise if called
    monkeypatch.setattr(
        CellMapMultiDataset,
        "__init__",
        lambda *a, **k: (_ for _ in ()).throw(Exception("Should not be called")),
    )

    input_arrays = {"in": {"shape": (32, 32, 32), "scale": (1.0, 1.0, 1.0)}}
    target_arrays = {"out": {"shape": (32, 32, 32), "scale": (1.0, 1.0, 1.0)}}
    classes = ["a", "b"]

    # Use __new__ directly to trigger the logic
    ds = CellMapDataset.__new__(
        CellMapDataset,
        raw_path="dummy_raw_path",
        target_path="dummy_path",
        classes=classes,
        input_arrays=input_arrays,
        target_arrays=target_arrays,
        spatial_transforms=None,
        raw_value_transforms=None,
        target_value_transforms=None,
        class_relation_dict=None,
        is_train=False,
        axis_order="zyx",
        context=None,
        rng=None,
        force_has_data=False,
        empty_value=torch.nan,
        pad=True,
        device=None,
    )

    # Should return a CellMapDataset instance, not a CellMapMultiDataset
    assert isinstance(ds, CellMapDataset)
