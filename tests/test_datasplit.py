"""Tests for CellMapDataSplit."""

from __future__ import annotations

import csv
import os

import torch

from cellmap_data import CellMapDataSplit
from cellmap_data.multidataset import CellMapMultiDataset

from .test_helpers import create_test_dataset

INPUT_ARRAYS = {"raw": {"shape": (4, 4, 4), "scale": (8.0, 8.0, 8.0)}}
TARGET_ARRAYS = {"labels": {"shape": (4, 4, 4), "scale": (8.0, 8.0, 8.0)}}
CLASSES = ["mito", "er"]


def _make_split_from_dict(tmp_path):
    train_info = create_test_dataset(tmp_path / "train", classes=CLASSES)
    val_info = create_test_dataset(tmp_path / "val", classes=CLASSES)
    dataset_dict = {
        "train": [{"raw": train_info["raw_path"], "gt": train_info["gt_path"]}],
        "validate": [{"raw": val_info["raw_path"], "gt": val_info["gt_path"]}],
    }
    return CellMapDataSplit(
        input_arrays=INPUT_ARRAYS,
        target_arrays=TARGET_ARRAYS,
        classes=CLASSES,
        dataset_dict=dataset_dict,
        force_has_data=True,
    )


class TestCellMapDataSplit:
    def test_init_from_dict(self, tmp_path):
        split = _make_split_from_dict(tmp_path)
        assert len(split.train_datasets) == 1
        assert len(split._validation_datasets) == 1

    def test_train_datasets_combined_type(self, tmp_path):
        split = _make_split_from_dict(tmp_path)
        combined = split.train_datasets_combined
        assert isinstance(combined, CellMapMultiDataset)

    def test_validation_datasets_combined_type(self, tmp_path):
        split = _make_split_from_dict(tmp_path)
        combined = split.validation_datasets_combined
        assert isinstance(combined, CellMapMultiDataset)

    def test_validation_datasets_property(self, tmp_path):
        split = _make_split_from_dict(tmp_path)
        assert len(split.validation_datasets) == 1

    def test_validation_blocks(self, tmp_path):
        from torch.utils.data import Subset

        split = _make_split_from_dict(tmp_path)
        blocks = split.validation_blocks
        assert isinstance(blocks, Subset)
        assert len(blocks) > 0

    def test_class_counts_keys(self, tmp_path):
        split = _make_split_from_dict(tmp_path)
        counts = split.class_counts
        assert "train" in counts
        assert "validate" in counts

    def test_init_from_csv(self, tmp_path):
        train_info = create_test_dataset(tmp_path / "train", classes=CLASSES)
        val_info = create_test_dataset(tmp_path / "val", classes=CLASSES)
        csv_path = str(tmp_path / "split.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["train", train_info["raw_path"], train_info["gt_path"]])
            w.writerow(["validate", val_info["raw_path"], val_info["gt_path"]])
        split = CellMapDataSplit(
            input_arrays=INPUT_ARRAYS,
            target_arrays=TARGET_ARRAYS,
            classes=CLASSES,
            csv_path=csv_path,
            force_has_data=True,
        )
        assert len(split.train_datasets) == 1
        assert len(split._validation_datasets) == 1

    def test_set_raw_value_transforms(self, tmp_path):
        split = _make_split_from_dict(tmp_path)
        import torchvision.transforms.v2 as T

        tx = T.ToDtype(torch.float, scale=True)
        split.set_raw_value_transforms(train_transforms=tx, val_transforms=tx)
        # Check that train datasets have the new transform
        for ds in split.train_datasets:
            assert ds.raw_value_transforms is tx

    def test_invalidate_clears_combined(self, tmp_path):
        split = _make_split_from_dict(tmp_path)
        combined1 = split.train_datasets_combined
        split._invalidate()
        combined2 = split.train_datasets_combined
        # After invalidation, a new CellMapMultiDataset is created
        assert combined1 is not combined2

    def test_repr(self, tmp_path):
        split = _make_split_from_dict(tmp_path)
        r = repr(split)
        assert "CellMapDataSplit" in r

    def test_init_empty(self):
        split = CellMapDataSplit(
            input_arrays=INPUT_ARRAYS,
            target_arrays=TARGET_ARRAYS,
            classes=CLASSES,
        )
        assert len(split.train_datasets) == 0
        assert len(split._validation_datasets) == 0

    def test_init_from_csv_5col(self, tmp_path):
        """5-column challenge CSV format: split, zarr_path, raw_ds, zarr_path, gt_ds."""
        train_info = create_test_dataset(tmp_path / "train5", classes=CLASSES)
        val_info = create_test_dataset(tmp_path / "val5", classes=CLASSES)

        # Simulate challenge CSV: split zarr_path and sub-path across columns 1+2 and 3+4
        train_zarr = str(tmp_path / "train5")
        train_raw_ds = os.path.relpath(train_info["raw_path"], train_zarr)
        train_gt_ds = os.path.relpath(
            train_info["gt_path"].split("[")[0].rstrip(os.sep), train_zarr
        )
        train_classes = ",".join(CLASSES)

        val_zarr = str(tmp_path / "val5")
        val_raw_ds = os.path.relpath(val_info["raw_path"], val_zarr)
        val_gt_ds = os.path.relpath(
            val_info["gt_path"].split("[")[0].rstrip(os.sep), val_zarr
        )

        csv_path = str(tmp_path / "split5.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "train",
                    train_zarr,
                    train_raw_ds,
                    train_zarr,
                    f"{train_gt_ds}[{train_classes}]",
                ]
            )
            w.writerow(
                [
                    "validate",
                    val_zarr,
                    val_raw_ds,
                    val_zarr,
                    f"{val_gt_ds}[{','.join(CLASSES)}]",
                ]
            )

        split = CellMapDataSplit(
            input_arrays=INPUT_ARRAYS,
            target_arrays=TARGET_ARRAYS,
            classes=CLASSES,
            csv_path=csv_path,
            force_has_data=True,
        )
        assert len(split.train_datasets) == 1
        assert len(split._validation_datasets) == 1

    def test_init_from_datasets(self, tmp_path):
        from cellmap_data import CellMapDataset

        train_info = create_test_dataset(tmp_path / "d1", classes=CLASSES)
        ds = CellMapDataset(
            raw_path=train_info["raw_path"],
            target_path=train_info["gt_path"],
            classes=CLASSES,
            input_arrays=INPUT_ARRAYS,
            target_arrays=TARGET_ARRAYS,
            force_has_data=True,
        )
        split = CellMapDataSplit(
            input_arrays=INPUT_ARRAYS,
            target_arrays=TARGET_ARRAYS,
            classes=CLASSES,
            datasets={"train": [ds], "validate": []},
        )
        assert len(split.train_datasets) == 1
        assert len(split._validation_datasets) == 0
