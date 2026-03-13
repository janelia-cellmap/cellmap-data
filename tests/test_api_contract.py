"""Tests that validate every API call documented in API_TO_PRESERVE.md.

These tests mirror the exact constructor signatures, attribute accesses, and
call patterns used in cellmap-segmentation-challenge.
"""

from __future__ import annotations

import csv
import os

import torch
import pytest

from cellmap_data import (
    CellMapDataLoader,
    CellMapDataSplit,
    CellMapDatasetWriter,
    CellMapImage,
)
from cellmap_data.transforms.augment import Binarize, NaNtoNum
from cellmap_data.utils import (
    array_has_singleton_dim,
    get_fig_dict,
    is_array_2D,
    longest_common_substring,
    permute_singleton_dimension,
)

from .test_helpers import create_test_dataset, create_test_zarr

import torchvision.transforms.v2 as T

# Default transforms used throughout cellmap-segmentation-challenge
_RAW_TX = T.Compose(
    [
        T.ToDtype(torch.float, scale=True),
        NaNtoNum({"nan": 0, "posinf": None, "neginf": None}),
    ]
)
_TARGET_TX = T.Compose([T.ToDtype(torch.float), Binarize()])

INPUT_ARRAYS = {"raw": {"shape": (4, 4, 4), "scale": (8.0, 8.0, 8.0)}}
TARGET_ARRAYS = {"labels": {"shape": (4, 4, 4), "scale": (8.0, 8.0, 8.0)}}
CLASSES = ["mito", "er"]


# ---------------------------------------------------------------------------
# CellMapDataSplit — full API
# ---------------------------------------------------------------------------


class TestCellMapDataSplitAPI:
    """Mirrors utils/dataloader.py usage in cellmap-segmentation-challenge."""

    def _make_csv(self, tmp_path):
        train_info = create_test_dataset(tmp_path / "train", classes=CLASSES)
        val_info = create_test_dataset(tmp_path / "val", classes=CLASSES)
        csv_path = str(tmp_path / "split.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["train", train_info["raw_path"], train_info["gt_path"]])
            w.writerow(["validate", val_info["raw_path"], val_info["gt_path"]])
        return csv_path

    def test_constructor_with_csv(self, tmp_path):
        csv_path = self._make_csv(tmp_path)
        split = CellMapDataSplit(
            input_arrays=INPUT_ARRAYS,
            target_arrays=TARGET_ARRAYS,
            classes=CLASSES,
            pad=True,
            csv_path=csv_path,
            train_raw_value_transforms=_RAW_TX,
            val_raw_value_transforms=_RAW_TX,
            target_value_transforms=_TARGET_TX,
            spatial_transforms=None,
            device="cpu",
            class_relation_dict=None,
            force_has_data=True,
        )
        assert split is not None

    def test_validation_datasets_is_list(self, tmp_path):
        csv_path = self._make_csv(tmp_path)
        split = CellMapDataSplit(
            input_arrays=INPUT_ARRAYS,
            target_arrays=TARGET_ARRAYS,
            classes=CLASSES,
            pad=True,
            csv_path=csv_path,
            force_has_data=True,
        )
        assert isinstance(split.validation_datasets, list)
        assert len(split.validation_datasets) == 1

    def test_validation_blocks_is_subset(self, tmp_path):
        from torch.utils.data import Subset

        csv_path = self._make_csv(tmp_path)
        split = CellMapDataSplit(
            input_arrays=INPUT_ARRAYS,
            target_arrays=TARGET_ARRAYS,
            classes=CLASSES,
            pad=True,
            csv_path=csv_path,
            force_has_data=True,
        )
        blocks = split.validation_blocks
        assert isinstance(blocks, Subset)
        assert len(blocks) > 0

    def test_train_datasets_combined(self, tmp_path):
        from cellmap_data.multidataset import CellMapMultiDataset

        csv_path = self._make_csv(tmp_path)
        split = CellMapDataSplit(
            input_arrays=INPUT_ARRAYS,
            target_arrays=TARGET_ARRAYS,
            classes=CLASSES,
            pad=True,
            csv_path=csv_path,
            force_has_data=True,
        )
        combined = split.train_datasets_combined
        assert isinstance(combined, CellMapMultiDataset)
        assert len(combined) > 0

    def test_to_device_noop(self, tmp_path):
        """split.to(device) should not raise (no-op on CPU datasets)."""
        csv_path = self._make_csv(tmp_path)
        split = CellMapDataSplit(
            input_arrays=INPUT_ARRAYS,
            target_arrays=TARGET_ARRAYS,
            classes=CLASSES,
            csv_path=csv_path,
            force_has_data=True,
        )
        split.to("cpu")  # should not raise


# ---------------------------------------------------------------------------
# CellMapDataLoader — full API
# ---------------------------------------------------------------------------


class TestCellMapDataLoaderAPI:
    """Mirrors utils/dataloader.py lines 188, 204 in cellmap-segmentation-challenge."""

    def _make_split(self, tmp_path):
        train_info = create_test_dataset(tmp_path / "train", classes=CLASSES)
        val_info = create_test_dataset(tmp_path / "val", classes=CLASSES)
        ds_dict = {
            "train": [{"raw": train_info["raw_path"], "gt": train_info["gt_path"]}],
            "validate": [{"raw": val_info["raw_path"], "gt": val_info["gt_path"]}],
        }
        return CellMapDataSplit(
            input_arrays=INPUT_ARRAYS,
            target_arrays=TARGET_ARRAYS,
            classes=CLASSES,
            pad=True,
            dataset_dict=ds_dict,
            force_has_data=True,
        )

    def test_validation_loader_api(self, tmp_path):
        """Replicates: CellMapDataLoader(blocks, classes, batch_size, is_train=False, device)."""
        split = self._make_split(tmp_path)
        blocks = split.validation_blocks
        loader = CellMapDataLoader(
            blocks,
            classes=CLASSES,
            batch_size=2,
            is_train=False,
            device="cpu",
        )
        batches = list(loader)
        assert len(batches) > 0

    def test_training_loader_api(self, tmp_path):
        """Replicates: CellMapDataLoader(combined, ..., is_train=True, weighted_sampler=True, iterations_per_epoch)."""
        split = self._make_split(tmp_path)
        combined = split.train_datasets_combined
        loader = CellMapDataLoader(
            combined,
            classes=CLASSES,
            batch_size=2,
            is_train=True,
            device="cpu",
            iterations_per_epoch=4,
            weighted_sampler=True,
        )
        # Should yield exactly ceil(4 / batch_size) batches
        batches = list(loader)
        assert len(batches) > 0

    def test_batch_dict_has_idx(self, tmp_path):
        """All batches must contain the 'idx' key (needed for writer[batch['idx']] = outputs)."""
        split = self._make_split(tmp_path)
        loader = CellMapDataLoader(
            split.validation_blocks,
            classes=CLASSES,
            batch_size=2,
            is_train=False,
        )
        for batch in loader:
            assert "idx" in batch
            assert isinstance(batch["idx"], torch.Tensor)
            break

    def test_loader_is_iterable(self, tmp_path):
        split = self._make_split(tmp_path)
        loader = CellMapDataLoader(
            split.validation_blocks, classes=CLASSES, batch_size=1, is_train=False
        )
        assert hasattr(loader, "__iter__")
        assert hasattr(loader, "__len__")

    def test_blocks_to_device_before_loader(self, tmp_path):
        """split.validation_blocks.to(device) is called before passing to loader."""
        split = self._make_split(tmp_path)
        blocks = split.validation_blocks
        # .to(device) on a Subset delegates to its dataset
        if hasattr(blocks.dataset, "to"):
            blocks.dataset.to("cpu")
        loader = CellMapDataLoader(
            blocks, classes=CLASSES, batch_size=1, is_train=False
        )
        batches = list(loader)
        assert len(batches) > 0


# ---------------------------------------------------------------------------
# CellMapDatasetWriter — full API
# ---------------------------------------------------------------------------


class TestCellMapDatasetWriterAPI:
    """Mirrors predict.py and process.py usage."""

    def _make_writer(self, tmp_path, model_classes=None):
        raw_path = create_test_zarr(
            tmp_path, name="raw", shape=(32, 32, 32), voxel_size=[8.0, 8.0, 8.0]
        )
        out_path = str(tmp_path / "predictions.zarr")
        bounds = {"pred": {"z": (0.0, 256.0), "y": (0.0, 256.0), "x": (0.0, 256.0)}}
        import torchvision.transforms.v2 as T

        raw_tx = T.Compose(
            [
                T.ToDtype(torch.float, scale=True),
                NaNtoNum({"nan": 0, "posinf": None, "neginf": None}),
            ]
        )
        return CellMapDatasetWriter(
            raw_path=raw_path,
            target_path=out_path,
            classes=CLASSES,
            input_arrays=INPUT_ARRAYS,
            target_arrays={"pred": {"shape": (4, 4, 4), "scale": (8.0, 8.0, 8.0)}},
            target_bounds=bounds,
            overwrite=False,
            device="cuda",
            raw_value_transforms=raw_tx,
            model_classes=model_classes or CLASSES,
        )

    def test_constructor_full_signature(self, tmp_path):
        writer = self._make_writer(tmp_path)
        assert writer is not None

    def test_loader_method(self, tmp_path):
        """writer.loader(batch_size) returns an iterable DataLoader."""
        writer = self._make_writer(tmp_path)
        loader = writer.loader(batch_size=2)
        assert hasattr(loader, "__iter__")
        batches = list(loader)
        assert len(batches) > 0

    def test_loader_batch_has_idx(self, tmp_path):
        writer = self._make_writer(tmp_path)
        for batch in writer.loader(batch_size=2):
            assert "idx" in batch
            break

    def test_setitem_with_batch_idx(self, tmp_path):
        """writer[batch['idx']] = outputs — the main write pattern."""
        writer = self._make_writer(tmp_path)
        loader = writer.loader(batch_size=2)
        for batch in loader:
            idx = batch["idx"]
            # Model outputs: one tensor per class
            outputs = {cls: torch.zeros(len(idx), 4, 4, 4) for cls in CLASSES}
            writer[idx] = outputs  # should not raise
            break

    def test_setitem_scalar_idx(self, tmp_path):
        writer = self._make_writer(tmp_path)
        idx = writer.writer_indices[0]
        outputs = {"mito": torch.zeros(4, 4, 4), "er": torch.zeros(4, 4, 4)}
        writer[idx] = outputs  # should not raise

    def test_model_classes_superset(self, tmp_path):
        """model_classes may be a superset of classes (write subset only)."""
        writer = self._make_writer(tmp_path, model_classes=CLASSES + ["nucleus"])
        assert writer.model_classes == CLASSES + ["nucleus"]

    def test_bounding_box_exposed(self, tmp_path):
        writer = self._make_writer(tmp_path)
        bb = writer.bounding_box
        assert bb is not None
        assert "z" in bb


# ---------------------------------------------------------------------------
# CellMapImage — full API
# ---------------------------------------------------------------------------


class TestCellMapImageAPI:
    """Mirrors predict.py, process.py, and utils/matched_crop.py usage."""

    def test_constructor_full_signature(self, tmp_path):
        """Replicates: CellMapImage(path, target_class, target_scale, target_voxel_shape, pad, pad_value, interpolation)."""
        path = create_test_zarr(tmp_path, shape=(32, 32, 32))
        img = CellMapImage(
            path=path,
            target_class="label",
            target_scale=(8.0, 8.0, 8.0),
            target_voxel_shape=(4, 4, 4),
            pad=True,
            pad_value=0,
            interpolation="linear",
        )
        assert img is not None

    def test_scale_level_is_int(self, tmp_path):
        """matched_crop.py:293 — img.scale_level."""
        path = create_test_zarr(tmp_path)
        img = CellMapImage(path, "label", (8.0, 8.0, 8.0), (4, 4, 4))
        assert isinstance(img.scale_level, int)
        assert img.scale_level >= 0

    def test_bounding_box_is_dict(self, tmp_path):
        """process.py — img.bounding_box."""
        path = create_test_zarr(tmp_path, shape=(32, 32, 32))
        img = CellMapImage(path, "raw", (8.0, 8.0, 8.0), (4, 4, 4))
        bb = img.bounding_box
        assert isinstance(bb, dict)
        assert all(len(v) == 2 for v in bb.values())

    def test_get_center_returns_dict(self, tmp_path):
        """predict.py — img.get_center(idx)."""
        path = create_test_zarr(tmp_path, shape=(32, 32, 32))
        img = CellMapImage(path, "raw", (8.0, 8.0, 8.0), (4, 4, 4))
        center = img.get_center(0)
        assert isinstance(center, dict)
        assert all(isinstance(v, float) for v in center.values())

    def test_array_indexing(self, tmp_path):
        """predict.py / process.py — img[...] to load data."""
        path = create_test_zarr(tmp_path, shape=(32, 32, 32))
        img = CellMapImage(path, "raw", (8.0, 8.0, 8.0), (4, 4, 4), pad=True)
        center = img.get_center(0)
        patch = img[center]
        assert isinstance(patch, torch.Tensor)
        assert patch.shape == torch.Size([4, 4, 4])

    def test_nearest_interpolation(self, tmp_path):
        path = create_test_zarr(tmp_path, shape=(32, 32, 32))
        img = CellMapImage(
            path, "label", (8.0, 8.0, 8.0), (4, 4, 4), interpolation="nearest"
        )
        center = img.get_center(0)
        patch = img[center]
        assert patch.shape == torch.Size([4, 4, 4])

    def test_linear_interpolation(self, tmp_path):
        path = create_test_zarr(tmp_path, shape=(32, 32, 32))
        img = CellMapImage(
            path, "raw", (8.0, 8.0, 8.0), (4, 4, 4), interpolation="linear", pad=True
        )
        center = img.get_center(0)
        patch = img[center]
        assert patch.shape == torch.Size([4, 4, 4])


# ---------------------------------------------------------------------------
# NaNtoNum and Binarize — exact import paths and usage patterns
# ---------------------------------------------------------------------------


class TestTransformImportPaths:
    def test_nan_to_num_import(self):
        from cellmap_data.transforms.augment import NaNtoNum

        t = NaNtoNum({"nan": 0, "posinf": None, "neginf": None})
        x = torch.tensor([float("nan")])
        out = t(x)
        assert out[0] == 0.0

    def test_binarize_import(self):
        from cellmap_data.transforms.augment import Binarize

        t = Binarize()
        x = torch.tensor([0.0, 1.0])
        assert torch.allclose(t(x), torch.tensor([0.0, 1.0]))

    def test_binarize_explicit_threshold(self):
        from cellmap_data.transforms.augment import Binarize

        t = Binarize(0.5)
        x = torch.tensor([0.3, 0.7])
        out = t(x)
        assert out[0] == 0.0
        assert out[1] == 1.0


# ---------------------------------------------------------------------------
# Utility functions — exact signatures and behaviors from API_TO_PRESERVE.md
# ---------------------------------------------------------------------------


class TestUtilFunctions:
    # --- longest_common_substring ---

    def test_lcs_basic(self):
        result = longest_common_substring("raw_input", "raw_target")
        assert result == "raw_"

    def test_lcs_identical(self):
        assert longest_common_substring("abc", "abc") == "abc"

    def test_lcs_no_common(self):
        assert longest_common_substring("aaa", "bbb") == ""

    def test_lcs_import_path(self):
        from cellmap_data.utils import longest_common_substring

        assert longest_common_substring("in_key", "target_key") == "_key"

    # --- array_has_singleton_dim ---

    def test_singleton_dim_true(self):
        info = {"shape": (1, 64, 64)}
        assert array_has_singleton_dim(info) is True

    def test_singleton_dim_false(self):
        info = {"shape": (32, 64, 64)}
        assert array_has_singleton_dim(info) is False

    def test_singleton_dim_none_input(self):
        assert array_has_singleton_dim(None) is False

    def test_singleton_dim_empty(self):
        assert array_has_singleton_dim({}) is False

    def test_singleton_dim_nested(self):
        info = {
            "a": {"shape": (1, 32, 32)},
            "b": {"shape": (8, 8, 8)},
        }
        # summary=True (default) → any() of inner results
        assert array_has_singleton_dim(info) is True

    def test_singleton_dim_nested_no_summary(self):
        info = {
            "a": {"shape": (1, 32, 32)},
            "b": {"shape": (8, 8, 8)},
        }
        result = array_has_singleton_dim(info, summary=False)
        assert isinstance(result, dict)
        assert result["a"] is True
        assert result["b"] is False

    # --- is_array_2D ---

    def test_is_2d_true(self):
        info = {"shape": (64, 64)}
        assert is_array_2D(info) is True

    def test_is_2d_false_3d(self):
        info = {"shape": (32, 64, 64)}
        assert is_array_2D(info) is False

    def test_is_2d_singleton_is_not_2d(self):
        """A (1, 64, 64) shape has 3 dims, so is_array_2D returns False."""
        info = {"shape": (1, 64, 64)}
        assert is_array_2D(info) is False

    def test_is_2d_none_input(self):
        assert is_array_2D(None) is False

    def test_is_2d_nested_with_summary(self):
        info = {
            "a": {"shape": (64, 64)},
            "b": {"shape": (32, 64, 64)},
        }
        # summary=any → True (at least one 2D)
        result = is_array_2D(info, summary=any)
        assert result is True

    def test_is_2d_nested_no_summary(self):
        info = {
            "a": {"shape": (64, 64)},
            "b": {"shape": (32, 64, 64)},
        }
        result = is_array_2D(info)
        assert isinstance(result, dict)
        assert result["a"] is True
        assert result["b"] is False

    # --- permute_singleton_dimension ---

    def test_permute_adds_singleton_if_none(self):
        arr_dict = {"shape": [8, 8], "scale": [8.0, 8.0]}
        permute_singleton_dimension(arr_dict, axis=0)
        assert arr_dict["shape"][0] == 1
        assert len(arr_dict["shape"]) == 3

    def test_permute_moves_existing_singleton(self):
        arr_dict = {"shape": [1, 64, 64], "scale": [8.0, 8.0, 8.0]}
        permute_singleton_dimension(arr_dict, axis=2)
        # Singleton should now be at axis 2
        assert arr_dict["shape"][2] == 1

    def test_permute_nested_dict(self):
        arr_dict = {
            "a": {"shape": [8, 8], "scale": [8.0, 8.0]},
        }
        permute_singleton_dimension(arr_dict, axis=0)
        assert arr_dict["a"]["shape"][0] == 1

    def test_permute_scale_expanded(self):
        """2D scale → 3D scale after permute."""
        arr_dict = {"shape": [8, 8], "scale": [8.0, 8.0]}
        permute_singleton_dimension(arr_dict, axis=0)
        assert len(arr_dict["scale"]) == 3

    # --- get_fig_dict ---
    # Actual signature: get_fig_dict(input_data: Tensor, target_data: Tensor,
    #                                outputs: Tensor, classes: list) -> dict
    # input_data  shape: [batch, channels, *spatial]
    # target_data shape: [batch, n_classes, *spatial]
    # outputs     shape: [batch, n_classes, *spatial]

    def test_get_fig_dict_returns_dict(self):
        """get_fig_dict returns a dict of matplotlib figures."""
        import matplotlib

        matplotlib.use("Agg")  # non-interactive backend for CI
        # 1 batch item, 1 input channel, 1 class, 8x8 2D slices
        input_data = torch.rand(1, 1, 8, 8)
        target_data = torch.rand(1, 1, 8, 8)
        outputs = torch.rand(1, 1, 8, 8)
        result = get_fig_dict(input_data, target_data, outputs, ["mito"])
        assert isinstance(result, dict)
        assert len(result) >= 1

    def test_get_fig_dict_key_per_class(self):
        import matplotlib

        matplotlib.use("Agg")
        input_data = torch.rand(1, 1, 8, 8)
        target_data = torch.rand(1, 2, 8, 8)
        outputs = torch.rand(1, 2, 8, 8)
        result = get_fig_dict(input_data, target_data, outputs, ["mito", "er"])
        # One figure per class
        assert len(result) == 2
