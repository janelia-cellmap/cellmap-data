"""Tests for CellMapDataLoader."""

from __future__ import annotations

import torch

from cellmap_data import CellMapDataLoader, CellMapDataset

from .test_helpers import create_test_dataset

INPUT_ARRAYS = {"raw": {"shape": (4, 4, 4), "scale": (8.0, 8.0, 8.0)}}
TARGET_ARRAYS = {"labels": {"shape": (4, 4, 4), "scale": (8.0, 8.0, 8.0)}}
CLASSES = ["mito", "er"]


def _make_ds(tmp_path):
    info = create_test_dataset(tmp_path, classes=CLASSES)
    return CellMapDataset(
        raw_path=info["raw_path"],
        target_path=info["gt_path"],
        classes=CLASSES,
        input_arrays=INPUT_ARRAYS,
        target_arrays=TARGET_ARRAYS,
        force_has_data=True,
        pad=True,
    )


class TestCellMapDataLoader:
    def test_basic_iteration(self, tmp_path):
        ds = _make_ds(tmp_path)
        loader = CellMapDataLoader(ds, classes=CLASSES, batch_size=2, is_train=False)
        batches = list(loader)
        assert len(batches) > 0

    def test_batch_contains_idx(self, tmp_path):
        ds = _make_ds(tmp_path)
        loader = CellMapDataLoader(ds, classes=CLASSES, batch_size=2, is_train=False)
        for batch in loader:
            assert "idx" in batch
            break

    def test_batch_raw_shape(self, tmp_path):
        ds = _make_ds(tmp_path)
        loader = CellMapDataLoader(ds, classes=CLASSES, batch_size=2, is_train=False)
        for batch in loader:
            raw = batch["raw"]
            assert isinstance(raw, torch.Tensor)
            # batch_size is 2 but last batch may be smaller
            assert raw.shape[1:] == torch.Size([1, 4, 4, 4])
            break

    def test_len(self, tmp_path):
        ds = _make_ds(tmp_path)
        loader = CellMapDataLoader(ds, classes=CLASSES, batch_size=1, is_train=False)
        assert len(loader) > 0

    def test_weighted_sampler_train(self, tmp_path):
        from cellmap_data.sampler import ClassBalancedSampler

        ds = _make_ds(tmp_path)
        loader = CellMapDataLoader(
            ds, classes=CLASSES, batch_size=2, is_train=True, weighted_sampler=True
        )
        assert isinstance(loader._sampler, ClassBalancedSampler)

    def test_no_weighted_sampler_val(self, tmp_path):
        ds = _make_ds(tmp_path)
        loader = CellMapDataLoader(
            ds,
            classes=CLASSES,
            batch_size=2,
            is_train=False,
            weighted_sampler=True,
        )
        # weighted_sampler is only used when is_train=True
        assert loader._sampler is None

    def test_collate_fn_stacks_tensors(self):
        batch = [
            {"idx": torch.tensor(0), "raw": torch.zeros(4, 4, 4)},
            {"idx": torch.tensor(1), "raw": torch.ones(4, 4, 4)},
        ]
        result = CellMapDataLoader.collate_fn(batch)
        assert result["raw"].shape == torch.Size([2, 4, 4, 4])
        assert result["idx"].shape == torch.Size([2])

    def test_repr(self, tmp_path):
        ds = _make_ds(tmp_path)
        loader = CellMapDataLoader(ds, classes=CLASSES, batch_size=1, is_train=False)
        r = repr(loader)
        assert "CellMapDataLoader" in r
