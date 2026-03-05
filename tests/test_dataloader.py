"""Tests for CellMapDataLoader."""

from __future__ import annotations

import numpy as np
import torch

from cellmap_data import CellMapDataLoader, CellMapDataset
from cellmap_data.dataloader import _collect_datasets

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

    # ------------------------------------------------------------------
    # Seeding / determinism
    # ------------------------------------------------------------------

    def test_loader_seeds_dataset_rng(self, tmp_path):
        """CellMapDataLoader must seed _rng from torch.initial_seed()."""
        ds = _make_ds(tmp_path)
        torch.manual_seed(7)
        CellMapDataLoader(ds, classes=CLASSES, batch_size=1, is_train=False,
                          device="cpu")
        rng_state_7 = ds._rng.random()

        ds2 = _make_ds(tmp_path)
        torch.manual_seed(7)
        CellMapDataLoader(ds2, classes=CLASSES, batch_size=1, is_train=False,
                          device="cpu")
        rng_state_7b = ds2._rng.random()

        ds3 = _make_ds(tmp_path)
        torch.manual_seed(99)
        CellMapDataLoader(ds3, classes=CLASSES, batch_size=1, is_train=False,
                          device="cpu")
        rng_state_99 = ds3._rng.random()

        assert rng_state_7 == rng_state_7b, "same seed must yield same rng state"
        assert rng_state_7 != rng_state_99, "different seeds must yield different rng state"

    def test_augmentation_reproducible_same_seed(self, tmp_path):
        """Same torch seed → identical augmented batches across two loader runs."""
        SPATIAL = {"mirror": {"z": 0.5, "y": 0.5, "x": 0.5}}
        info = create_test_dataset(tmp_path, classes=["mito"])

        def get_first_batch(seed):
            ds = CellMapDataset(
                raw_path=info["raw_path"],
                target_path=info["gt_path"],
                classes=info["classes"],
                input_arrays=INPUT_ARRAYS,
                target_arrays=TARGET_ARRAYS,
                spatial_transforms=SPATIAL,
                force_has_data=True,
                pad=True,
            )
            torch.manual_seed(seed)
            loader = CellMapDataLoader(ds, classes=info["classes"], batch_size=1,
                                       is_train=False, device="cpu")
            return next(iter(loader))["raw"]

        b1 = get_first_batch(42)
        b2 = get_first_batch(42)
        b3 = get_first_batch(99)
        assert torch.allclose(b1, b2), "same seed must produce identical augmentation"
        assert not torch.allclose(b1, b3), "different seeds must produce different augmentation"

    def test_dataset_seed_param(self, tmp_path):
        """CellMapDataset(seed=N) seeds _rng at construction."""
        info = create_test_dataset(tmp_path)
        ds_a = CellMapDataset(
            raw_path=info["raw_path"], target_path=info["gt_path"],
            classes=info["classes"], input_arrays=INPUT_ARRAYS,
            target_arrays=TARGET_ARRAYS, force_has_data=True, seed=123,
        )
        ds_b = CellMapDataset(
            raw_path=info["raw_path"], target_path=info["gt_path"],
            classes=info["classes"], input_arrays=INPUT_ARRAYS,
            target_arrays=TARGET_ARRAYS, force_has_data=True, seed=123,
        )
        ds_c = CellMapDataset(
            raw_path=info["raw_path"], target_path=info["gt_path"],
            classes=info["classes"], input_arrays=INPUT_ARRAYS,
            target_arrays=TARGET_ARRAYS, force_has_data=True, seed=456,
        )
        v_a = ds_a._rng.random()
        v_b = ds_b._rng.random()
        v_c = ds_c._rng.random()
        assert v_a == v_b, "same seed must give same first draw"
        assert v_a != v_c, "different seeds must give different first draw"

    def test_collect_datasets_flat(self, tmp_path):
        """_collect_datasets on a single CellMapDataset returns that dataset."""
        ds = _make_ds(tmp_path)
        collected = _collect_datasets(ds)
        assert collected == [ds]

    def test_collect_datasets_multidataset(self, tmp_path):
        """_collect_datasets traverses CellMapMultiDataset."""
        from cellmap_data import CellMapMultiDataset

        info = create_test_dataset(tmp_path)
        ds1 = CellMapDataset(
            raw_path=info["raw_path"], target_path=info["gt_path"],
            classes=info["classes"], input_arrays=INPUT_ARRAYS,
            target_arrays=TARGET_ARRAYS, force_has_data=True,
        )
        ds2 = CellMapDataset(
            raw_path=info["raw_path"], target_path=info["gt_path"],
            classes=info["classes"], input_arrays=INPUT_ARRAYS,
            target_arrays=TARGET_ARRAYS, force_has_data=True,
        )
        multi = CellMapMultiDataset([ds1, ds2], info["classes"], INPUT_ARRAYS, TARGET_ARRAYS)
        collected = _collect_datasets(multi)
        assert set(id(d) for d in collected) == {id(ds1), id(ds2)}
