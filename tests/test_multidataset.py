"""Tests for CellMapMultiDataset."""

from __future__ import annotations

import numpy as np
import torch

from cellmap_data import CellMapDataset, CellMapMultiDataset

from .test_helpers import create_test_dataset

INPUT_ARRAYS = {"raw": {"shape": (4, 4, 4), "scale": (8.0, 8.0, 8.0)}}
TARGET_ARRAYS = {"labels": {"shape": (4, 4, 4), "scale": (8.0, 8.0, 8.0)}}
CLASSES = ["mito", "er"]


def _make_ds(tmp_path, suffix="", **kwargs):
    import tempfile, pathlib

    sub = tmp_path / suffix if suffix else tmp_path / "ds0"
    sub.mkdir(parents=True, exist_ok=True)
    info = create_test_dataset(sub, classes=CLASSES, **kwargs)
    return CellMapDataset(
        raw_path=info["raw_path"],
        target_path=info["gt_path"],
        classes=CLASSES,
        input_arrays=INPUT_ARRAYS,
        target_arrays=TARGET_ARRAYS,
        force_has_data=True,
        pad=True,
    )


class TestCellMapMultiDataset:
    def test_len_sum(self, tmp_path):
        ds1 = _make_ds(tmp_path, "d1")
        ds2 = _make_ds(tmp_path, "d2")
        multi = CellMapMultiDataset([ds1, ds2], CLASSES, INPUT_ARRAYS, TARGET_ARRAYS)
        assert len(multi) == len(ds1) + len(ds2)

    def test_getitem_returns_dict(self, tmp_path):
        ds1 = _make_ds(tmp_path, "d1")
        multi = CellMapMultiDataset([ds1], CLASSES, INPUT_ARRAYS, TARGET_ARRAYS)
        item = multi[0]
        assert "raw" in item
        assert "idx" in item

    def test_getitem_index_mapping(self, tmp_path):
        ds1 = _make_ds(tmp_path, "d1")
        ds2 = _make_ds(tmp_path, "d2")
        n1 = len(ds1)
        multi = CellMapMultiDataset([ds1, ds2], CLASSES, INPUT_ARRAYS, TARGET_ARRAYS)
        # Index 0 should come from ds1
        item0 = multi[0]
        assert item0["idx"].item() == 0
        # Index n1 should come from ds2 with local idx 0
        item_n1 = multi[n1]
        assert item_n1["idx"].item() == 0

    def test_class_counts_keys(self, tmp_path):
        ds1 = _make_ds(tmp_path, "d1")
        multi = CellMapMultiDataset([ds1], CLASSES, INPUT_ARRAYS, TARGET_ARRAYS)
        counts = multi.class_counts
        assert "totals" in counts
        assert "totals_total" in counts
        assert all(c in counts["totals"] for c in CLASSES)
        assert all(c in counts["totals_total"] for c in CLASSES)

    def test_class_counts_total_equals_volume_size(self, tmp_path):
        """totals_total should reflect the actual array volume, not sum of fg counts."""
        shape = (8, 8, 8)
        voxel_size = [8.0, 8.0, 8.0]
        ds1 = _make_ds(tmp_path, "d1", shape=shape, voxel_size=voxel_size)
        multi = CellMapMultiDataset([ds1], CLASSES, INPUT_ARRAYS, TARGET_ARRAYS)
        counts = multi.class_counts
        expected_total = int(np.prod(shape))
        for cls in CLASSES:
            assert counts["totals_total"][cls] == expected_total, (
                f"totals_total[{cls!r}] should equal array volume {expected_total}, "
                f"got {counts['totals_total'][cls]}"
            )

    def test_class_weights_bg_uses_volume_size(self, tmp_path):
        """bg in class_weights should be total_voxels - fg, not sum(fg) - fg."""
        shape = (8, 8, 8)
        voxel_size = [8.0, 8.0, 8.0]
        ds1 = _make_ds(tmp_path, "d1", shape=shape, voxel_size=voxel_size)
        multi = CellMapMultiDataset([ds1], CLASSES, INPUT_ARRAYS, TARGET_ARRAYS)
        counts = multi.class_counts
        weights = multi.class_weights
        total = int(np.prod(shape))
        for cls in CLASSES:
            fg = counts["totals"][cls]
            expected_bg = total - fg
            expected_weight = float(max(expected_bg, 0)) / float(max(fg, 1))
            assert abs(weights[cls] - expected_weight) < 1e-6, (
                f"class_weights[{cls!r}] mismatch: expected {expected_weight}, got {weights[cls]}"
            )

    def test_get_crop_class_matrix_shape(self, tmp_path):
        ds1 = _make_ds(tmp_path, "d1")
        ds2 = _make_ds(tmp_path, "d2")
        multi = CellMapMultiDataset([ds1, ds2], CLASSES, INPUT_ARRAYS, TARGET_ARRAYS)
        mat = multi.get_crop_class_matrix()
        assert mat.shape == (2, len(CLASSES))  # 1 row per dataset

    def test_validation_indices_non_empty(self, tmp_path):
        ds1 = _make_ds(tmp_path, "d1")
        multi = CellMapMultiDataset([ds1], CLASSES, INPUT_ARRAYS, TARGET_ARRAYS)
        indices = multi.validation_indices
        assert len(indices) > 0

    def test_validation_indices_within_bounds(self, tmp_path):
        ds1 = _make_ds(tmp_path, "d1")
        ds2 = _make_ds(tmp_path, "d2")
        multi = CellMapMultiDataset([ds1, ds2], CLASSES, INPUT_ARRAYS, TARGET_ARRAYS)
        for idx in multi.validation_indices:
            assert 0 <= idx < len(multi)

    def test_repr(self, tmp_path):
        ds1 = _make_ds(tmp_path, "d1")
        multi = CellMapMultiDataset([ds1], CLASSES, INPUT_ARRAYS, TARGET_ARRAYS)
        r = repr(multi)
        assert "CellMapMultiDataset" in r

    def test_verify(self, tmp_path):
        ds1 = _make_ds(tmp_path, "d1")
        multi = CellMapMultiDataset([ds1], CLASSES, INPUT_ARRAYS, TARGET_ARRAYS)
        assert multi.verify()
