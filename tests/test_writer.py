"""Tests for CellMapDatasetWriter and ImageWriter."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cellmap_data import CellMapDatasetWriter
from cellmap_data.image_writer import ImageWriter

from .test_helpers import create_test_zarr

INPUT_ARRAYS = {"raw": {"shape": (4, 4, 4), "scale": (8.0, 8.0, 8.0)}}
TARGET_ARRAYS = {"pred": {"shape": (4, 4, 4), "scale": (8.0, 8.0, 8.0)}}


class TestImageWriter:
    def test_write_and_read_back(self, tmp_path):
        out_path = str(tmp_path / "out.zarr" / "mito")
        bounding_box = {"z": (0.0, 128.0), "y": (0.0, 128.0), "x": (0.0, 128.0)}
        writer = ImageWriter(
            path=out_path,
            target_class="mito",
            scale={"z": 8.0, "y": 8.0, "x": 8.0},
            bounding_box=bounding_box,
            write_voxel_shape={"z": 4, "y": 4, "x": 4},
            overwrite=True,
        )
        data = torch.ones(4, 4, 4) * 0.5
        center = {"z": 16.0, "y": 16.0, "x": 16.0}
        writer[center] = data
        # Read back
        readback = writer[center]
        assert torch.allclose(readback, torch.ones(4, 4, 4) * 0.5, atol=1e-4)

    def test_shape_property(self, tmp_path):
        out_path = str(tmp_path / "out.zarr" / "mito")
        writer = ImageWriter(
            path=out_path,
            target_class="mito",
            scale={"z": 8.0, "y": 8.0, "x": 8.0},
            bounding_box={"z": (0.0, 128.0), "y": (0.0, 128.0), "x": (0.0, 128.0)},
            write_voxel_shape={"z": 4, "y": 4, "x": 4},
            overwrite=True,
        )
        # 128 nm / 8 nm/voxel = 16 voxels per axis
        assert writer.shape == {"z": 16, "y": 16, "x": 16}

    def test_repr(self, tmp_path):
        writer = ImageWriter(
            path=str(tmp_path / "out.zarr" / "mito"),
            target_class="mito",
            scale={"z": 8.0, "y": 8.0, "x": 8.0},
            bounding_box={"z": (0.0, 64.0), "y": (0.0, 64.0), "x": (0.0, 64.0)},
            write_voxel_shape={"z": 4, "y": 4, "x": 4},
        )
        assert "ImageWriter" in repr(writer)


class TestCellMapDatasetWriter:
    def _make_writer(self, tmp_path):
        raw_path = create_test_zarr(
            tmp_path, name="raw", shape=(32, 32, 32), voxel_size=[8.0, 8.0, 8.0]
        )
        out_path = str(tmp_path / "predictions.zarr")
        bounds = {"pred": {"z": (0.0, 256.0), "y": (0.0, 256.0), "x": (0.0, 256.0)}}
        writer = CellMapDatasetWriter(
            raw_path=raw_path,
            target_path=out_path,
            classes=["mito"],
            input_arrays=INPUT_ARRAYS,
            target_arrays=TARGET_ARRAYS,
            target_bounds=bounds,
            overwrite=True,
        )
        return writer

    def test_len_positive(self, tmp_path):
        writer = self._make_writer(tmp_path)
        assert len(writer) > 0

    def test_bounding_box(self, tmp_path):
        writer = self._make_writer(tmp_path)
        bb = writer.bounding_box
        assert bb is not None
        assert "z" in bb

    def test_getitem_returns_dict_with_idx(self, tmp_path):
        writer = self._make_writer(tmp_path)
        item = writer[0]
        assert "idx" in item
        assert isinstance(item["raw"], torch.Tensor)

    def test_writer_indices_non_empty(self, tmp_path):
        writer = self._make_writer(tmp_path)
        assert len(writer.writer_indices) > 0

    def test_setitem_scalar(self, tmp_path):
        """Writing a single prediction should not raise."""
        writer = self._make_writer(tmp_path)
        idx = writer.writer_indices[0]
        output = {"mito": torch.zeros(4, 4, 4)}
        writer[idx] = output  # should not raise

    def test_setitem_batch(self, tmp_path):
        """Writing a batch (tensor of indices) should not raise."""
        writer = self._make_writer(tmp_path)
        indices = writer.writer_indices[:2]
        idx_tensor = torch.tensor(indices)
        # Batch of predictions: [batch, *spatial]
        output = {"mito": torch.zeros(2, 4, 4, 4)}
        writer[idx_tensor] = output  # should not raise

    def test_setitem_batch_scalar_raises(self, tmp_path):
        """Passing a scalar value in a batch write must raise TypeError."""
        writer = self._make_writer(tmp_path)
        indices = writer.writer_indices[:2]
        idx_tensor = torch.tensor(indices)
        # Scalar instead of a batched array — should raise
        output = {"mito": 1.0}
        with pytest.raises(TypeError, match="Scalar writes are not supported"):
            writer[idx_tensor] = output

    def test_loader_iterable(self, tmp_path):
        writer = self._make_writer(tmp_path)
        loader = writer.loader(batch_size=2)
        batches = list(loader)
        assert len(batches) > 0
        assert "idx" in batches[0]

    def test_repr(self, tmp_path):
        writer = self._make_writer(tmp_path)
        r = repr(writer)
        assert "CellMapDatasetWriter" in r
