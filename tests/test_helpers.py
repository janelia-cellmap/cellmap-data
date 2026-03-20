"""Helpers for creating minimal zarr test fixtures."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import zarr


def _write_ome_ngff(
    path: str,
    data: np.ndarray,
    voxel_size: list[float],
    *,
    axes: list[str] | None = None,
    origin: list[float] | None = None,
    level: str = "s0",
) -> None:
    """Write a single-level OME-NGFF zarr group."""
    if axes is None:
        axes = ["z", "y", "x"][-data.ndim :]
    if origin is None:
        origin = [0.0] * len(axes)

    os.makedirs(path, exist_ok=True)
    z_attrs = {
        "multiscales": [
            {
                "axes": [
                    {"name": ax, "type": "space", "unit": "nanometer"} for ax in axes
                ],
                "datasets": [
                    {
                        "path": level,
                        "coordinateTransformations": [
                            {"type": "scale", "scale": voxel_size},
                            {"type": "translation", "translation": origin},
                        ],
                    }
                ],
                "version": "0.4",
            }
        ]
    }
    with open(os.path.join(path, ".zattrs"), "w") as f:
        json.dump(z_attrs, f)
    with open(os.path.join(path, ".zgroup"), "w") as f:
        f.write('{"zarr_format": 2}')

    arr_path = os.path.join(path, level)
    zarr.open_array(
        arr_path,
        mode="w",
        shape=data.shape,
        dtype=data.dtype,
        chunks=data.shape,
    )[:] = data


def create_test_zarr(
    tmp_path: Path,
    name: str = "test",
    shape: tuple[int, ...] = (20, 20, 20),
    voxel_size: list[float] | None = None,
    origin: list[float] | None = None,
    data: np.ndarray | None = None,
    axes: list[str] | None = None,
) -> str:
    """Create a minimal OME-NGFF zarr group under *tmp_path*.

    Returns the path to the zarr group (the directory).
    """
    ndim = len(shape)
    if axes is None:
        axes = ["z", "y", "x"][-ndim:]
    if voxel_size is None:
        voxel_size = [8.0] * ndim
    if origin is None:
        origin = [0.0] * ndim
    if data is None:
        rng = np.random.default_rng(0)
        data = (rng.random(shape) * 255).astype(np.uint8)

    path = str(tmp_path / f"{name}.zarr")
    _write_ome_ngff(path, data, voxel_size, axes=axes, origin=origin)
    return path


def create_test_dataset(
    tmp_path: Path,
    classes: list[str] | None = None,
    shape: tuple[int, ...] = (32, 32, 32),
    voxel_size: list[float] | None = None,
) -> dict:
    """Create a minimal raw + label zarr dataset for testing.

    Returns a dict with keys ``raw_path``, ``gt_path``, ``classes``.
    """
    if classes is None:
        classes = ["mito", "er"]
    ndim = len(shape)
    if voxel_size is None:
        voxel_size = [8.0] * ndim
    axes = ["z", "y", "x"][-ndim:]

    rng = np.random.default_rng(42)
    raw_data = (rng.random(shape) * 255).astype(np.uint8)
    raw_path = str(tmp_path / "raw.zarr")
    _write_ome_ngff(raw_path, raw_data, voxel_size, axes=axes)

    gt_base = str(tmp_path / "gt.zarr")
    os.makedirs(gt_base, exist_ok=True)
    with open(os.path.join(gt_base, ".zgroup"), "w") as f:
        f.write('{"zarr_format": 2}')

    for cls in classes:
        label_data = rng.integers(0, 2, size=shape).astype(np.uint8)
        cls_path = os.path.join(gt_base, cls)
        _write_ome_ngff(cls_path, label_data, voxel_size, axes=axes)

    class_str = ",".join(classes)
    gt_path = f"{gt_base}/[{class_str}]"

    return {
        "raw_path": raw_path,
        "gt_path": gt_path,
        "classes": classes,
        "shape": shape,
        "voxel_size": voxel_size,
    }
