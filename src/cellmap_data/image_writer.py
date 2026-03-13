"""ImageWriter: writes patch data back to a zarr array at world coordinates."""

from __future__ import annotations

import logging
import os
from functools import cached_property
from typing import Mapping, Optional, Sequence

import numpy as np
import torch
import zarr
from numpy.typing import ArrayLike
from upath import UPath

from .utils.metadata import create_multiscale_metadata

logger = logging.getLogger(__name__)


class ImageWriter:
    """Write patches of a single class to a single-resolution zarr array.

    Parameters
    ----------
    path:
        Base path of the zarr group (e.g. ``/out/predictions.zarr/mito``).
    target_class:
        Semantic label (e.g. ``"mito"``).
    scale:
        Voxel size in nm per spatial axis (dict or sequence in *axis_order*).
    bounding_box:
        World bounding box ``{axis: (min_nm, max_nm)}``.
    write_voxel_shape:
        Patch size in voxels (dict or sequence in *axis_order*).
    scale_level:
        Scale level index to write (default 0 = full resolution).
    axis_order:
        Spatial axis names.
    overwrite:
        If ``True``, existing data at *path* is overwritten.
    dtype:
        Output dtype (default ``float32``).
    fill_value:
        Value to pre-fill the output array with (default ``0``).
    """

    def __init__(
        self,
        path: str | UPath,
        target_class: str,
        scale: Mapping[str, float] | Sequence[float],
        bounding_box: Mapping[str, Sequence[float]],
        write_voxel_shape: Mapping[str, int] | Sequence[int],
        scale_level: int = 0,
        axis_order: str = "zyx",
        context: Optional[object] = None,  # ignored – kept for API compat
        overwrite: bool = False,
        dtype: np.dtype = np.float32,
        fill_value: float | int = 0,
    ) -> None:
        self.base_path = str(path)
        self.label_class = self.target_class = target_class
        self.scale_level = scale_level
        self.overwrite = overwrite
        self.dtype = dtype
        self.fill_value = fill_value

        if isinstance(scale, Sequence):
            scale = {c: float(s) for c, s in zip(axis_order, scale)}
        self.scale: dict[str, float] = dict(scale)

        self.axes: str = axis_order
        self.spatial_axes: list[str] = list(axis_order[-len(self.scale) :])

        if isinstance(write_voxel_shape, Sequence):
            if len(axis_order) > len(write_voxel_shape):
                write_voxel_shape = [1] * (
                    len(axis_order) - len(write_voxel_shape)
                ) + list(write_voxel_shape)
            write_voxel_shape = {
                c: int(t) for c, t in zip(axis_order, write_voxel_shape)
            }
        self.write_voxel_shape: dict[str, int] = dict(write_voxel_shape)
        self.write_world_shape: dict[str, float] = {
            c: self.write_voxel_shape[c] * self.scale[c] for c in self.spatial_axes
        }

        self.bounding_box: dict[str, tuple[float, float]] = {
            c: (float(bounding_box[c][0]), float(bounding_box[c][1]))
            for c in self.spatial_axes
        }

    # ------------------------------------------------------------------
    # Cached properties
    # ------------------------------------------------------------------

    @cached_property
    def offset(self) -> dict[str, float]:
        return {c: self.bounding_box[c][0] for c in self.spatial_axes}

    @cached_property
    def world_shape(self) -> dict[str, float]:
        return {
            c: self.bounding_box[c][1] - self.bounding_box[c][0]
            for c in self.spatial_axes
        }

    @cached_property
    def shape(self) -> dict[str, int]:
        return {
            c: int(np.ceil(self.world_shape[c] / self.scale[c]))
            for c in self.spatial_axes
        }

    @cached_property
    def chunk_shape(self) -> list[int]:
        return [self.write_voxel_shape[c] for c in self.spatial_axes]

    @cached_property
    def array_path(self) -> str:
        return str(UPath(self.base_path) / f"s{self.scale_level}")

    @cached_property
    def _zarr_array(self) -> zarr.Array:
        """Open (creating if necessary) the output zarr array."""
        os.makedirs(str(UPath(self.base_path)), exist_ok=True)

        # Ensure every ancestor group has a .zgroup
        group_path = str(self.base_path).split(".zarr")[0] + ".zarr"
        inner = UPath(str(self.base_path).split(".zarr")[-1])
        for part in [""] + list(inner.parts)[1:]:
            gp = str(UPath(group_path) / part)
            zgroup = UPath(gp) / ".zgroup"
            if not zgroup.exists():
                os.makedirs(gp, exist_ok=True)
                zgroup.write_text('{"zarr_format": 2}')
            group_path = gp

        # Write OME-NGFF multiscale metadata
        create_multiscale_metadata(
            ds_name=self.base_path,
            voxel_size=[self.scale[c] for c in self.spatial_axes],
            translation=[self.offset[c] for c in self.spatial_axes],
            units="nanometer",
            axes=self.spatial_axes,
            base_scale_level=self.scale_level,
            levels_to_add=0,
            out_path=str(UPath(self.base_path) / ".zattrs"),
        )

        total_shape = [self.shape[c] for c in self.spatial_axes]
        arr = zarr.open_array(
            self.array_path,
            mode="w" if self.overwrite else "a",
            shape=total_shape,
            dtype=self.dtype,
            chunks=self.chunk_shape,
            fill_value=self.fill_value,
        )
        # Empty attrs for scale-level array
        with open(str(UPath(self.array_path) / ".zattrs"), "w") as f:
            f.write("{}")
        return arr

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def __setitem__(
        self,
        coords: Mapping[str, float] | Mapping[str, Sequence],
        data: torch.Tensor | ArrayLike,
    ) -> None:
        """Write *data* at the location given by *coords*.

        *coords* can be:
        - ``{axis: float}`` centre coordinates — single patch.
        - ``{axis: Sequence[float]}`` centres — batch.

        Raises
        ------
        TypeError
            If *data* is a scalar (i.e. ``np.isscalar(data)`` is ``True``, including
            Python and NumPy scalar types). Use a non-scalar array or tensor with
            shape matching the patch instead. Zero-dimensional arrays/tensors are
            also not supported for writes.
        """
        if np.isscalar(data):
            raise TypeError(
                "Scalar writes are not supported. "
                "Pass an array or tensor with shape matching the patch."
            )
        first = next(iter(coords.values()))
        if isinstance(first, (int, float)):
            self._write_single(coords, data)  # type: ignore[arg-type]
        else:
            self._write_batch(coords, data)  # type: ignore[arg-type]

    def _write_single(
        self,
        center: Mapping[str, float],
        data: torch.Tensor | ArrayLike,
    ) -> None:
        arr = self._zarr_array
        arr_shape = [self.shape[c] for c in self.spatial_axes]

        slices: list[slice] = []
        src_starts: list[int] = []
        for i, c in enumerate(self.spatial_axes):
            start_nm = center[c] - self.write_world_shape[c] / 2.0
            start_vox = int(round((start_nm - self.offset[c]) / self.scale[c]))
            end_vox = start_vox + self.write_voxel_shape[c]
            clamp_start = max(0, start_vox)
            clamp_end = min(arr_shape[i], end_vox)
            # Where the visible region starts inside the source patch along this axis
            src_start = clamp_start - start_vox
            slices.append(slice(clamp_start, clamp_end))
            src_starts.append(src_start)

        if isinstance(data, torch.Tensor):
            data_np = data.detach().cpu().numpy()
        else:
            data_np = np.asarray(data)

        if data_np.ndim == 0:
            raise TypeError(
                "Scalar writes are not supported. "
                "Pass an array or tensor with shape matching the patch."
            )

        data_np = data_np.astype(self.dtype)

        # Strip batch / channel leading dims of size 1
        while data_np.ndim > len(self.spatial_axes) and data_np.shape[0] == 1:
            data_np = data_np.squeeze(0)

        # Crop data to clamped region (near array edges)
        actual = tuple(s.stop - s.start for s in slices)
        if data_np.shape != actual:
            # Use per-axis offsets so that when start_vox < 0, we skip the out-of-bounds prefix
            data_np = data_np[
                tuple(
                    slice(src_starts[i], src_starts[i] + actual[i])
                    for i in range(len(self.spatial_axes))
                )
            ]

        arr[tuple(slices)] = data_np

    def _write_batch(
        self,
        batch_coords: Mapping[str, Sequence],
        data: torch.Tensor | ArrayLike,
    ) -> None:
        n = len(next(iter(batch_coords.values())))
        for i in range(n):
            center = {ax: float(batch_coords[ax][i]) for ax in self.spatial_axes}
            self._write_single(center, data[i])  # type: ignore[index]

    def __getitem__(self, coords: Mapping[str, float]) -> torch.Tensor:
        """Read the patch centred at *coords*."""
        arr = self._zarr_array
        arr_shape = [self.shape[c] for c in self.spatial_axes]
        slices: list[slice] = []
        for i, c in enumerate(self.spatial_axes):
            start_nm = coords[c] - self.write_world_shape[c] / 2.0
            start_vox = int(round((start_nm - self.offset[c]) / self.scale[c]))
            end_vox = start_vox + self.write_voxel_shape[c]
            slices.append(slice(max(0, start_vox), min(arr_shape[i], end_vox)))
        return torch.from_numpy(np.array(arr[tuple(slices)]))

    def __repr__(self) -> str:
        return (
            f"ImageWriter({self.base_path!r}: {self.label_class!r} "
            f"@ {list(self.scale.values())} nm)"
        )
