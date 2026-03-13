"""CellMapDatasetWriter: writes model predictions to zarr."""

from __future__ import annotations

import logging
from functools import cached_property
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from .image import CellMapImage
from .image_writer import ImageWriter
from .utils.geometry import box_shape, box_union

logger = logging.getLogger(__name__)

_SKIP_KEYS = frozenset({"idx", "__metadata__"})


class CellMapDatasetWriter(Dataset):
    """Writes model predictions back into zarr arrays at world coordinates.

    Parameters
    ----------
    raw_path:
        Path to the raw EM zarr group (for reading input patches).
    target_path:
        Base path for output zarr groups (class sub-groups are written
        under this path).
    classes:
        Classes to write.
    input_arrays:
        Input array specs for reading raw EM patches.
    target_arrays:
        Output array specs (shape/scale) for each prediction.
    target_bounds:
        ``{array_name: {axis: (min_nm, max_nm)}}`` bounding boxes for
        each target array.  Determines the spatial extent of the output.
    overwrite:
        If ``True``, existing output data is overwritten.
    device:
        Ignored (API compatibility).
    raw_value_transforms:
        Value transform applied to raw input patches.
    model_classes:
        Full list of classes the model was trained on (superset of
        *classes*).  Used to map model outputs to the correct channel.
    """

    def __init__(
        self,
        raw_path: str,
        target_path: str,
        classes: Sequence[str],
        input_arrays: Mapping[str, Mapping[str, Any]],
        target_arrays: Mapping[str, Mapping[str, Any]],
        target_bounds: Optional[Mapping[str, Mapping[str, Sequence[float]]]] = None,
        overwrite: bool = False,
        device: Optional[str | torch.device] = None,
        raw_value_transforms: Optional[Callable] = None,
        model_classes: Optional[Sequence[str]] = None,
        axis_order: str = "zyx",
        context: Optional[Any] = None,  # ignored – API compat
        **kwargs: Any,
    ) -> None:
        self.raw_path = raw_path
        self.target_path = target_path
        self.classes = list(classes)
        self.input_arrays = dict(input_arrays)
        self.target_arrays = dict(target_arrays)
        self.target_bounds = dict(target_bounds) if target_bounds else {}
        self.overwrite = overwrite
        self.raw_value_transforms = raw_value_transforms
        self.model_classes = list(model_classes) if model_classes else list(classes)
        self.axis_order = axis_order

        # Build input sources
        self.input_sources: dict[str, CellMapImage] = {}
        for arr_name, arr_spec in self.input_arrays.items():
            self.input_sources[arr_name] = CellMapImage(
                path=raw_path,
                target_class=arr_name,
                target_scale=arr_spec["scale"],
                target_voxel_shape=arr_spec["shape"],
                pad=True,
                pad_value=0.0,
                interpolation="linear",
                value_transform=raw_value_transforms,
            )

        # Build output ImageWriter instances per (target_array, class)
        self.target_array_writers: dict[str, dict[str, ImageWriter]] = {}
        for arr_name, arr_spec in self.target_arrays.items():
            bounds = self.target_bounds.get(arr_name, {})
            self.target_array_writers[arr_name] = {}
            for cls in self.classes:
                writer_path = f"{target_path}/{cls}"
                self.target_array_writers[arr_name][cls] = ImageWriter(
                    path=writer_path,
                    target_class=cls,
                    scale=arr_spec["scale"],
                    bounding_box=bounds,
                    write_voxel_shape=arr_spec["shape"],
                    axis_order=axis_order,
                    overwrite=overwrite,
                )

    # ------------------------------------------------------------------
    # Spatial properties
    # ------------------------------------------------------------------

    @cached_property
    def bounding_box(self) -> dict[str, tuple[float, float]] | None:
        """Union of all target bounds."""
        result = None
        for bounds in self.target_bounds.values():
            box = {ax: (float(bounds[ax][0]), float(bounds[ax][1])) for ax in bounds}
            result = box if result is None else box_union(result, box)
        return result

    @cached_property
    def _write_scale(self) -> dict[str, float]:
        """Scale of the first target array."""
        first_spec = next(iter(self.target_arrays.values()))
        axes = list(self.axis_order[-len(first_spec["scale"]) :])
        return {c: float(s) for c, s in zip(axes, first_spec["scale"])}

    @cached_property
    def _write_voxel_shape(self) -> dict[str, int]:
        first_spec = next(iter(self.target_arrays.values()))
        axes = list(self.axis_order[-len(first_spec["shape"]) :])
        return {c: int(t) for c, t in zip(axes, first_spec["shape"])}

    @cached_property
    def sampling_box(self) -> dict[str, tuple[float, float]] | None:
        """Bounding box shrunk by half the write patch size."""
        bb = self.bounding_box
        if bb is None:
            return None
        result: dict[str, tuple[float, float]] = {}
        half = {
            ax: self._write_scale[ax] * self._write_voxel_shape[ax] / 2.0
            for ax in self._write_scale
        }
        for ax in bb:
            h = half.get(ax, 0.0)
            lo = bb[ax][0] + h
            hi = bb[ax][1] - h
            if lo >= hi:
                return None
            result[ax] = (lo, hi)
        return result

    @cached_property
    def writer_indices(self) -> list[int]:
        """Non-overlapping tile indices covering the sampling box."""
        return self.get_indices(
            {
                ax: self._write_scale[ax] * self._write_voxel_shape[ax]
                for ax in self._write_scale
            }
        )

    @cached_property
    def blocks(self) -> Subset:
        """Subset of this dataset covering non-overlapping write tiles."""
        return Subset(self, self.writer_indices)

    def __len__(self) -> int:
        sb = self.sampling_box
        if sb is None:
            return 0
        grid = box_shape(sb, self._write_scale)
        total = 1
        for v in grid.values():
            total *= v
        return total

    def get_center(self, idx: int) -> dict[str, float]:
        """World centre coordinates for flat index *idx*."""
        sb = self.sampling_box
        if sb is None:
            raise IndexError("sampling_box is None")
        scale = self._write_scale
        grid = box_shape(sb, scale)
        axes = list(sb.keys())
        shape_tuple = tuple(grid[ax] for ax in axes)
        vox_idx = np.unravel_index(int(idx) % max(1, len(self)), shape_tuple)
        return {
            ax: sb[ax][0] + (vox_idx[i] + 0.5) * scale[ax] for i, ax in enumerate(axes)
        }

    def get_indices(self, chunk_size: Mapping[str, float]) -> list[int]:
        """Flat indices tiling the sampling box with chunk_size steps."""
        sb = self.sampling_box
        if sb is None:
            return []
        scale = self._write_scale
        grid = box_shape(sb, scale)
        axes = list(sb.keys())
        shape_tuple = tuple(grid[ax] for ax in axes)

        chunk_grid = {
            ax: max(
                1, int(round((sb[ax][1] - sb[ax][0]) / chunk_size.get(ax, scale[ax])))
            )
            for ax in axes
        }
        chunk_tuple = tuple(chunk_grid[ax] for ax in axes)
        indices = []
        for chunk_idx in np.ndindex(*chunk_tuple):
            vox_idx = tuple(
                int(chunk_idx[i] * shape_tuple[i] / chunk_tuple[i])
                for i in range(len(axes))
            )
            flat = int(np.ravel_multi_index(vox_idx, shape_tuple))
            indices.append(flat)
        return indices

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return input patches for index *idx* (for DataLoader iteration)."""
        center = self.get_center(int(idx))
        result: dict[str, Any] = {"idx": torch.tensor(idx)}
        for arr_name, src in self.input_sources.items():
            patch = src[center]
            if patch.ndim > 0 and patch.shape[0] != 1:
                patch = patch.unsqueeze(0)
            result[arr_name] = patch
        return result

    def __setitem__(
        self,
        idx: int | torch.Tensor | np.ndarray | Sequence[int],
        arrays: dict[str, torch.Tensor | np.ndarray],
    ) -> None:
        """Write prediction *arrays* at the spatial location of *idx*.

        *idx* can be a scalar or a 1-D batch tensor.  *arrays* is a dict
        ``{class_name: tensor}`` (or ``{array_name: {class_name: tensor}}``).
        """
        if isinstance(idx, torch.Tensor):
            idx = idx.cpu().numpy()
        if isinstance(idx, (np.ndarray, list, tuple)) and np.ndim(idx) > 0:
            for batch_i, i in enumerate(idx):
                item: dict[str, Any] = {}
                for key, val in arrays.items():
                    if key in _SKIP_KEYS:
                        continue
                    if np.isscalar(val):
                        raise TypeError(
                            f"Scalar writes are not supported (key={key!r}). "
                            "Pass an array or tensor with a leading batch dimension."
                        )
                    if isinstance(val, dict):
                        item[key] = {k: v[batch_i] for k, v in val.items()}
                    else:
                        item[key] = val[batch_i]
                self.__setitem__(int(i), item)
            return

        center = self.get_center(int(idx))

        for key, val in arrays.items():
            if key in _SKIP_KEYS:
                continue
            # Find which target array and class this key maps to
            for arr_name, writers in self.target_array_writers.items():
                if key in writers:
                    writers[key][center] = val
                elif key in self.classes:
                    # Flat class key — write to first matching target array
                    if key in writers:
                        writers[key][center] = val
                    else:
                        # Write per channel if val is multi-channel
                        cls_idx = (
                            self.model_classes.index(key)
                            if key in self.model_classes
                            else None
                        )
                        if key in writers:
                            writers[key][center] = (
                                val[cls_idx]
                                if cls_idx is not None and val.ndim > 0
                                else val
                            )
                break

    # ------------------------------------------------------------------
    # DataLoader helper
    # ------------------------------------------------------------------

    def loader(
        self,
        batch_size: int = 1,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> Any:
        """Return a DataLoader that iterates over non-overlapping write tiles."""
        from .dataloader import CellMapDataLoader

        return CellMapDataLoader(
            Subset(self, self.writer_indices),
            batch_size=batch_size,
            num_workers=num_workers,
            is_train=False,
            **kwargs,
        ).loader

    def __repr__(self) -> str:
        return (
            f"CellMapDatasetWriter(target={self.target_path!r}, "
            f"classes={self.classes}, len={len(self)})"
        )
