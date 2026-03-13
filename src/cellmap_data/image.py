"""CellMapImage: reads patches from OME-NGFF multiscale zarr arrays."""

from __future__ import annotations

import logging
from functools import cached_property
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import zarr
from scipy.ndimage import rotate as scipy_rotate

from .utils.geometry import box_shape

logger = logging.getLogger(__name__)


class CellMapImage:
    """Load a patch from a single OME-NGFF multiscale zarr array.

    Parameters
    ----------
    path:
        Path to the zarr group (e.g. ``/data/jrc_hela-2.zarr/raw``).
    target_class:
        Semantic label for this array (e.g. ``"raw"``, ``"mito"``).
    target_scale:
        Desired voxel size in nm per axis, ordered according to *axis_order*.
    target_voxel_shape:
        Output patch size in voxels, ordered according to *axis_order*.
    pad:
        Whether to pad with *pad_value* when the patch extends beyond the
        array bounds.  If ``False`` the center is clamped so patches never
        extend outside.
    pad_value:
        Fill value for out-of-bounds regions.  Defaults to ``nan`` so the
        model can mask unknown data in the loss.
    interpolation:
        ``"linear"`` for bilinear/trilinear resampling (raw EM),
        ``"nearest"`` for nearest-neighbour (labels).
    axis_order:
        Axis names in the order they appear in *target_scale* /
        *target_voxel_shape*.  Defaults to ``"zyx"``.
    value_transform:
        Optional callable applied to the output tensor (e.g. ``Binarize``).
    context:
        Ignored (kept for API compatibility with the old TensorStore code).
    device:
        Ignored – tensors are returned on CPU; the DataLoader moves them.
    """

    def __init__(
        self,
        path: str,
        target_class: str,
        target_scale: Sequence[float],
        target_voxel_shape: Sequence[int],
        pad: bool = False,
        pad_value: float = float("nan"),
        interpolation: str = "nearest",
        axis_order: str | Sequence[str] = "zyx",
        value_transform: Optional[Callable] = None,
        context: Any = None,
        device: Optional[str | torch.device] = None,
    ) -> None:
        self.path = path
        self.label_class = target_class
        axis_order = list(axis_order)

        # Pad scale / shape to match axis_order length (preserve existing behaviour)
        if len(axis_order) > len(target_scale):
            target_scale = [target_scale[0]] * (
                len(axis_order) - len(target_scale)
            ) + list(target_scale)
        if len(axis_order) > len(target_voxel_shape):
            ndim_fix = len(axis_order) - len(target_voxel_shape)
            target_voxel_shape = [1] * ndim_fix + list(target_voxel_shape)

        self.pad = pad
        self.pad_value = pad_value
        self.interpolation = interpolation
        self.axes: list[str] = axis_order[: len(target_voxel_shape)]
        self.scale: dict[str, float] = {
            c: float(s) for c, s in zip(axis_order, target_scale)
        }
        self.output_shape: dict[str, int] = {
            c: int(t) for c, t in zip(axis_order, target_voxel_shape)
        }
        self.output_size: dict[str, float] = {
            c: self.output_shape[c] * self.scale[c] for c in self.axes
        }
        self.value_transform = value_transform
        self._current_spatial_transforms: Optional[dict] = None

    # ------------------------------------------------------------------
    # Zarr / OME-NGFF metadata (all cached after first access)
    # ------------------------------------------------------------------

    @cached_property
    def _zarr_group(self) -> zarr.Group:
        return zarr.open_group(self.path, mode="r")

    @cached_property
    def _multiscale_attrs(self) -> dict:
        return dict(self._zarr_group.attrs)["multiscales"][0]

    @cached_property
    def _spatial_axes_order(self) -> list[str]:
        """Spatial axis names as declared in the multiscale metadata."""
        result = []
        for ax in self._multiscale_attrs.get("axes", []):
            if isinstance(ax, dict):
                if ax.get("type", "space") == "space":
                    result.append(ax["name"])
            else:
                result.append(str(ax))
        return result or self.axes

    @cached_property
    def _level_info(self) -> list[tuple[str, dict[str, float], dict[str, float]]]:
        """For each scale level: (path, voxel_size, origin) dicts keyed by axis."""
        levels = []
        spatial = self._spatial_axes_order
        for ds in self._multiscale_attrs["datasets"]:
            level_path: str = ds["path"]
            voxel_size: dict[str, float] = {}
            origin: dict[str, float] = {}
            for tx in ds.get("coordinateTransformations", []):
                if tx["type"] == "scale":
                    scales = tx["scale"]
                    # Skip leading non-spatial dims (e.g. channel)
                    spatial_scales = scales[-len(spatial) :]
                    voxel_size = {c: float(s) for c, s in zip(spatial, spatial_scales)}
                elif tx["type"] == "translation":
                    trans = tx["translation"]
                    spatial_trans = trans[-len(spatial) :]
                    origin = {c: float(t) for c, t in zip(spatial, spatial_trans)}
            if not origin:
                origin = {c: 0.0 for c in spatial}
            levels.append((level_path, voxel_size, origin))
        return levels

    @cached_property
    def scale_level(self) -> int:
        """Index of the best-matching scale level for ``self.scale``."""
        best_idx = 0
        best_score = float("inf")
        for i, (_, vox_size, _) in enumerate(self._level_info):
            # Sum of relative differences across requested axes
            diffs = [
                abs(vox_size.get(ax, 0) - self.scale[ax]) / max(self.scale[ax], 1e-9)
                for ax in self.axes
                if ax in vox_size
            ]
            score = sum(diffs)
            # Prefer finer (smaller) voxel sizes when equally close
            if score < best_score:
                best_score = score
                best_idx = i
        return best_idx

    @cached_property
    def _selected_level(self) -> tuple[str, dict[str, float], dict[str, float]]:
        return self._level_info[self.scale_level]

    @cached_property
    def _zarr_array(self) -> zarr.Array:
        level_path, _, _ = self._selected_level
        return zarr.open_array(f"{self.path}/{level_path}", mode="r")

    @cached_property
    def _voxel_size(self) -> dict[str, float]:
        _, vox_size, _ = self._selected_level
        return vox_size

    @cached_property
    def _origin(self) -> dict[str, float]:
        _, _, origin = self._selected_level
        return origin

    # ------------------------------------------------------------------
    # Spatial properties
    # ------------------------------------------------------------------

    @cached_property
    def bounding_box(self) -> dict[str, tuple[float, float]]:
        """World bounding box ``{axis: (min_nm, max_nm)}`` of the selected level."""
        arr_shape = self._zarr_array.shape
        n_spatial = len(self.axes)
        spatial_shape = arr_shape[-n_spatial:]
        result: dict[str, tuple[float, float]] = {}
        for i, ax in enumerate(self.axes):
            start = self._origin.get(ax, 0.0)
            end = start + spatial_shape[i] * self._voxel_size.get(ax, 1.0)
            result[ax] = (start, end)
        return result

    @cached_property
    def sampling_box(self) -> dict[str, tuple[float, float]] | None:
        """Bounding box where patch centres can be drawn.

        When the array is large enough to contain the full output patch, the
        box is the bounding box shrunk by ``output_size / 2`` on each side so
        that reads never extend outside the array.

        When the array is *smaller* than the output patch:
        - If ``pad=True``: returns a single-centre box positioned at the
          midpoint of the bounding box.  ``len(dataset)`` will be 1 and
          out-of-bounds voxels are filled with ``pad_value``.
        - If ``pad=False``: returns ``None`` (dataset excluded).
        """
        bb = self.bounding_box
        result: dict[str, tuple[float, float]] = {}
        for ax in self.axes:
            half = self.output_size[ax] / 2.0
            lo = bb[ax][0] + half
            hi = bb[ax][1] - half
            if lo >= hi:
                if not self.pad:
                    return None
                # Crop smaller than output: expose a single sample at the centre
                bb_center = (bb[ax][0] + bb[ax][1]) / 2.0
                result[ax] = (
                    bb_center - self.scale[ax] / 2.0,
                    bb_center + self.scale[ax] / 2.0,
                )
            else:
                result[ax] = (lo, hi)
        return result

    def get_center(self, idx: int) -> dict[str, float]:
        """World coordinates of the centre voxel for flat index *idx*.

        *idx* indexes into the regular grid defined by the sampling box and
        ``self.scale`` (one point per output voxel).
        """
        sb = self.sampling_box
        if sb is None:
            raise ValueError(
                f"sampling_box is None for {self.path!r} "
                f"(array too small for requested patch size)"
            )
        grid = box_shape(sb, self.scale)
        axes = list(sb.keys())
        shape_tuple = tuple(grid[ax] for ax in axes)
        vox_idx = np.unravel_index(int(idx), shape_tuple)
        return {
            ax: sb[ax][0] + (vox_idx[i] + 0.5) * self.scale[ax]
            for i, ax in enumerate(axes)
        }

    # ------------------------------------------------------------------
    # Spatial-transform API (called by CellMapDataset before each read)
    # ------------------------------------------------------------------

    def set_spatial_transforms(self, transforms: dict | None) -> None:
        """Store the spatial transforms that will be applied in the next ``__getitem__``."""
        self._current_spatial_transforms = transforms

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def _compute_read_shape(self) -> list[int]:
        """Number of zarr voxels to read to cover ``output_shape * target_scale``
        nm of world space.

        When the selected zarr level has a different voxel size from the target
        scale, reading ``output_shape`` zarr voxels covers the wrong world
        extent.  The correct count is::

            zarr_voxels = round(output_voxels * target_scale / zarr_voxel_size)

        The rotation case further enlarges the read to accommodate the
        oversized pre-rotation patch.
        """
        base = [
            max(
                1,
                int(
                    round(
                        self.output_shape[ax]
                        * self.scale[ax]
                        / self._voxel_size.get(ax, self.scale[ax])
                    )
                ),
            )
            for ax in self.axes
        ]
        if self._current_spatial_transforms is None:
            return base
        R = self._current_spatial_transforms.get("rotation_matrix")
        if R is None:
            return base
        R = np.asarray(R, dtype=float)
        n = len(self.axes)
        return [
            int(np.ceil(base[i] * sum(abs(R[i, j]) for j in range(n))))
            for i in range(n)
        ]

    def __getitem__(self, center: Mapping[str, float]) -> torch.Tensor:
        """Return a tensor patch centred at *center* (world coords in nm)."""
        # Legacy: centre values may be arrays (take mean)
        center = {
            k: float(np.mean(v)) if not isinstance(v, (int, float)) else float(v)
            for k, v in center.items()
        }

        read_shape = self._compute_read_shape()
        arr = self._zarr_array
        spatial_ndim = len(self.axes)
        arr_shape = arr.shape[-spatial_ndim:]
        n_leading = arr.ndim - spatial_ndim

        slices: list[slice] = []
        pad_widths: list[tuple[int, int]] = []

        for i, ax in enumerate(self.axes):
            vs = self._voxel_size.get(ax, self.scale[ax])
            vox_center = (center[ax] - self._origin.get(ax, 0.0)) / vs
            start = int(np.floor(vox_center - read_shape[i] / 2.0))

            # Compute pad and slice such that pad_lo + valid_len + pad_hi == read_shape[i]
            # even when the read window is partially or fully outside the array.
            pad_lo = max(0, min(read_shape[i], -start))
            remaining = read_shape[i] - pad_lo
            arr_start = max(0, start)
            valid_len = max(0, min(arr_shape[i], arr_start + remaining) - arr_start)
            pad_hi = remaining - valid_len

            if valid_len == 0:
                logger.warning(
                    "Fully out-of-bounds read for %r axis %r: centre=%.1f is "
                    "outside array extent [%.1f, %.1f] nm.  This should not "
                    "happen in normal usage — check that the centre was drawn "
                    "from within sampling_box.",
                    self.path,
                    ax,
                    center[ax],
                    self._origin.get(ax, 0.0),
                    self._origin.get(ax, 0.0) + arr_shape[i] * vs,
                )

            slices.append(slice(arr_start, arr_start + valid_len))
            pad_widths.append((pad_lo, pad_hi))

        # Prepend slices for leading non-spatial dims (e.g. channel)
        full_slices: list[Any] = [slice(None)] * n_leading + slices
        leading_pads: list[tuple[int, int]] = [(0, 0)] * n_leading

        data = torch.from_numpy(np.asarray(arr[tuple(full_slices)], dtype=np.float32))

        # Pad if any region was out of bounds
        if any(p for pair in pad_widths for p in pair):
            if self.pad:
                flat_pad: list[int] = []
                for pw in reversed(leading_pads + pad_widths):
                    flat_pad += [pw[0], pw[1]]
                data = F.pad(data, flat_pad, mode="constant", value=self.pad_value)
            else:
                # Clamp: just use what we could read (shape will be smaller than
                # requested if near an edge; caller gets less data)
                pass

        # Resample if voxel size differs from target scale
        needs_resample = any(
            abs(self._voxel_size.get(ax, self.scale[ax]) - self.scale[ax])
            / max(self.scale[ax], 1e-9)
            > 0.01
            for ax in self.axes
        )
        if needs_resample:
            out_spatial = [self.output_shape[ax] for ax in self.axes]
            # Bring data to [N, C, *spatial] for interpolate
            orig_ndim = data.ndim
            while data.ndim < spatial_ndim + 2:
                data = data.unsqueeze(0)

            if self.interpolation == "nearest":
                mode = "nearest"
                extra: dict = {}
            else:
                mode = "trilinear" if spatial_ndim == 3 else "bilinear"
                extra = {"align_corners": False}

            data = F.interpolate(data, size=out_spatial, mode=mode, **extra)

            while data.ndim > orig_ndim:
                data = data.squeeze(0)

        # Apply rotation if requested
        R = (
            self._current_spatial_transforms.get("rotation_matrix")
            if self._current_spatial_transforms
            else None
        )
        if R is not None:
            data = self._apply_rotation(data, np.asarray(R, dtype=float))
            # Crop centre to target output shape
            target_shape = [self.output_shape[ax] for ax in self.axes]
            crop_slices: list[Any] = [slice(None)] * (data.ndim - spatial_ndim)
            for i in range(spatial_ndim):
                curr = data.shape[data.ndim - spatial_ndim + i]
                lo = (curr - target_shape[i]) // 2
                crop_slices.append(slice(lo, lo + target_shape[i]))
            data = data[tuple(crop_slices)]

        # Mirror
        if self._current_spatial_transforms is not None:
            mirror = self._current_spatial_transforms.get("mirror")
            if mirror is not None:
                for i, ax in enumerate(self.axes):
                    flag = (
                        mirror.get(ax, False) if isinstance(mirror, dict) else mirror[i]
                    )
                    if flag:
                        data = data.flip(data.ndim - spatial_ndim + i)

        # Transpose
        if self._current_spatial_transforms is not None:
            perm = self._current_spatial_transforms.get("transpose")
            if perm is not None:
                n_lead = data.ndim - spatial_ndim
                full_perm = list(range(n_lead)) + [n_lead + p for p in perm]
                data = data.permute(*full_perm).contiguous()

        # Enforce exact output_shape: centre-crop oversized dims, pad undersized
        # dims.  Undersized dims arise from near-boundary reads when pad=False;
        # padding them here makes the output shape predictable regardless of the
        # pad setting.
        target_shape = [self.output_shape[ax] for ax in self.axes]
        actual = [data.shape[data.ndim - spatial_ndim + i] for i in range(spatial_ndim)]
        if any(actual[i] != target_shape[i] for i in range(spatial_ndim)):
            # Centre-crop any oversized spatial dims
            crop_sl: list[Any] = [slice(None)] * (data.ndim - spatial_ndim)
            for i in range(spatial_ndim):
                curr, tgt = actual[i], target_shape[i]
                lo = max(0, (curr - tgt) // 2)
                crop_sl.append(slice(lo, lo + min(curr, tgt)))
            data = data[tuple(crop_sl)]

            # Pad any undersized spatial dims (symmetric, pad_value fill)
            actual = [
                data.shape[data.ndim - spatial_ndim + i] for i in range(spatial_ndim)
            ]
            pad_needed = [target_shape[i] - actual[i] for i in range(spatial_ndim)]
            if any(p > 0 for p in pad_needed):
                size_pad: list[int] = []
                for p in reversed(pad_needed):
                    size_pad += [p // 2, p - p // 2]
                size_pad += [0, 0] * (data.ndim - spatial_ndim)
                data = F.pad(data, size_pad, mode="constant", value=self.pad_value)

        if self.value_transform is not None:
            data = self.value_transform(data)

        return data

    def _apply_rotation(self, data: torch.Tensor, R: np.ndarray) -> torch.Tensor:
        """Apply a rotation matrix to the spatial dimensions of *data*.

        Uses ``torch.nn.functional.affine_grid`` + ``grid_sample`` so that
        gradients can flow through if needed.  For labels (``interpolation ==
        "nearest"``) nearest-neighbour resampling is used.
        """
        spatial_ndim = len(self.axes)
        original_ndim = data.ndim

        # Bring to [N, C, *spatial]
        while data.ndim < spatial_ndim + 2:
            data = data.unsqueeze(0)

        N = data.shape[0]

        # affine_grid expects the INVERSE transform (output → input mapping)
        R_inv = torch.tensor(R.T, dtype=torch.float32)

        if spatial_ndim == 3:
            theta = (
                torch.cat([R_inv, torch.zeros(3, 1)], dim=1)
                .unsqueeze(0)
                .expand(N, -1, -1)
            )
        else:
            theta = (
                torch.cat([R_inv[:2, :2], torch.zeros(2, 1)], dim=1)
                .unsqueeze(0)
                .expand(N, -1, -1)
            )

        grid = F.affine_grid(theta, list(data.shape), align_corners=False)
        mode = "nearest" if self.interpolation == "nearest" else "bilinear"
        rotated = F.grid_sample(
            data.float(), grid, mode=mode, padding_mode="zeros", align_corners=False
        )

        # Replace zero-padded corners with pad_value (only matters for continuous data)
        if self.pad_value != 0.0:
            # grid_sample fills OOB with 0; patch with pad_value
            oob_mask = (grid[..., 0].abs() > 1) | (grid[..., 1].abs() > 1)
            if spatial_ndim == 3:
                oob_mask = oob_mask | (grid[..., 2].abs() > 1)
            oob_mask = oob_mask.unsqueeze(1).expand_as(rotated)
            rotated[oob_mask] = self.pad_value

        while rotated.ndim > original_ndim:
            rotated = rotated.squeeze(0)

        return rotated

    # ------------------------------------------------------------------
    # Class counts (for weighted sampling)
    # ------------------------------------------------------------------

    @property
    def class_counts(self) -> dict[str, int]:
        """Foreground voxel count at s0, normalised to training-resolution voxels.

        Fast path reads pre-cached counts from ``s0/.zattrs``.  Slow path
        counts non-zero voxels in the s0 array and writes the result back.
        """
        # Fast path: check for cached counts in s0 attrs
        try:
            s0_path = self._level_info[0][0]
            s0_attrs = dict(zarr.open_group(self.path, mode="r")[s0_path].attrs)
            counts = s0_attrs.get("class_counts")
            if counts is not None and self.label_class in counts:
                raw_count = counts[self.label_class]
                return {self.label_class: self._scale_count(raw_count, s0_idx=0)}
        except Exception:
            pass

        # Slow path: count non-zero voxels in s0
        try:
            s0_path = self._level_info[0][0]
            s0_arr = zarr.open_array(f"{self.path}/{s0_path}", mode="r")
            fg_count = int(np.count_nonzero(s0_arr[:]))
            # Cache result in s0/.zattrs
            try:
                g = zarr.open_group(self.path, mode="r+")
                attrs = dict(g[s0_path].attrs)
                if "class_counts" not in attrs:
                    attrs["class_counts"] = {}
                attrs["class_counts"][self.label_class] = fg_count
                g[s0_path].attrs.update(attrs)
            except Exception:
                pass
            return {self.label_class: self._scale_count(fg_count, s0_idx=0)}
        except Exception as exc:
            logger.warning("class_counts failed for %s: %s", self.path, exc)
            return {self.label_class: 0}

    @property
    def total_voxels(self) -> int:
        """Total number of voxels in the data volume at training resolution.

        Computed as the product of the s0 array's spatial dimensions scaled
        to the training-resolution voxel size via :meth:`_scale_count`.
        """
        try:
            s0_path = self._level_info[0][0]
            s0_arr = zarr.open_array(f"{self.path}/{s0_path}", mode="r")
            n_spatial = len(self.axes)
            spatial_shape = s0_arr.shape[-n_spatial:]
            total_s0 = int(np.prod(spatial_shape))
            return self._scale_count(total_s0, s0_idx=0)
        except Exception as exc:
            logger.warning("total_voxels failed for %s: %s", self.path, exc)
            return 0

    def _scale_count(self, s0_count: int, s0_idx: int = 0) -> int:
        """Scale a voxel count from s0 resolution to training resolution."""
        try:
            _, s0_vox, _ = self._level_info[s0_idx]
            s0_vol = 1.0
            train_vol = 1.0
            for ax in self.axes:
                s0_vol *= s0_vox.get(ax, 1.0)
                train_vol *= self.scale.get(ax, 1.0)
            if train_vol == 0:
                return s0_count
            return int(s0_count * (s0_vol / train_vol))
        except Exception:
            return s0_count

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def to(self, device: str | torch.device) -> "CellMapImage":
        """No-op (tensors are always returned on CPU). Kept for API compatibility."""
        return self

    def __repr__(self) -> str:
        return (
            f"CellMapImage({self.path!r}, class={self.label_class!r}, "
            f"scale={list(self.scale.values())}, shape={list(self.output_shape.values())})"
        )
