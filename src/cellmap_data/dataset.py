"""CellMapDataset: PyTorch Dataset for CellMap OME-NGFF data."""

from __future__ import annotations

import logging
import os
from functools import cached_property
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .empty_image import EmptyImage
from .image import CellMapImage
from .utils import split_target_path
from .utils.geometry import box_intersection, box_shape

logger = logging.getLogger(__name__)


def _make_rotation_matrix(axes: list[str], rotation_config: dict) -> np.ndarray | None:
    """Build a rotation matrix from a per-axis angle dict (degrees).

    Returns a (n, n) orthonormal rotation matrix, or ``None`` if all angles
    are zero.
    """
    n = len(axes)
    R = np.eye(n)
    for ax, angle_deg in rotation_config.items():
        if angle_deg == 0 or ax not in axes:
            continue
        theta = np.deg2rad(angle_deg)
        # Determine the two axes to rotate in (perpendicular to *ax*)
        ax_idx = axes.index(ax)
        other = [i for i in range(n) if i != ax_idx]
        if len(other) < 2:
            continue
        i, j = other[0], other[1]
        Ri = np.eye(n)
        Ri[i, i] = np.cos(theta)
        Ri[i, j] = -np.sin(theta)
        Ri[j, i] = np.sin(theta)
        Ri[j, j] = np.cos(theta)
        R = R @ Ri
    return None if np.allclose(R, np.eye(n)) else R


class CellMapDataset(Dataset):
    """PyTorch Dataset that reads patches from a single CellMap zarr dataset.

    Parameters
    ----------
    raw_path:
        Path to the raw EM zarr group.
    target_path:
        Path template for GT labels, with classes in brackets, e.g.
        ``"/data/jrc.zarr/labels/[mito,er]"``.  Each class occupies a
        sub-group of the base path (``base/mito``, ``base/er``, …).
    classes:
        Segmentation classes to load.
    input_arrays:
        ``{array_name: {"shape": (z,y,x), "scale": (z,y,x)}}`` specs for
        input patches.
    target_arrays:
        ``{array_name: {"shape": (z,y,x), "scale": (z,y,x)}}`` specs for
        target patches.  All classes share these specs.
    pad:
        Whether to pad reads that extend beyond array bounds with
        ``pad_value`` (NaN by default).
    spatial_transforms:
        Augmentation config dict with optional keys ``"mirror"``,
        ``"transpose"``, and ``"rotate"``.  Example::

            {
                "mirror": {"z": True, "y": True, "x": True},
                "transpose": True,
                "rotate": {"z": 45},   # max degrees
            }
    raw_value_transforms:
        Callable applied to each raw input tensor.
    target_value_transforms:
        Callable (or ``{class: callable}`` dict) applied to each target
        tensor.
    class_relation_dict:
        Stored for API compatibility; not used in inference currently.
    force_has_data:
        Skip the empty-data check when ``True``.
    device:
        Ignored — all tensors are returned on CPU.
    """

    def __init__(
        self,
        raw_path: str,
        target_path: str,
        classes: Sequence[str],
        input_arrays: Mapping[str, Mapping[str, Any]],
        target_arrays: Mapping[str, Mapping[str, Any]],
        pad: bool = False,
        spatial_transforms: Optional[Mapping[str, Any]] = None,
        raw_value_transforms: Optional[Callable] = None,
        target_value_transforms: Optional[Callable | Mapping[str, Callable]] = None,
        class_relation_dict: Optional[Mapping[str, Sequence[str]]] = None,
        force_has_data: bool = False,
        device: Optional[str | torch.device] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.raw_path = raw_path
        self.target_path = target_path
        self.classes = list(classes)
        self.input_arrays = dict(input_arrays)
        self.target_arrays = dict(target_arrays)
        self.pad = pad
        self.spatial_transforms_config = spatial_transforms
        self.raw_value_transforms = raw_value_transforms
        self.target_value_transforms = target_value_transforms
        self.class_relation_dict = class_relation_dict
        self.force_has_data = force_has_data
        self._rng = np.random.default_rng(seed)

        # Parse target path to get template and annotated classes
        gt_path_template, annotated_classes = split_target_path(target_path)
        self._gt_path_template = gt_path_template
        self._annotated_classes = set(annotated_classes)

        # Build input sources
        self.input_sources: dict[str, CellMapImage] = {}
        for arr_name, arr_spec in self.input_arrays.items():
            self.input_sources[arr_name] = CellMapImage(
                path=raw_path,
                target_class=arr_name,
                target_scale=arr_spec["scale"],
                target_voxel_shape=arr_spec["shape"],
                pad=pad,
                value_transform=raw_value_transforms,
            )

        # Build target sources: one CellMapImage or EmptyImage per class
        # Use the first (and typically only) target array spec
        first_target_spec = next(iter(target_arrays.values()))
        self.target_sources: dict[str, CellMapImage | EmptyImage] = {}
        for cls in self.classes:
            cls_path = gt_path_template.format(label=cls)
            value_tx = self._class_value_transform(cls)
            if cls in self._annotated_classes and os.path.exists(cls_path):
                self.target_sources[cls] = CellMapImage(
                    path=cls_path,
                    target_class=cls,
                    target_scale=first_target_spec["scale"],
                    target_voxel_shape=first_target_spec["shape"],
                    pad=pad,
                    interpolation="nearest",
                    value_transform=value_tx,
                )
            else:
                self.target_sources[cls] = EmptyImage(
                    path=cls_path,
                    target_class=cls,
                    target_scale=first_target_spec["scale"],
                    target_voxel_shape=first_target_spec["shape"],
                )

    def _class_value_transform(self, cls: str) -> Optional[Callable]:
        """Return the value transform for a specific class."""
        if self.target_value_transforms is None:
            return None
        if callable(self.target_value_transforms):
            return self.target_value_transforms
        if isinstance(self.target_value_transforms, Mapping):
            return self.target_value_transforms.get(cls)
        return None

    # ------------------------------------------------------------------
    # Spatial properties
    # ------------------------------------------------------------------

    @cached_property
    def bounding_box(self) -> dict[str, tuple[float, float]] | None:
        """Intersection of all source bounding boxes."""
        box = None
        for src in list(self.input_sources.values()) + list(
            self.target_sources.values()
        ):
            bb = src.bounding_box
            if bb is None:
                continue
            box = bb if box is None else box_intersection(box, bb)
            if box is None:
                return None
        return box

    @cached_property
    def sampling_box(self) -> dict[str, tuple[float, float]] | None:
        """Intersection of all source sampling boxes.

        ``EmptyImage`` sources (``bounding_box is None``) are skipped.
        A ``CellMapImage`` with a crop smaller than the output patch returns
        a single-centre ``sampling_box`` when ``pad=True`` (so
        ``len(dataset)`` becomes 1), or ``None`` when ``pad=False`` (which
        causes this method to return ``None`` and exclude the dataset).
        """
        box = None
        for src in list(self.input_sources.values()) + list(
            self.target_sources.values()
        ):
            sb = src.sampling_box
            if sb is None:
                if src.bounding_box is None:
                    continue  # EmptyImage — no spatial constraint
                return None  # pad=False and crop too small → exclude
            box = sb if box is None else box_intersection(box, sb)
            if box is None:
                return None
        return box

    @cached_property
    def _target_scale(self) -> dict[str, float]:
        """Scale of the first target array spec."""
        first_target_src = next(iter(self.target_sources.values()))
        return dict(first_target_src.scale)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        sb = self.sampling_box
        if sb is None:
            return 0
        grid = box_shape(sb, self._target_scale)
        total = 1
        for v in grid.values():
            total *= v
        return total

    def __getitem__(self, idx: int) -> dict[str, Any]:
        center = self._idx_to_center(idx)
        transforms = self._generate_spatial_transforms()

        # Set transforms on all sources
        for src in list(self.input_sources.values()) + list(
            self.target_sources.values()
        ):
            src.set_spatial_transforms(transforms)

        result: dict[str, Any] = {"idx": torch.tensor(idx)}

        for name, src in self.input_sources.items():
            tensor = src[center]
            # Drop any singleton spatial dims (e.g. Z=1 for flat-3D inputs),
            # then prepend C=1 so the batch has shape [N, C, *spatial] as
            # expected by PyTorch convolutions.
            if 1 in tensor.shape:
                tensor = tensor.squeeze()
            result[name] = tensor.unsqueeze(0)  # [C=1, *spatial]

        # Stack per-class tensors under each target array name.
        # The challenge (and train.py) accesses targets via target_arrays keys
        # (e.g. batch["output"]), not individual class names.
        if self.target_arrays:
            class_tensors = []
            for cls in self.classes:
                t = self.target_sources[cls][center]
                # Match the same singleton-dim squeeze applied to inputs so
                # spatial dims are consistent between inputs and targets.
                if 1 in t.shape:
                    t = t.squeeze()
                class_tensors.append(t)
            stacked = torch.stack(class_tensors, dim=0)  # [n_classes, *spatial]
            for arr_name in self.target_arrays:
                result[arr_name] = stacked

        # Reset spatial transforms
        for src in list(self.input_sources.values()) + list(
            self.target_sources.values()
        ):
            src.set_spatial_transforms(None)

        return result

    def _idx_to_center(self, idx: int) -> dict[str, float]:
        """Convert flat index to world centre coordinates."""
        sb = self.sampling_box
        if sb is None:
            raise IndexError(f"sampling_box is None for {self.raw_path!r}")
        scale = self._target_scale
        grid = box_shape(sb, scale)
        axes = list(sb.keys())
        shape_tuple = tuple(grid[ax] for ax in axes)
        vox_idx = np.unravel_index(int(idx) % max(1, len(self)), shape_tuple)
        return {
            ax: sb[ax][0] + (vox_idx[i] + 0.5) * scale[ax] for i, ax in enumerate(axes)
        }

    def _generate_spatial_transforms(self) -> dict | None:
        """Generate random spatial transforms from the config for one sample."""
        cfg = self.spatial_transforms_config
        if not cfg:
            return None

        result: dict[str, Any] = {}

        # Mirror
        mirror_cfg = cfg.get("mirror")
        if mirror_cfg:
            if isinstance(mirror_cfg, dict):
                # Support {"axes": {"x": 0.5, ...}} wrapper or flat {"x": 0.5, ...}
                axis_probs = mirror_cfg.get("axes", mirror_cfg)
                if isinstance(axis_probs, dict):
                    result["mirror"] = {
                        ax: bool(self._rng.random() < prob)
                        for ax, prob in axis_probs.items()
                    }
                else:
                    axes = next(iter(self.input_sources.values())).axes
                    result["mirror"] = {
                        ax: bool(self._rng.random() < 0.5) for ax in axes
                    }
            else:
                axes = next(iter(self.input_sources.values())).axes
                result["mirror"] = {ax: bool(self._rng.random() < 0.5) for ax in axes}

        # Transpose
        if cfg.get("transpose"):
            axes = next(iter(self.input_sources.values())).axes
            n = len(axes)
            perm = list(self._rng.permutation(n))
            result["transpose"] = perm

        # Rotate
        rotate_cfg = cfg.get("rotate")
        if rotate_cfg:
            axes = next(iter(self.input_sources.values())).axes
            if isinstance(rotate_cfg, dict):
                # Support {"axes": {"x": 45, ...}} wrapper or flat {"x": 45, ...}
                axis_angles = rotate_cfg.get("axes", rotate_cfg)
                if isinstance(axis_angles, dict):
                    angle_dict: dict[str, float] = {}
                    for ax, max_angle in axis_angles.items():
                        if isinstance(max_angle, (list, tuple)):
                            lo, hi = max_angle
                        else:
                            lo, hi = -float(max_angle), float(max_angle)
                        angle_dict[ax] = float(self._rng.uniform(lo, hi))
                    R = _make_rotation_matrix(axes, angle_dict)
                    if R is not None:
                        result["rotation_matrix"] = R

        return result if result else None

    # ------------------------------------------------------------------
    # Sampling utilities
    # ------------------------------------------------------------------

    def get_indices(self, chunk_size: Mapping[str, float]) -> list[int]:
        """Flat indices that tile the sampling box without overlap.

        Parameters
        ----------
        chunk_size:
            World-space tile size per axis in nm (e.g. target output size).
        """
        sb = self.sampling_box
        if sb is None:
            return []
        scale = self._target_scale
        grid = box_shape(sb, scale)
        axes = list(sb.keys())

        # Tile with the given chunk size
        chunk_grid = {
            ax: max(
                1, int(round((sb[ax][1] - sb[ax][0]) / chunk_size.get(ax, scale[ax])))
            )
            for ax in axes
        }
        indices = []
        shape_tuple = tuple(grid[ax] for ax in axes)
        chunk_tuple = tuple(chunk_grid[ax] for ax in axes)
        # Step through the sampling box in chunks
        for chunk_idx in np.ndindex(*chunk_tuple):
            vox_idx = tuple(
                int(chunk_idx[i] * shape_tuple[i] / chunk_tuple[i])
                for i in range(len(axes))
            )
            flat = int(np.ravel_multi_index(vox_idx, shape_tuple))
            indices.append(flat)
        return indices

    def get_crop_class_matrix(self) -> np.ndarray:
        """Return a ``[1, n_classes]`` boolean row for ClassBalancedSampler.

        True where the class is annotated (non-empty CellMapImage).
        """
        row = np.array(
            [
                not isinstance(self.target_sources.get(cls), EmptyImage)
                for cls in self.classes
            ],
            dtype=bool,
        ).reshape(1, -1)
        return row

    # ------------------------------------------------------------------
    # Class counts
    # ------------------------------------------------------------------

    @property
    def class_counts(self) -> dict[str, Any]:
        """Aggregate per-class foreground voxel counts from all target sources."""
        totals: dict[str, int] = {}
        for cls in self.classes:
            src = self.target_sources.get(cls)
            if src is not None:
                counts = src.class_counts
                totals[cls] = counts.get(cls, 0)
            else:
                totals[cls] = 0
        return {"totals": totals}

    @property
    def total_voxels(self) -> dict[str, int]:
        """Total voxels in the data volume per class, normalised to training-resolution voxels."""
        totals: dict[str, int] = {}
        for cls in self.classes:
            src = self.target_sources.get(cls)
            totals[cls] = src.total_voxels if src is not None else 0
        return totals

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def verify(self) -> bool:
        """Return True if the dataset has at least one valid sample."""
        return len(self) > 0 or self.force_has_data

    def set_raw_value_transforms(self, transforms: Optional[Callable]) -> None:
        self.raw_value_transforms = transforms
        for src in self.input_sources.values():
            src.value_transform = transforms
        # Reset cached properties that depend on sources
        self.__dict__.pop("bounding_box", None)
        self.__dict__.pop("sampling_box", None)

    def set_target_value_transforms(
        self, transforms: Optional[Callable | Mapping[str, Callable]]
    ) -> None:
        self.target_value_transforms = transforms
        for cls, src in self.target_sources.items():
            if isinstance(src, CellMapImage):
                src.value_transform = self._class_value_transform(cls)

    def set_spatial_transforms(self, transforms: Optional[Mapping[str, Any]]) -> None:
        self.spatial_transforms_config = transforms

    def to(self, device: str | torch.device) -> "CellMapDataset":
        """No-op for API compatibility (tensors returned on CPU)."""
        return self

    def __repr__(self) -> str:
        return (
            f"CellMapDataset(raw={self.raw_path!r}, "
            f"classes={self.classes}, len={len(self)})"
        )
