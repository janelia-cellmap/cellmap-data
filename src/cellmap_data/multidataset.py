"""CellMapMultiDataset: combines multiple CellMapDataset instances."""

from __future__ import annotations

import logging
from functools import cached_property
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import ConcatDataset
from tqdm import tqdm

from .dataset import CellMapDataset

logger = logging.getLogger(__name__)


class CellMapMultiDataset(ConcatDataset):
    """Concatenates multiple :class:`CellMapDataset` instances.

    Provides aggregate ``class_counts``, ``get_crop_class_matrix``, and
    ``validation_indices`` over all constituent datasets, which are required
    by :class:`~cellmap_data.sampler.ClassBalancedSampler` and
    :class:`~cellmap_data.dataloader.CellMapDataLoader`.

    Parameters
    ----------
    datasets:
        List of :class:`CellMapDataset` objects to concatenate.
    classes:
        Shared segmentation classes (must match each dataset's ``classes``).
    input_arrays:
        Shared input array specs.
    target_arrays:
        Shared target array specs.
    """

    def __init__(
        self,
        datasets: Sequence[CellMapDataset],
        classes: Sequence[str],
        input_arrays: Mapping[str, Mapping[str, Any]],
        target_arrays: Mapping[str, Mapping[str, Any]],
    ) -> None:
        super().__init__(datasets)  # initialises ConcatDataset
        self.classes = list(classes)
        self.input_arrays = dict(input_arrays)
        self.target_arrays = dict(target_arrays)

    # ------------------------------------------------------------------
    # Class weights / sampling
    # ------------------------------------------------------------------

    @property
    def class_counts(self) -> dict[str, Any]:
        """Aggregate foreground voxel counts across all datasets.

        Sequential scan (parallelism offers no benefit over NFS; see
        project MEMORY.md notes on ``CellMapMultiDataset.class_counts``).

        Returns a dict with:
        - ``"totals"``: per-class foreground voxel counts.
        - ``"totals_total"``: per-class total voxel counts (full array sizes).
        """
        totals: dict[str, int] = {cls: 0 for cls in self.classes}
        totals_total: dict[str, int] = {cls: 0 for cls in self.classes}
        for ds in tqdm(self.datasets, desc="Counting class voxels", leave=False):
            ds_counts = ds.class_counts
            for cls in self.classes:
                totals[cls] += ds_counts.get("totals", {}).get(cls, 0)
                totals_total[cls] += ds_counts.get("totals_total", {}).get(cls, 0)
        return {"totals": totals, "totals_total": totals_total}

    @property
    def class_weights(self) -> dict[str, float]:
        """Per-class sampling weight: ``bg_voxels / fg_voxels``.

        Background voxels for each class are derived from the actual data
        volume size (``totals_total``) minus foreground voxels, so the ratio
        correctly reflects the class imbalance within each volume.
        """
        counts = self.class_counts
        fg_counts = counts["totals"]
        total_counts = counts["totals_total"]
        weights: dict[str, float] = {}
        for cls in self.classes:
            fg = fg_counts.get(cls, 0)
            total = total_counts.get(cls, 0)
            bg = max(total - fg, 0)
            weights[cls] = float(bg) / float(max(fg, 1))
        return weights

    def get_crop_class_matrix(self) -> np.ndarray:
        """Stack ``[n_crops, n_classes]`` bool matrix from all datasets."""
        return np.vstack([ds.get_crop_class_matrix() for ds in self.datasets])

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @cached_property
    def validation_indices(self) -> list[int]:
        """Non-overlapping tile indices across all datasets (for validation)."""
        indices: list[int] = []
        offset = 0
        for ds in self.datasets:
            # Use the output size of the first target array as the tile size
            first_target_spec = next(iter(ds.target_arrays.values()))
            scale = {
                c: float(s)
                for c, s in zip(
                    next(iter(ds.target_sources.values())).axes,
                    first_target_spec["scale"],
                )
            }
            shape = {
                c: int(t)
                for c, t in zip(
                    next(iter(ds.target_sources.values())).axes,
                    first_target_spec["shape"],
                )
            }
            chunk_size = {ax: scale[ax] * shape[ax] for ax in scale}
            local_indices = ds.get_indices(chunk_size)
            indices.extend(i + offset for i in local_indices)
            offset += len(ds)
        return indices

    def verify(self) -> bool:
        return len(self) > 0

    # ------------------------------------------------------------------
    # Transform setters (delegate to all datasets)
    # ------------------------------------------------------------------

    def set_raw_value_transforms(self, transforms: Optional[Callable]) -> None:
        for ds in self.datasets:
            ds.set_raw_value_transforms(transforms)
        self.__dict__.pop("validation_indices", None)

    def set_target_value_transforms(
        self, transforms: Optional[Callable | Mapping[str, Callable]]
    ) -> None:
        for ds in self.datasets:
            ds.set_target_value_transforms(transforms)

    def set_spatial_transforms(self, transforms: Optional[Mapping[str, Any]]) -> None:
        for ds in self.datasets:
            ds.set_spatial_transforms(transforms)

    def to(self, device: str | torch.device) -> "CellMapMultiDataset":
        for ds in self.datasets:
            ds.to(device)
        return self

    def __repr__(self) -> str:
        return (
            f"CellMapMultiDataset({len(self.datasets)} datasets, "
            f"classes={self.classes}, len={len(self)})"
        )
