"""EmptyImage: a NaN-filled placeholder for unannotated classes."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence

import torch


class EmptyImage:
    """Returns a NaN tensor for every read.

    Used when a class is not annotated in a given dataset.  NaN signals
    *unknown* to the model, as opposed to *absent* (zeros).

    The constructor signature mirrors :class:`CellMapImage` so the two can
    be used interchangeably inside :class:`CellMapDataset`.
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
        if len(axis_order) > len(target_scale):
            target_scale = [target_scale[0]] * (
                len(axis_order) - len(target_scale)
            ) + list(target_scale)
        if len(axis_order) > len(target_voxel_shape):
            ndim_fix = len(axis_order) - len(target_voxel_shape)
            target_voxel_shape = [1] * ndim_fix + list(target_voxel_shape)
        self.axes: list[str] = axis_order[: len(target_voxel_shape)]
        self.scale = {c: float(s) for c, s in zip(axis_order, target_scale)}
        self.output_shape = {c: int(t) for c, t in zip(axis_order, target_voxel_shape)}
        self._nan_tensor = torch.full(
            [self.output_shape[ax] for ax in self.axes], float("nan")
        )

    def __getitem__(self, center: Mapping[str, float]) -> torch.Tensor:
        return self._nan_tensor.clone()

    def set_spatial_transforms(self, transforms: dict | None) -> None:
        pass

    @property
    def bounding_box(self) -> None:
        return None

    @property
    def sampling_box(self) -> None:
        return None

    @property
    def class_counts(self) -> dict[str, int]:
        return {self.label_class: 0}

    @property
    def total_voxels(self) -> int:
        return 0

    def to(self, device: str | torch.device) -> "EmptyImage":
        self._nan_tensor = self._nan_tensor.to(device)
        return self

    def __repr__(self) -> str:
        return (
            f"EmptyImage(class={self.label_class!r}, "
            f"shape={list(self.output_shape.values())})"
        )
