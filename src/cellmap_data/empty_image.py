import torch
from typing import Any, Mapping, Optional, Sequence


class EmptyImage:
    """
    A class for handling empty image data.

    This class is used to create an empty image object, which can be used as a placeholder for images that do not exist in the dataset. It can be used to maintain a consistent API for image objects even when no data is present.

    Attributes:
        label_class (str): The intended label class of the image.
        target_scale (Sequence[float]): The intended scale of the image in physical space.
        target_voxel_shape (Sequence[int]): The intended shape of the image in voxels.
        store (Optional[torch.Tensor]): The tensor to return.
        axis_order (str): The intended order of the axes in the image.
        empty_value (float | int): The value to fill the image with.

    Methods:
        __getitem__(center: Mapping[str, float]) -> torch.Tensor: Returns the empty image data.
        to(device: str): Moves the image data to the given device.
        set_spatial_transforms(transforms: Mapping[str, Any] | None):
        Imitates the method in CellMapImage, but does nothing for an EmptyImage object.

    Properties:
        bounding_box (None): Returns None.
        sampling_box (None): Returns None.
        bg_count (float): Returns zero.
        class_counts (float): Returns zero.
    """

    def __init__(
        self,
        target_class: str,
        target_scale: Sequence[float],
        target_voxel_shape: Sequence[int],
        store: Optional[torch.Tensor] = None,
        axis_order: str = "zyx",
        empty_value: float | int = -100,
    ):
        self.label_class = target_class
        self.target_scale = target_scale
        if len(target_voxel_shape) < len(axis_order):
            axis_order = axis_order[-len(target_voxel_shape) :]
        self.output_shape = {c: target_voxel_shape[i] for i, c in enumerate(axis_order)}
        self.output_size = {
            c: t * s for c, t, s in zip(axis_order, target_voxel_shape, target_scale)
        }
        self.axes = axis_order
        self._bounding_box = None
        self._class_counts = 0.0
        self._bg_count = 0.0
        self.scale = {c: sc for c, sc in zip(self.axes, self.target_scale)}
        self.empty_value = empty_value
        if store is not None:
            self.store = store
        else:
            self.store = (
                torch.ones([self.output_shape[c] for c in self.axes]) * self.empty_value
            )

    def __getitem__(self, center: Mapping[str, float]) -> torch.Tensor:
        return self.store

    @property
    def bounding_box(self) -> None:
        return self._bounding_box

    @property
    def sampling_box(self) -> None:
        return self._bounding_box

    @property
    def bg_count(self) -> float:
        return self._bg_count

    @property
    def class_counts(self) -> float:
        return self._class_counts

    def to(self, device: str, non_blocking: bool = True) -> None:
        self.store = self.store.to(device, non_blocking=non_blocking)

    def set_spatial_transforms(self, transforms: Mapping[str, Any] | None) -> None:
        pass
