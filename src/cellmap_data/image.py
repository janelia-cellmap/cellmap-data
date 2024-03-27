from typing import Iterable
import torch
import tensorstore as ts
from fibsem_tools.io.core import read_xarray


class CellMapImage:
    path: str
    translation: tuple[float, ...]
    shape: tuple[float, ...]
    scale: tuple[float, ...]
    label_class: str
    class_count: int
    store: ts.TensorStore

    def __init__(
        self,
        path: str,
        target_class: str,
        target_scale: Iterable[float],
        target_voxel_shape: Iterable[int],
    ):
        self.path = path
        self.label_class = target_class
        self.scale = tuple(target_scale)
        self.output_shape = tuple(target_voxel_shape)
        self._bounding_box = None
        self._class_counts = None
        self.construct()

    def construct(self): ...

    def __getitem__(self, center: Iterable[float]):
        """Returns image data centered around the given point, based on the scale and shape of the target output image."""
        ...

    @property
    def bounding_box(self) -> dict[str, list[float]]:
        """Returns the bounding box of the dataset."""
        if self._bounding_box is None:
            self._bounding_box = {
                c: [self.translation[i], self.shape[i] + self.translation[i]]
                for i, c in enumerate("xyz")
            }
        return self._bounding_box

    @property
    def class_counts(self) -> int:
        """Returns the number of pixels for the contained class in the ground truth data, normalized by the resolution."""
        if self._class_counts is None:
            # Get from cellmap-schemas metadata, then normalize by resolution
            # TODO
            ...
        return self._class_counts


class EmptyImage:
    shape: tuple[float, ...]
    label_class: str
    class_count: int
    store: torch.Tensor

    def __init__(
        self,
        target_class: str,
        target_voxel_shape: Iterable[int],
    ):
        self.label_class = target_class
        self.output_shape = tuple(target_voxel_shape)
        self._bounding_box = None
        self._class_counts = 0
        self.store = torch.zeros(self.output_shape)

    def __getitem__(self, center: Iterable[float]):
        """Returns image data centered around the given point, based on the scale and shape of the target output image."""
        return self.store

    @property
    def bounding_box(self) -> None:
        """Returns the bounding box of the dataset."""
        return self._bounding_box

    @property
    def class_counts(self) -> int:
        """Returns the number of pixels for the contained class in the ground truth data, normalized by the resolution."""
        return self._class_counts
