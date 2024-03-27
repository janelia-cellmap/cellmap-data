from typing import Iterable, Optional
import torch
import tensorstore as ts
from fibsem_tools.io.core import read_xarray


class CellMapImage:
    path: str
    translation: tuple[float, ...]
    shape: tuple[float, ...]
    scale: tuple[float, ...]
    label_class: str
    store: ts.TensorStore

    def __init__(
        self,
        path: str,
        target_class: str,
        target_scale: Iterable[float],
        target_voxel_shape: Iterable[int],
    ):
        """Initializes a CellMapImage object.

        Args:
            path (str): The path to the image file.
            target_class (str): The label class of the image.
            target_scale (Iterable[float]): The scale of the image in physical space.
            target_voxel_shape (Iterable[int]): The shape of the image in voxels.
        """

        self.path = path
        self.label_class = target_class
        self.scale = tuple(
            target_scale
        )  # TODO: this should be a dictionary of scales for each axis
        self.output_shape = tuple(
            target_voxel_shape
        )  # TODO: this should be a dictionary of shapes for each axis
        self.construct()

    def construct(self):
        self._bounding_box = None
        self._sampling_box = None
        self._class_counts = None

    def __getitem__(self, center: Iterable[float]) -> torch.Tensor:
        """Returns image data centered around the given point, based on the scale and shape of the target output image."""
        # Get the image data using Tensorstore
        # Ensure that the data is the right scale and shape
        # TODO
        ...

    @property
    def bounding_box(self) -> dict[str, list[float]]:
        """Returns the bounding box of the dataset."""
        if self._bounding_box is None:
            self._bounding_box = {
                c: [self.translation[i], self.shape[i] + self.translation[i]]
                for i, c in enumerate("zyx")
            }
        return self._bounding_box

    @property
    def sampling_box(self) -> dict[str, list[float]]:
        """Returns the sampling box of the dataset (i.e. where centers can be drawn from and still have full samples drawn from within the bounding box)."""
        if self._sampling_box is None:
            self._sampling_box = {}
            output_padding = {c: (s / 2) for c, s in zip("zyx", self.output_shape)}
            for c, (start, stop) in self.bounding_box.items():
                self._sampling_box[c] = [
                    start + output_padding[c],
                    stop - output_padding[c],
                ]
        return self._sampling_box

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
    store: torch.Tensor

    def __init__(
        self,
        target_class: str,
        target_voxel_shape: Iterable[int],
        store: Optional[torch.Tensor] = None,
    ):
        """Initializes an empty image object.

        Args:
            target_class (str): The label class of the image.
            target_voxel_shape (Iterable[int]): The shape of the image in voxels.
            store (Optional[torch.Tensor], optional): The tensor to return. Defaults to None.
        """
        self.label_class = target_class
        self.output_shape = tuple(target_voxel_shape)
        self._bounding_box = None
        self._class_counts = 0
        if store is not None:
            self.store = store
        else:
            self.store = torch.zeros(self.output_shape)

    def __getitem__(self, center: Iterable[float]) -> torch.Tensor:
        """Returns image data centered around the given point, based on the scale and shape of the target output image."""
        return self.store

    @property
    def bounding_box(self) -> None:
        """Returns the bounding box of the dataset."""
        return self._bounding_box

    @property
    def sampling_box(self) -> None:
        """Returns the sampling box of the dataset (i.e. where centers can be drawn from and still have full samples drawn from within the bounding box)."""
        return self._bounding_box

    @property
    def class_counts(self) -> int:
        """Returns the number of pixels for the contained class in the ground truth data, normalized by the resolution."""
        return self._class_counts
