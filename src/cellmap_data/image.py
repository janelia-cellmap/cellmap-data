import os
from typing import Any, Callable, Iterable, Optional, Sequence
import torch
from fibsem_tools.io.core import read_xarray
import xarray
import tensorstore
import xarray_tensorstore as xt
import numpy as np
from cellmap_schemas.annotation import AnnotationGroup
import zarr


class CellMapImage:
    path: str
    _translation: dict[str, float]
    scale: dict[str, float]
    output_shape: dict[str, int]
    output_size: dict[str, float]
    label_class: str
    _array: xarray.DataArray
    axes: str | Sequence[str]
    post_image_transforms: Sequence[str] = ["transpose"]
    value_transform: Optional[Callable]
    context: Optional[tensorstore.Context] = None  # type: ignore

    def __init__(
        self,
        path: str,
        target_class: str,
        target_scale: Sequence[float],
        target_voxel_shape: Sequence[int],
        axis_order: str | Sequence[str] = "zyx",
        value_transform: Optional[Callable] = None,
        # TODO: Add global grid enforcement to ensure that all images are on the same grid
        context: Optional[tensorstore.Context] = None,  # type: ignore
    ):
        """Initializes a CellMapImage object.

        Args:
            path (str): The path to the image file.
            target_class (str): The label class of the image.
            target_scale (Sequence[float]): The scale of the image in physical space.
            target_voxel_shape (Sequence[int]): The shape of the image in voxels.
            axis_order (str, optional): The order of the axes in the image. Defaults to "zyx".
            value_transform (Optional[callable], optional): A function to transform the image data. Defaults to None.
            context (Optional[tensorstore.Context], optional): The context for the image data. Defaults to None.
        """

        self.path = path
        self.label_class = target_class
        # TODO: Below makes assumptions about image scale, and also locks which axis is sliced to 2D
        if len(axis_order) > len(target_scale):
            target_scale = [target_scale[0]] * (
                len(axis_order) - len(target_scale)
            ) + list(target_scale)
        if len(axis_order) > len(target_voxel_shape):
            target_voxel_shape = [1] * (
                len(axis_order) - len(target_voxel_shape)
            ) + list(target_voxel_shape)

        self.scale = {c: s for c, s in zip(axis_order, target_scale)}
        self.output_shape = {c: t for c, t in zip(axis_order, target_voxel_shape)}
        self.output_size = {
            c: t * s for c, t, s in zip(axis_order, target_voxel_shape, target_scale)
        }
        self.axes = axis_order[: len(target_voxel_shape)]
        self.value_transform = value_transform
        self.context = context
        self._bounding_box = None
        self._sampling_box = None
        self._class_counts = None
        self._current_spatial_transforms = None
        self._last_coords = None
        self._original_scale = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.xs = self.array.coords["x"]
        # self.ys = self.array.coords["y"]
        # self.zs = self.array.coords["z"]

    def __getitem__(self, center: dict[str, float]) -> torch.Tensor:
        """Returns image data centered around the given point, based on the scale and shape of the target output image."""
        # Find vectors of coordinates in world space to pull data from
        coords = {}
        for c in self.axes:
            if center[c] - self.output_size[c] / 2 < self.bounding_box[c][0]:
                # raise ValueError(
                UserWarning(
                    f"Center {center[c]} is out of bounds for axis {c} in image {self.path}. {center[c] - self.output_size[c] / 2} would be less than {self.bounding_box[c][0]}"
                )
                # center[c] = self.bounding_box[c][0] + self.output_size[c] / 2
            if center[c] + self.output_size[c] / 2 > self.bounding_box[c][1]:
                # raise ValueError(
                UserWarning(
                    f"Center {center[c]} is out of bounds for axis {c} in image {self.path}. {center[c] + self.output_size[c] / 2} would be greater than {self.bounding_box[c][1]}"
                )
                # center[c] = self.bounding_box[c][1] - self.output_size[c] / 2
            coords[c] = np.linspace(
                center[c] - self.output_size[c] / 2,
                center[c] + self.output_size[c] / 2,
                self.output_shape[c],
            )

        # Apply any spatial transformations to the coordinates and return the image data as a PyTorch tensor
        data = self.apply_spatial_transforms(coords)

        # Apply any value transformations to the data
        if self.value_transform is not None:
            data = self.value_transform(data)
        return data.to(self.device)

    def __repr__(self) -> str:
        # TODO: array_path instead of path
        return f"CellMapImage({self.path})"

    @property
    def array(self) -> xarray.DataArray:
        if not hasattr(self, "_array"):
            self.group = read_xarray(self.path)
            # Find correct multiscale level based on target scale
            self.scale_level = self.find_level(self.scale)
            self.array_path = os.path.join(self.path, self.scale_level)
            # Construct an xarray with Tensorstore backend
            ds = read_xarray(self.array_path)
            spec = xt._zarr_spec_from_path(self.array_path)
            array_future = tensorstore.open(  # type: ignore
                spec, read=True, write=False, context=self.context
            )
            array = array_future.result()
            new_data = xt._TensorStoreAdapter(array)
            self._array = ds.copy(data=new_data)  # type: ignore
        return self._array  # type: ignore

    @property
    def translation(self) -> dict[str, float]:
        """Returns the translation of the image."""
        if not hasattr(self, "_translation"):
            # Get the translation of the image
            self._translation = {
                c: self.array.coords[c].values.min() for c in self.axes
            }
        return self._translation

    @property
    def original_scale(self) -> dict[str, Any] | None:
        """Returns the original scale of the image."""
        if self._original_scale is None:
            # Get the original scale of the image from poorly formatted metadata
            for level_data in self.group.attrs["multiscales"][0]["datasets"]:
                if level_data["path"] == self.scale_level:
                    level_data = level_data["coordinateTransformations"]
                    for transform in level_data:
                        if transform["type"] == "scale":
                            self._original_scale = {
                                c: transform["scale"][i]
                                for i, c in enumerate(self.axes)
                            }
                    break
        return self._original_scale

    @property
    def bounding_box(self) -> dict[str, list[float]]:
        """Returns the bounding box of the dataset in world units."""
        if self._bounding_box is None:
            self._bounding_box = {
                c: [self.translation[c], self.array.coords[c].values.max()]
                for c in self.axes
            }
        return self._bounding_box

    @property
    def sampling_box(self) -> dict[str, list[float]]:
        """Returns the sampling box of the dataset (i.e. where centers can be drawn from and still have full samples drawn from within the bounding box), in world units."""
        if self._sampling_box is None:
            self._sampling_box = {}
            output_padding = {c: np.ceil(s / 2) for c, s in self.output_size.items()}
            for c, (start, stop) in self.bounding_box.items():
                self._sampling_box[c] = [
                    start + output_padding[c],
                    stop - output_padding[c],
                ]
                assert (
                    self._sampling_box[c][0] < self._sampling_box[c][1]
                ), f"Sampling box for axis {c} is invalid: {self._sampling_box[c]} for image {self.path}. Image is not large enough to sample from as requested."
        return self._sampling_box

    @property
    def class_counts(self) -> int:
        """Returns the number of pixels for the contained class in the ground truth data, normalized by the resolution."""
        if self._class_counts is None:
            # Get from cellmap-schemas metadata, then normalize by resolution
            try:
                group = zarr.open(self.path, mode="r")
                annotation_group = AnnotationGroup.from_zarr(group)  # type: ignore
                self._class_counts = (
                    np.prod(self.array.shape)
                    - annotation_group.members[
                        self.scale_level
                    ].attrs.cellmap.annotation.complement_counts["absent"]
                ) * np.prod(list(self.scale.values()))
            except Exception as e:
                print(e)
                self._class_counts = 0
        return self._class_counts

    def to(self, device: str) -> None:
        """Sets what device returned image data will be loaded onto."""
        self.device = device

    def find_level(self, target_scale: dict[str, float]) -> str:
        """Finds the multiscale level that is closest to the target scale."""
        # Get the order of axes in the image
        axes = []
        for axis in self.group.attrs["multiscales"][0]["axes"]:
            if axis["type"] == "space":
                axes.append(axis["name"])

        last_path: str | None = None
        scale = {}
        for level in self.group.attrs["multiscales"][0]["datasets"]:
            for transform in level["coordinateTransformations"]:
                if "scale" in transform:
                    scale = {c: s for c, s in zip(axes, transform["scale"])}
                    break
            for c in axes:
                if scale[c] > target_scale[c]:
                    if last_path is None:
                        return level["path"]
                    else:
                        return last_path
            last_path = level["path"]
        return last_path  # type: ignore

    def set_spatial_transforms(self, transforms: dict[str, Any] | None):
        """Sets spatial transformations for the image data."""
        self._current_spatial_transforms = transforms

    def apply_spatial_transforms(
        self, coords: dict[str, Sequence[float]]
    ) -> torch.Tensor:
        """Applies spatial transformations to the given coordinates."""
        # Apply spatial transformations to the coordinates
        # TODO: Implement non-90 degree rotations
        if self._current_spatial_transforms is not None:
            for transform, params in self._current_spatial_transforms.items():
                if transform not in self.post_image_transforms:
                    if transform == "mirror":
                        for axis in params:
                            # TODO: Make sure this works and doesn't collapse to coords
                            coords[axis] = coords[axis][::-1]
                    else:
                        raise ValueError(f"Unknown spatial transform: {transform}")
        self._last_coords = coords

        # Pull data from the image
        data = self.return_data(coords)
        data = data.values

        # Apply and spatial transformations that require the image array (e.g. transpose)
        if self._current_spatial_transforms is not None:
            for transform, params in self._current_spatial_transforms.items():
                if transform in self.post_image_transforms:
                    if transform == "transpose":
                        # TODO ... make sure this works
                        # data = data.transpose(*params)
                        new_order = [params[c] for c in self.axes]
                        data = np.transpose(data, new_order)
                    else:
                        raise ValueError(f"Unknown spatial transform: {transform}")

        return torch.tensor(data)

    def return_data(self, coords: dict[str, Sequence[float]]):
        # Pull data from the image based on the given coordinates. This interpolates the data to the nearest pixel automatically.
        data = self.array.sel(
            **coords,  # type: ignore
            method="nearest",
        )

        return data


class EmptyImage:
    label_class: str
    axes: str
    store: torch.Tensor

    def __init__(
        self,
        target_class: str,
        target_voxel_shape: Sequence[int],
        store: Optional[torch.Tensor] = None,
        axis_order: str = "zyx",
        empty_value: float | int = -100,
    ):
        """Initializes an empty image object.

        Args:
            target_class (str): The label class of the image.
            target_voxel_shape (Sequence[int]): The shape of the image in voxels.
            store (Optional[torch.Tensor], optional): The tensor to return. Defaults to None.
            axis_order (str, optional): The order of the axes in the image. Defaults to "zyx".
        """
        self.label_class = target_class
        if len(target_voxel_shape) < len(axis_order):
            axis_order = axis_order[-len(target_voxel_shape) :]
        self.output_shape = {c: target_voxel_shape[i] for i, c in enumerate(axis_order)}
        self.axes = axis_order
        self._bounding_box = {c: [0, 2**32] for c in axis_order}
        self._class_counts = 0
        self.scale = {c: 1 for c in self.axes}
        self.empty_value = empty_value
        if store is not None:
            self.store = store
        else:
            self.store = (
                torch.ones([1] + [self.output_shape[c] for c in self.axes])
                * self.empty_value
            )

    def __getitem__(self, center: dict[str, float]) -> torch.Tensor:
        """Returns image data centered around the given point, based on the scale and shape of the target output image."""
        return self.store

    @property
    def bounding_box(self) -> dict[str, list[float]]:
        """Returns the bounding box of the dataset."""
        return self._bounding_box

    @property
    def sampling_box(self) -> dict[str, list[float]]:
        """Returns the sampling box of the dataset (i.e. where centers can be drawn from and still have full samples drawn from within the bounding box)."""
        return self._bounding_box

    @property
    def class_counts(self) -> int:
        """Returns the number of pixels for the contained class in the ground truth data, normalized by the resolution."""
        return self._class_counts

    def to(self, device: str) -> None:
        """Moves the image data to the given device."""
        self.store = self.store.to(device)

    def set_spatial_transforms(self, transforms: dict[str, Any] | None):
        pass
