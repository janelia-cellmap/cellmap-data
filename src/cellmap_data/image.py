import os
from typing import Callable, Iterable, Optional, Sequence
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
    translation: dict[str, float]
    scale: dict[str, float]
    output_shape: dict[str, int]
    output_size: dict[str, float]
    label_class: str
    array: xarray.DataArray
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
        self.scale = {c: s for c, s in zip(axis_order, target_scale)}
        self.output_shape = {c: target_voxel_shape[i] for i, c in enumerate(axis_order)}
        self.output_size = {
            c: target_voxel_shape[i] * target_scale[i] for i, c in enumerate(axis_order)
        }
        self.axes = axis_order
        self.value_transform = value_transform
        self.context = context
        self.construct()

    def __getitem__(self, center: dict[str, float]) -> torch.Tensor:
        """Returns image data centered around the given point, based on the scale and shape of the target output image."""
        # Find vectors of coordinates in world space to pull data from
        coords = {}
        for c in self.axes:
            if center[c] - self.output_size[c] / 2 < self.bounding_box[c][0]:
                raise ValueError(
                    f"Center {center[c]} is out of bounds for axis {c} in image {self.path}. {center[c] - self.output_size[c] / 2} would be less than {self.bounding_box[c][0]}"
                )
            if center[c] + self.output_size[c] / 2 > self.bounding_box[c][1]:
                raise ValueError(
                    f"Center {center[c]} is out of bounds for axis {c} in image {self.path}. {center[c] + self.output_size[c] / 2} would be greater than {self.bounding_box[c][1]}"
                )
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
        return data

    def construct(self):
        self._bounding_box = None
        self._sampling_box = None
        self._class_counts = None
        self._current_spatial_transforms = None
        self._last_coords = None
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
        self.array = ds.copy(data=new_data)  # type: ignore
        self.get_spatial_metadata()
        # self.xs = self.array.coords["x"]
        # self.ys = self.array.coords["y"]
        # self.zs = self.array.coords["z"]

    def find_level(self, target_scale: dict[str, float]) -> str:
        """Finds the multiscale level that is closest to the target scale."""
        # Get the order of axes in the image
        axes = []
        for axis in self.group.attrs["multiscales"][0]["axes"]:
            if axis["type"] == "space":
                axes.append(axis["name"])

        last_path: str = ""
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
        return last_path

    def get_spatial_metadata(self):
        """Gets the spatial metadata for the image."""
        # Get the translation of the image
        self.translation = {c: min(self.array.coords[c].values) for c in self.axes}

        # Get the original scale of the image from poorly formatted metadata
        for level_data in self.group.attrs["multiscales"][0]["datasets"]:
            if level_data["path"] == self.scale_level:
                level_data = level_data["coordinateTransformations"]
                for transform in level_data:
                    if transform["type"] == "scale":
                        self.original_scale = {
                            c: transform["scale"][i] for i, c in enumerate(self.axes)
                        }
                break

    def set_spatial_transforms(self, transforms: dict[str, any] | None):
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

    @property
    def bounding_box(self) -> dict[str, list[float]]:
        """Returns the bounding box of the dataset in world units."""
        if self._bounding_box is None:
            self._bounding_box = {
                c: [self.translation[c], max(self.array.coords[c].values)]
                for c in self.axes
            }
        return self._bounding_box

    @property
    def sampling_box(self) -> dict[str, list[float]]:
        """Returns the sampling box of the dataset (i.e. where centers can be drawn from and still have full samples drawn from within the bounding box), in world units."""
        if self._sampling_box is None:
            self._sampling_box = {}
            output_padding = {c: (s / 2) for c, s in self.output_size.items()}
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
            try:
                group = zarr.open(self.path, mode="r")
                annotation_group = AnnotationGroup.from_zarr(group)  # type: ignore
                self._class_counts = (
                    np.prod(self.array.shape)
                    - annotation_group.members[
                        self.scale_level
                    ].attrs.cellmap.annotation.complement_counts["absent"]
                )
            except Exception as e:
                print(e)
                self._class_counts = -1
        return self._class_counts


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
    ):
        """Initializes an empty image object.

        Args:
            target_class (str): The label class of the image.
            target_voxel_shape (Sequence[int]): The shape of the image in voxels.
            store (Optional[torch.Tensor], optional): The tensor to return. Defaults to None.
            axis_order (str, optional): The order of the axes in the image. Defaults to "zyx".
        """
        self.label_class = target_class
        self.output_shape = {c: target_voxel_shape[i] for i, c in enumerate(axis_order)}
        self.axes = axis_order
        self._bounding_box = None
        self._class_counts = 0
        self.scale = {c: 1 for c in self.axes}
        if store is not None:
            self.store = store
        else:
            self.store = torch.zeros([1] + [self.output_shape[c] for c in self.axes])

    def __getitem__(self, center: dict[str, float]) -> torch.Tensor:
        """Returns image data centered around the given point, based on the scale and shape of the target output image."""
        return self.store

    def set_spatial_transforms(self, transforms: dict[str, any] | None):
        pass

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
