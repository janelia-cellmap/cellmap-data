import os
from typing import Any, Callable, Mapping, Optional, Sequence
import torch

from xarray_ome_ngff.v04.multiscale import coords_from_transforms
from pydantic_ome_ngff.v04.multiscale import GroupAttrs, MultiscaleMetadata
from pydantic_ome_ngff.v04.transform import Scale, Translation

import xarray
import tensorstore
import xarray_tensorstore as xt
import numpy as np
import zarr

from scipy.spatial.transform import Rotation as rot


class CellMapImage:
    """
    A class for handling image data from a CellMap dataset.

    This class is used to load image data from a CellMap dataset, and can apply spatial transformations to the data. It also handles the loading of the image data from the dataset, and can apply value transformations to the data. The image data is returned as a PyTorch tensor formatted for use in training, and can be loaded onto a specified device.

    Attributes:
        path (str): The path to the multiscale image file.
        label_class (str): The label class of the image.
        scale (Mapping[str, float]): The scale of the image in physical space.
        output_shape (Mapping[str, int]): The shape of data returned from image queries in voxels.
        axes (str): The order of the axes in the image.
        value_transform (Optional[Callable]): A function to transform the image data, such as adding gaussian noise.
        context (Optional[tensorstore.Context]): The context for the TensorStore.

    Methods:
        __getitem__(center: Mapping[str, float]) -> torch.Tensor: Returns image data centered around the given point.
        __repr__() -> str: Returns a string representation of the CellMapImage object.
        apply_spatial_transforms(coords: Mapping[str, Sequence[float]]) -> torch.Tensor: Applies spatial transformations to the given coordinates and returns the tensor of the resulting data.
        return_data(coords: Mapping[str, Sequence[float]]) -> xarray.DataArray: Pulls data from the image based on the given coordinates and returns the data as an xarray DataArray.
        set_spatial_transforms(transforms: Mapping[str, Any] | None): Sets spatial transformations for the image data, used for setting uniform transformations to all images within a dataset.
        to(device: str): Sets what device returned image data will be loaded onto.

    Properties:

        shape (Mapping[str, int]): Returns the shape of the image in voxels.
        center (Mapping[str, float]): Returns the center of the image in world units.
        multiscale_attrs: Returns CellMap specified multiscale metadata of the image.
        coordinateTransformations: Returns the coordinate transformations of the image specified in its metadata (e.g. scale and translation).
        full_coords: Returns the full coordinates of the image in world units.
        scale_level (str): Returns the path extension for the multiscale level of the image (e.g. 's0').
        group (zarr.Group): Returns the multiscale zarr group object for the image.
        array_path (str): Returns the path to the single-scale image array.
        array (xarray.DataArray): Returns the image data as an xarray DataArray.
        translation (Mapping[str, float]): Returns the translation of the image in world units.
        bounding_box (Mapping[str, list[float]]): Returns the bounding box of the dataset in world units.
        sampling_box (Mapping[str, list[float]]): Returns the sampling box of the dataset in world units.
        bg_count (float): Returns the number of background pixels in the ground truth data, normalized by the resolution.
        class_counts (float): Returns the number of pixels for the contained class in the ground truth data, normalized by the resolution.
        device (str): Returns the device that the image data will be loaded onto.
    """

    def __init__(
        self,
        path: str,
        target_class: str,
        target_scale: Sequence[float],
        target_voxel_shape: Sequence[int],
        pad: bool = False,
        pad_value: float | int = np.nan,
        interpolation: str = "nearest",
        axis_order: str | Sequence[str] = "zyx",
        value_transform: Optional[Callable] = None,
        context: Optional[tensorstore.Context] = None,  # type: ignore
    ) -> None:
        """Initializes a CellMapImage object.

        Args:
            path (str): The path to the image file.
            target_class (str): The label class of the image.
            target_scale (Sequence[float]): The scale of the image data to return in physical space.
            target_voxel_shape (Sequence[int]): The shape of the image data to return in voxels.
            axis_order (str, optional): The order of the axes in the image. Defaults to "zyx".
            value_transform (Optional[callable], optional): A function to transform the image pixel data. Defaults to None.
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
        self.pad = pad
        self.pad_value = pad_value
        self.interpolation = interpolation
        self.scale = {c: s for c, s in zip(axis_order, target_scale)}
        self.output_shape = {c: t for c, t in zip(axis_order, target_voxel_shape)}
        self.output_size = {
            c: t * s for c, t, s in zip(axis_order, target_voxel_shape, target_scale)
        }
        self.axes = axis_order[: len(target_voxel_shape)]
        self.value_transform = value_transform
        self.context = context
        self._current_spatial_transforms = None
        self._current_coords = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __getitem__(self, center: Mapping[str, float]) -> torch.Tensor:
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
        """Returns a string representation of the CellMapImage object."""
        return f"CellMapImage({self.array_path})"

    @property
    def shape(self) -> Mapping[str, int]:
        """Returns the shape of the image."""
        try:
            return self._shape
        except AttributeError:
            self._shape: dict[str, int] = {
                c: self.group[self.scale_level].shape[i]
                for i, c in enumerate(self.axes)
            }
            return self._shape

    @property
    def center(self) -> Mapping[str, float]:
        """Returns the center of the image in world units."""
        try:
            return self._center
        except AttributeError:
            center = {}
            for c, (start, stop) in self.bounding_box.items():
                center[c] = start + (stop - start) / 2
            self._center: dict[str, float] = center
            return self._center

    @property
    def multiscale_attrs(self) -> MultiscaleMetadata:
        """Returns the multiscale metadata of the image."""
        try:
            return self._multiscale_attrs
        except AttributeError:
            self._multiscale_attrs: MultiscaleMetadata = GroupAttrs(
                multiscales=self.group.attrs["multiscales"]
            ).multiscales[0]
            return self._multiscale_attrs

    @property
    def coordinateTransformations(
        self,
    ) -> tuple[Scale] | tuple[Scale, Translation]:
        """Returns the coordinate transformations of the image, based on the multiscale metadata."""
        try:
            return self._coordinateTransformations
        except AttributeError:
            # multi_tx = multi.coordinateTransformations
            dset = [
                ds
                for ds in self.multiscale_attrs.datasets
                if ds.path == self.scale_level
            ][0]
            # tx_fused = normalize_transforms(multi_tx, dset.coordinateTransformations)
            self._coordinateTransformations = dset.coordinateTransformations
            return self._coordinateTransformations

    @property
    def full_coords(self) -> tuple[xarray.DataArray, ...]:
        """Returns the full coordinates of the image's axes in world units."""
        try:
            return self._full_coords
        except AttributeError:
            self._full_coords = coords_from_transforms(
                axes=self.multiscale_attrs.axes,
                transforms=self.coordinateTransformations,  # type: ignore
                # transforms=tx_fused,
                shape=self.group[self.scale_level].shape,  # type: ignore
            )
            return self._full_coords

    @property
    def scale_level(self) -> str:
        """Returns the multiscale level of the image."""
        try:
            return self._scale_level
        except AttributeError:
            self._scale_level = self.find_level(self.scale)
            return self._scale_level

    @property
    def group(self) -> zarr.Group:
        """Returns the zarr group object for the multiscale image."""
        try:
            return self._group
        except AttributeError:
            if self.path[:5] == "s3://":
                self._group = zarr.open_group(zarr.N5FSStore(self.path, anon=True))
            else:
                self._group = zarr.open_group(self.path)
            return self._group

    @property
    def array_path(self) -> str:
        """Returns the path to the single-scale image array."""
        try:
            return self._array_path
        except AttributeError:
            self._array_path = os.path.join(self.path, self.scale_level)
            return self._array_path

    @property
    def array(self) -> xarray.DataArray:
        """Returns the image data as an xarray DataArray."""
        try:
            return self._array
        except AttributeError:
            # Construct an xarray with Tensorstore backend
            spec = xt._zarr_spec_from_path(self.array_path)
            array_future = tensorstore.open(
                spec, read=True, write=False, context=self.context
            )
            try:
                array = array_future.result()
            except ValueError as e:
                Warning(e)
                UserWarning("Falling back to zarr3 driver")
                spec["driver"] = "zarr3"
                array_future = tensorstore.open(
                    spec, read=True, write=False, context=self.context
                )
                array = array_future.result()
            data = xt._TensorStoreAdapter(array)
            self._array = xarray.DataArray(data=data, coords=self.full_coords)
            return self._array

    @property
    def translation(self) -> Mapping[str, float]:
        """Returns the translation of the image."""
        try:
            return self._translation
        except AttributeError:
            # Get the translation of the image
            self._translation = {c: self.bounding_box[c][0] for c in self.axes}
            return self._translation

    @property
    def bounding_box(self) -> Mapping[str, list[float]]:
        """Returns the bounding box of the dataset in world units."""
        try:
            return self._bounding_box
        except AttributeError:
            self._bounding_box = {}
            for coord in self.full_coords:
                self._bounding_box[coord.dims[0]] = [coord.data.min(), coord.data.max()]
            return self._bounding_box

    @property
    def sampling_box(self) -> Mapping[str, list[float]]:
        """Returns the sampling box of the dataset (i.e. where centers can be drawn from and still have full samples drawn from within the bounding box), in world units."""
        try:
            return self._sampling_box
        except AttributeError:
            self._sampling_box = {}
            output_padding = {c: np.ceil(s / 2) for c, s in self.output_size.items()}
            for c, (start, stop) in self.bounding_box.items():
                self._sampling_box[c] = [
                    start + output_padding[c],
                    stop - output_padding[c],
                ]
                try:
                    assert (
                        self._sampling_box[c][0] < self._sampling_box[c][1]
                    ), f"Sampling box for axis {c} is invalid: {self._sampling_box[c]} for image {self.path}. Image is not large enough to sample from as requested."
                except AssertionError as e:
                    if self.pad:
                        self._sampling_box[c] = [
                            self.center[c] - self.scale[c],
                            self.center[c] + self.scale[c],
                        ]
                    else:
                        self._sampling_box = None
                        raise e
            return self._sampling_box

    @property
    def bg_count(self) -> float:
        """Returns the number of background pixels in the ground truth data, normalized by the resolution."""
        try:
            return self._bg_count
        except AttributeError:
            # Get from cellmap-schemas metadata, then normalize by resolution - get class counts at same time
            _ = self.class_counts
            return self._bg_count

    @property
    def class_counts(self) -> float:
        """Returns the number of pixels for the contained class in the ground truth data, normalized by the resolution."""
        try:
            return self._class_counts  # type: ignore
        except AttributeError:
            # Get from cellmap-schemas metadata, then normalize by resolution
            try:
                # TODO: Make work with HDF5 files
                bg_count = self.group[self.scale_level].attrs["cellmap"]["annotation"][
                    "complement_counts"
                ]["absent"]
                self._class_counts = (
                    np.prod(self.group[self.scale_level].shape) - bg_count
                ) * np.prod(list(self.scale.values()))
                self._bg_count = bg_count * np.prod(list(self.scale.values()))
            except Exception as e:
                print(e)
                self._class_counts = 0.1
                self._bg_count = 0.1
            return self._class_counts  # type: ignore

    def to(self, device: str) -> None:
        """Sets what device returned image data will be loaded onto."""
        self.device = device

    def find_level(self, target_scale: Mapping[str, float]) -> str:
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

    def rotate_coords(
        self, coords: Mapping[str, Sequence[float]], angles: Mapping[str, float]
    ) -> Mapping[str, tuple[Sequence[str], np.ndarray]] | Mapping[str, Sequence[float]]:
        """Rotates the given coordinates by the given angles."""
        # Check to see if a rotation is necessary
        if not any([a != 0 for a in angles.values()]):
            return coords

        # Convert the coordinates dictionary to a vector
        coords_vector, axes_lengths = self._coord_dict_to_vector(coords)

        # Recenter the coordinates around the origin
        center = coords_vector.mean(axis=0)
        coords_vector -= center

        rotation_vector = [angles[c] if c in angles else 0 for c in self.axes]
        rotator = rot.from_rotvec(rotation_vector, degrees=True)

        # Apply the rotation
        rotated_coords = rotator.apply(coords_vector)

        # Recenter the coordinates around the original center
        rotated_coords += center
        return self._coord_vector_to_grid_dict(rotated_coords, axes_lengths)

    def _coord_dict_to_vector(
        self, coords_dict: Mapping[str, Sequence[float]]
    ) -> tuple[np.ndarray, Mapping[str, int]]:
        """Converts a dictionary of coordinates to a vector, for use with rotate_coords."""
        coord_vector = np.stack(
            np.meshgrid(*[coords_dict[c] for c in self.axes]), axis=-1
        ).reshape(-1, len(self.axes))
        axes_lengths = {c: len(coords_dict[c]) for c in self.axes}
        return coord_vector, axes_lengths

    def _coord_vector_to_grid_dict(
        self, coords_vector: np.ndarray, axes_lengths: Mapping[str, int]
    ) -> Mapping[str, tuple[Sequence[str], np.ndarray]]:
        """Converts a vector of coordinates to a grid type dictionary."""
        shape = [axes_lengths[c] for c in self.axes]
        axes = [c for c in self.axes]
        coords_dict = {
            c: (axes, coords_vector[:, self.axes.index(c)].reshape(shape))
            for c in self.axes
        }

        return coords_dict

    def set_spatial_transforms(self, transforms: Mapping[str, Any] | None) -> None:
        """Sets spatial transformations for the image data, for setting global transforms at the 'dataset' level."""
        self._current_spatial_transforms = transforms

    def apply_spatial_transforms(self, coords) -> torch.Tensor:
        """Applies spatial transformations to the given coordinates."""
        # Apply spatial transformations to the coordinates
        # Because some spatial transformations require the image array, we need to apply them after pulling the data. This is done by separating the transforms into two groups
        if self._current_spatial_transforms is not None:
            # Because of the implementation details, we explicitly apply transforms in a specific order
            if "mirror" in self._current_spatial_transforms:
                for axis in self._current_spatial_transforms["mirror"]:
                    # Assumes the coords are the default xarray format
                    coords[axis] = coords[axis][::-1]
            if "rotate" in self._current_spatial_transforms:
                # Assumes the coords are the default xarray format, and that the rotation is in degrees
                # Converts the coordinates to a vector, rotates them, then converts them to a grid dictionary
                coords = self.rotate_coords(
                    coords, self._current_spatial_transforms["rotate"]
                )
            if "deform" in self._current_spatial_transforms:
                raise NotImplementedError("Deformations are not yet implemented.")
        self._current_coords = coords

        # Pull data from the image
        data = self.return_data(coords)
        data = data.values

        # Apply and spatial transformations that require the image array (e.g. transpose)
        if self._current_spatial_transforms is not None:
            for transform, params in self._current_spatial_transforms.items():
                # if transform in self.post_image_transforms:
                if transform == "transpose":
                    new_order = [params[c] for c in self.axes]
                    data = np.transpose(data, new_order)
                    # else:
                    #     raise ValueError(f"Unknown spatial transform: {transform}")

        return torch.tensor(data)

    def return_data(
        self,
        coords: (
            Mapping[str, Sequence[float]]
            | Mapping[str, tuple[Sequence[str], np.ndarray]]
        ),
    ) -> xarray.DataArray:
        """Pulls data from the image based on the given coordinates, applying interpolation if necessary, and returns the data as an xarray DataArray."""
        if not isinstance(coords[list(coords.keys())[0]][0], float | int):
            data = self.array.interp(
                coords=coords,
                method=self.interpolation,  # type: ignore
            )
        elif self.pad:
            try:
                tolerance = self._tolerance
            except AttributeError:
                self._tolerance = np.ones(coords[self.axes[0]].shape) * np.max(
                    list(self.scale.values())
                )
                tolerance = self._tolerance
            data = self.array.reindex(
                **coords,
                method="nearest",
                tolerance=tolerance,
                fill_value=self.pad_value,
            )
        else:
            data = self.array.sel(
                **coords,
                method="nearest",
            )
        return data


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
        """Initializes an empty image object.

        Args:

            target_class (str): The intended label class of the image.
            target_scale (Sequence[float]): The intended scale of the image in physical space.
            target_voxel_shape (Sequence[int]): The intended shape of the image in voxels.
            store (Optional[torch.Tensor], optional): The tensor to return. Defaults to None.
            axis_order (str, optional): The intended order of the axes in the image. Defaults to "zyx".
            empty_value (float | int, optional): The value to fill the image with. Defaults to -100.
        """
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
                torch.ones([1] + [self.output_shape[c] for c in self.axes])
                * self.empty_value
            )

    def __getitem__(self, center: Mapping[str, float]) -> torch.Tensor:
        """Returns the empty image data."""
        return self.store

    @property
    def bounding_box(self) -> None:
        """Returns the bounding box of the dataset. Returns None for an EmptyImage object."""
        return self._bounding_box

    @property
    def sampling_box(self) -> None:
        """Returns the sampling box of the dataset (i.e. where centers can be drawn from and still have full samples drawn from within the bounding box). Returns None for an EmptyImage object."""
        return self._bounding_box

    @property
    def bg_count(self) -> float:
        """Returns the number of background pixels in the ground truth data, normalized by the resolution. Returns zero for an EmptyImage object."""
        return self._bg_count

    @property
    def class_counts(self) -> float:
        """Returns the number of pixels for the contained class in the ground truth data, normalized by the resolution. Returns zero for an EmptyImage object."""
        return self._class_counts

    def to(self, device: str) -> None:
        """Moves the image data to the given device."""
        self.store = self.store.to(device)

    def set_spatial_transforms(self, transforms: Mapping[str, Any] | None) -> None:
        """Imitates the method in CellMapImage, but does nothing for an EmptyImage object."""
        pass
