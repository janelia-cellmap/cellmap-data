import os
import logging
from typing import Any, Callable, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

import numpy as np
import tensorstore
import dask.array as da
import torch
import xarray
import xarray_tensorstore as xt
import zarr
from pydantic_ome_ngff.v04.multiscale import MultiscaleGroupAttrs, MultiscaleMetadata
from pydantic_ome_ngff.v04.transform import (
    Scale,
    Translation,
    VectorScale,
)
from scipy.spatial.transform import Rotation as rot
from xarray_ome_ngff.v04.multiscale import coords_from_transforms


class CellMapImage:
    """
    A class for handling image data from a CellMap dataset.

    This class is used to load image data from a CellMap dataset, and can apply spatial transformations to the data. It also handles the loading of the image data from the dataset, and can apply value transformations to the data. The image data is returned as a PyTorch tensor formatted for use in training, and can be loaded onto a specified device.
    """

    def __init__(
        self,
        path: str,
        target_class: str,
        target_scale: Sequence[float],  # TODO: make work with dict
        target_voxel_shape: Sequence[int],  # TODO: make work with dict
        pad: bool = False,
        pad_value: float | int = np.nan,
        interpolation: str = "nearest",
        axis_order: str | Sequence[str] = "zyx",
        value_transform: Optional[Callable] = None,
        context: Optional[tensorstore.Context] = None,  # type: ignore
        device: Optional[str | torch.device] = None,
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
            device (Optional[str | torch.device], optional): The device to load the image data onto. Defaults to "cuda" if available, then "mps", then "cpu".
        """

        self.path = path
        self.label_class = target_class
        # Below makes assumptions about image scale, and also locks which axis is sliced to 2D (this should only be encountered if bypassing dataset)
        if len(axis_order) > len(target_scale):
            logger.info(
                f"Axis order {axis_order} has more axes than target scale {target_scale}. Padding target scale with first given scale ({target_scale[0]})."
            )
            target_scale = [target_scale[0]] * (
                len(axis_order) - len(target_scale)
            ) + list(target_scale)
        if len(axis_order) > len(target_voxel_shape):
            ndim_fix = len(axis_order) - len(target_voxel_shape)
            logger.warning(
                f"Axis order {axis_order} has more axes than target voxel shape {target_voxel_shape}. Padding first {ndim_fix} target voxel shapes with 1s."
            )
            target_voxel_shape = [1] * ndim_fix + list(target_voxel_shape)
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
        self._current_center = None
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

    def __getitem__(self, center: Mapping[str, float]) -> torch.Tensor:
        """Returns image data centered around the given point, based on the scale and shape of the target output image."""
        if isinstance(list(center.values())[0], int | float):
            self._current_center = center

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
                    center[c] - self.output_size[c] / 2 + self.scale[c] / 2,
                    center[c] + self.output_size[c] / 2 - self.scale[c] / 2,
                    self.output_shape[c],
                )

            # Apply any spatial transformations to the coordinates and return the image data as a PyTorch tensor
            data = self.apply_spatial_transforms(coords)
        else:
            self._current_center = {k: np.mean(v) for k, v in center.items()}
            self._current_coords = center
            # Optimized tensor creation: use torch.from_numpy when possible to avoid data copying
            array_data = self.return_data(self._current_coords).values
            if isinstance(array_data, np.ndarray):
                data = torch.from_numpy(array_data)
            else:
                data = torch.tensor(array_data)  # type: ignore

        # Apply any value transformations to the data
        if self.value_transform is not None:
            data = self.value_transform(data)

        # Return data on CPU - let the DataLoader handle device transfer with streams
        # This avoids redundant transfers and allows for optimized batch transfers
        return data

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
            self._multiscale_attrs: MultiscaleMetadata = MultiscaleGroupAttrs(
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
                self._group = zarr.open_group(
                    zarr.N5FSStore(self.path, anon=True), mode="r"
                )
            else:
                self._group = zarr.open_group(self.path, mode="r")
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
            if (
                os.environ.get("CELLMAP_DATA_BACKEND", "tensorstore").lower()
                != "tensorstore"
            ):
                data = da.from_array(
                    self.group[self.scale_level],
                    chunks="auto",
                )
            else:
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
                bg_count = self.group["s0"].attrs["cellmap"]["annotation"][
                    "complement_counts"
                ]["absent"]
                for scale in self.group.attrs["multiscales"][0]["datasets"]:
                    if scale["path"] == "s0":
                        for transform in scale["coordinateTransformations"]:
                            if "scale" in transform:
                                s0_scale = transform["scale"]
                                break
                        break
                self._class_counts = (
                    np.prod(self.group["s0"].shape) - bg_count
                ) * np.prod(s0_scale)
                self._bg_count = bg_count * np.prod(s0_scale)
            except Exception as e:
                logger.warning(f"Error: {e}")
                logger.warning(f"Unable to get class counts for {self.path}")
                # logger.warning("from metadata, falling back to giving foreground 1 pixel, and the rest to background.")
                self._class_counts = np.prod(list(self.scale.values()))
                self._bg_count = (
                    np.prod(self.group[self.scale_level].shape) - 1
                ) * np.prod(list(self.scale.values()))
            return self._class_counts  # type: ignore

    def to(self, device: str, *args, **kwargs) -> None:
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
            if "transpose" in self._current_spatial_transforms:
                new_order = [
                    self._current_spatial_transforms["transpose"][c] for c in self.axes
                ]
                data = np.transpose(data, new_order)

        # Optimized tensor creation: use torch.from_numpy when possible to avoid data copying
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data)
        else:
            return torch.tensor(data)

    @property
    def tolerance(self) -> float:
        """Returns the tolerance for nearest neighbor interpolation."""
        try:
            return self._tolerance
        except AttributeError:
            # Calculate the tolerance as half the norm of the original image scale (i.e. traversing half a pixel diagonally) +  epsilon (1e-6)
            actual_scale = [
                ct
                for ct in self.coordinateTransformations
                if isinstance(ct, VectorScale)
            ][0].scale
            half_diagonal = np.linalg.norm(actual_scale) / 2
            self._tolerance = float(half_diagonal + 1e-6)
            return self._tolerance

    def return_data(
        self,
        coords: (
            Mapping[str, Sequence[float]]
            | Mapping[str, tuple[Sequence[str], np.ndarray]]
        ),
    ) -> xarray.DataArray:
        """Pulls data from the image based on the given coordinates, applying interpolation if necessary, and returns the data as an xarray DataArray."""
        if not isinstance(list(coords.values())[0][0], float | int):
            data = self.array.interp(
                coords=coords,
                method=self.interpolation,  # type: ignore
            )
        elif self.pad:
            data = self.array.reindex(
                **coords,
                method="nearest",
                tolerance=self.tolerance,
                fill_value=self.pad_value,
            )
        else:
            data = self.array.sel(
                **coords,
                method="nearest",
            )
        if (
            os.environ.get("CELLMAP_DATA_BACKEND", "tensorstore").lower()
            != "tensorstore"
        ):
            # NOTE: Forcing eager loading of dask array here may cause high memory usage and block further lazy optimizations.
            # Consider removing this or delaying loading until strictly necessary.
            data.load(scheduler="threads")
        return data
