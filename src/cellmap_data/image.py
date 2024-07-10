import os
from typing import Any, Callable, Mapping, Optional, Sequence
import torch
from fibsem_tools.io.core import read_xarray
import xarray
import tensorstore
import xarray_tensorstore as xt
import numpy as np
from cellmap_schemas.annotation import AnnotationArray
import zarr

from scipy.spatial.transform import Rotation as rot


class CellMapImage:
    path: str
    scale: Mapping[str, float]
    output_shape: Mapping[str, int]
    output_size: Mapping[str, float]
    label_class: str
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
        pad: bool = False,
        pad_value: float | int = np.nan,
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
        self.pad = pad
        self.pad_value = pad_value
        self.scale = {c: s for c, s in zip(axis_order, target_scale)}
        self.output_shape = {c: t for c, t in zip(axis_order, target_voxel_shape)}
        self.output_size = {
            c: t * s for c, t, s in zip(axis_order, target_voxel_shape, target_scale)
        }
        self.axes = axis_order[: len(target_voxel_shape)]
        self.value_transform = value_transform
        self.context = context
        self._current_spatial_transforms = None
        self._last_coords = None
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
        # TODO: array_path instead of path
        return f"CellMapImage({self.path})"

    @property
    def center(self):
        center = {}
        for c, (start, stop) in self.bounding_box.items():
            center[c] = start + (stop - start) / 2
        return center

    @property
    def scale_level(self) -> str:
        """Returns the multiscale level of the image."""
        if not hasattr(self, "_scale_level"):
            self._scale_level = self.find_level(self.scale)
        return self._scale_level

    @property
    def group(self) -> xarray.DataArray:
        if not hasattr(self, "_group"):
            self._group = read_xarray(self.path)
        return self._group  # type: ignore

    @property
    def array_path(self) -> str:
        if not hasattr(self, "_array_path"):
            self._array_path = os.path.join(self.path, self.scale_level)
        return self._array_path

    @property
    def array(self) -> xarray.DataArray:
        if not hasattr(self, "_array"):
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
    def translation(self) -> Mapping[str, float]:
        """Returns the translation of the image."""
        if not hasattr(self, "_translation"):
            # Get the translation of the image
            self._translation = {
                c: self.array.coords[c].values.min() for c in self.axes
            }
        return self._translation

    @property
    def original_scale(self) -> Mapping[str, Any]:
        """Returns the original scale of the image."""
        if not hasattr(self, "_original_scale"):
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
    def bounding_box(self) -> Mapping[str, list[float]]:
        """Returns the bounding box of the dataset in world units."""
        if not hasattr(self, "_bounding_box"):
            self._bounding_box = {
                c: [self.translation[c], self.array.coords[c].values.max()]
                for c in self.axes
            }
        return self._bounding_box

    @property
    def sampling_box(self) -> Mapping[str, list[float]]:
        """Returns the sampling box of the dataset (i.e. where centers can be drawn from and still have full samples drawn from within the bounding box), in world units."""
        if not hasattr(self, "_sampling_box") or self._sampling_box is None:
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
        if hasattr(self, "_bg_count"):
            # Get from cellmap-schemas metadata, then normalize by resolution - get class counts at same time
            _ = self.class_counts
        return self._bg_count

    @property
    def class_counts(self) -> float:
        """Returns the number of pixels for the contained class in the ground truth data, normalized by the resolution."""
        if not hasattr(self, "_class_counts"):
            # Get from cellmap-schemas metadata, then normalize by resolution
            try:
                annotation_attrs = AnnotationArray.from_zarr(
                    zarr.open(self.array_path, mode="r")  # type: ignore
                )
                if hasattr(annotation_attrs, "attributes"):
                    annotation_attrs = annotation_attrs.attributes.cellmap.annotation
                else:
                    # For backwards compatibility
                    annotation_attrs = annotation_attrs.attrs.cellmap.annotation  # type: ignore

                bg_count = annotation_attrs.complement_counts["absent"]  # type: ignore
                self._class_counts = (np.prod(self.array.shape) - bg_count) * np.prod(
                    list(self.scale.values())
                )
                self._bg_count = bg_count * np.prod(list(self.scale.values()))
            except Exception as e:
                print(e)
                self._class_counts = 0.0
                self._bg_count = 0.0
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
    ) -> Mapping[str, tuple[Sequence[str], np.ndarray]]:
        """Rotates the given coordinates by the given angles."""
        # Check to see if a rotation is necessary
        if not any([a != 0 for a in angles.values()]):
            return coords

        # Convert the coordinates dictionary to a vector
        coords_vector, axes_lengths = self._coord_dict_to_vector(coords)

        # Check if any angles would rotate in a singular axis
        singular_axes = set([c for c, l in axes_lengths.items() if l == 1])
        for c, a in angles.items():
            if a != 0:
                # A rotation around one axis rotates the orthogonal axes
                invalid_axes = (set(angles.keys()) - set([c])).intersection(
                    singular_axes
                )
                assert (
                    invalid_axes == set()
                ), f"Cannot rotate around axis {c} by {a} degrees, as it would rotate around a singular axes: {invalid_axes}."

        # Recenter the coordinates around the origin
        center = coords_vector.mean(axis=0)
        coords_vector -= center

        rotation_vector = [angles[c] for c in self.axes]
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

    def set_spatial_transforms(self, transforms: Mapping[str, Any] | None):
        """Sets spatial transformations for the image data."""
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
                raise NotImplementedError(
                    "How to return data properly, given rotated data is not yet implemented."
                )
            if "deform" in self._current_spatial_transforms:
                raise NotImplementedError("Deformations are not yet implemented.")
        self._last_coords = coords

        # Pull data from the image
        data = self.return_data(coords)
        data = data.values

        # Apply and spatial transformations that require the image array (e.g. transpose)
        if self._current_spatial_transforms is not None:
            for transform, params in self._current_spatial_transforms.items():
                if transform in self.post_image_transforms:
                    if transform == "transpose":
                        new_order = [params[c] for c in self.axes]
                        data = np.transpose(data, new_order)
                    else:
                        raise ValueError(f"Unknown spatial transform: {transform}")

        return torch.tensor(data)

    def return_data(
        self,
        coords: (
            Mapping[str, Sequence[float]]
            | Mapping[str, tuple[Sequence[str], np.ndarray]]
        ),
    ):
        # Pull data from the image based on the given coordinates. This interpolates the data to the nearest pixel automatically.
        if not isinstance(coords[list(coords.keys())[0]][0], float | int):
            # TODO: Need to make this work with self.pad = True
            if self.pad:
                raise NotImplementedError(
                    "Interpolation with padding is not yet implemented."
                )
            data = self.array.interp(
                coords=coords,
                method="nearest",  # TODO: This should depend on whether the image is a segmentation or not
            )
        elif self.pad:
            if not hasattr(self, "_tolerance"):
                self._tolerance = np.ones(coords[self.axes[0]].shape) * np.max(
                    list(self.scale.values())
                )
            data = self.array.reindex(
                **coords,
                method="nearest",
                tolerance=self._tolerance,
                fill_value=self.pad_value,
            )
        else:
            data = self.array.sel(
                **coords,  # type: ignore
                method="nearest",
            )
        return data


class EmptyImage:
    label_class: str
    axes: str
    store: torch.Tensor
    output_shape: Mapping[str, int]
    output_size: Mapping[str, float]
    scale: Mapping[str, float]

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
            target_class (str): The label class of the image.
            target_voxel_shape (Sequence[int]): The shape of the image in voxels.
            store (Optional[torch.Tensor], optional): The tensor to return. Defaults to None.
            axis_order (str, optional): The order of the axes in the image. Defaults to "zyx".
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
    def bg_count(self) -> float:
        """Returns the number of background pixels in the ground truth data, normalized by the resolution."""
        return self._bg_count

    @property
    def class_counts(self) -> float:
        """Returns the number of pixels for the contained class in the ground truth data, normalized by the resolution."""
        return self._class_counts

    def to(self, device: str) -> None:
        """Moves the image data to the given device."""
        self.store = self.store.to(device)

    def set_spatial_transforms(self, transforms: Mapping[str, Any] | None):
        pass
