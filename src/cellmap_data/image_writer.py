import os
from typing import Mapping, Optional, Sequence, Union

import numpy as np
import tensorstore
import torch
import xarray
import xarray_tensorstore as xt
from numpy.typing import ArrayLike
from pydantic_ome_ngff.v04.axis import Axis
from pydantic_ome_ngff.v04.transform import VectorScale, VectorTranslation
from upath import UPath
from xarray_ome_ngff.v04.multiscale import coords_from_transforms

from cellmap_data.utils import create_multiscale_metadata


class ImageWriter:
    """
    This class is used to write image data to a single-resolution zarr.
    ...existing docstring...
    """

    def __init__(
        self,
        path: str | UPath,
        target_class: str,
        scale: Mapping[str, float] | Sequence[float],
        bounding_box: Mapping[str, list[float]],
        write_voxel_shape: Mapping[str, int] | Sequence[int],
        scale_level: int = 0,
        axis_order: str = "zyx",
        context: Optional[tensorstore.Context] = None,
        overwrite: bool = False,
        dtype: np.dtype = np.float32,
        fill_value: float | int = 0,
    ) -> None:
        self.base_path = str(path)
        self.path = (UPath(path) / f"s{scale_level}").path
        self.label_class = self.target_class = target_class
        if isinstance(scale, Sequence):
            scale = {c: s for c, s in zip(axis_order[::-1], scale[::-1])}
        self.scale = scale
        if isinstance(write_voxel_shape, Sequence):
            if len(axis_order) > len(write_voxel_shape):  # TODO: This might be a bug
                write_voxel_shape = [1] * (
                    len(axis_order) - len(write_voxel_shape)
                ) + list(write_voxel_shape)
            elif (
                len(axis_order) + 1 == len(write_voxel_shape) and "c" not in axis_order
            ):
                axis_order = "c" + axis_order
            write_voxel_shape = {c: t for c, t in zip(axis_order, write_voxel_shape)}
        self.axes = axis_order
        # Assume axes correspond to last dimensions of voxel shape
        self.spatial_axes = axis_order[-len(scale) :]
        self.bounding_box = bounding_box
        self.write_voxel_shape = write_voxel_shape
        self.write_world_shape = {
            c: write_voxel_shape[c] * scale[c] for c in self.spatial_axes
        }
        self.scale_level = scale_level
        self.context = context
        self.overwrite = overwrite
        self.dtype = dtype
        self.fill_value = fill_value
        self.metadata = {
            "offset": list(self.offset.values()),
            "axes": [c for c in axis_order],
            "voxel_size": list(self.scale.values()),
            "shape": list(self.shape.values()),
            "units": "nanometer",
            "chunk_shape": list(write_voxel_shape.values()),
        }

    @property
    def array(self) -> xarray.DataArray:
        try:
            return self._array
        except AttributeError:
            os.makedirs(UPath(self.base_path), exist_ok=True)
            group_path = str(self.base_path).split(".zarr")[0] + ".zarr"
            for group in [""] + list(
                UPath(str(self.base_path).split(".zarr")[-1]).parts
            )[1:]:
                group_path = UPath(group_path) / group
                with open(group_path / ".zgroup", "w") as f:
                    f.write('{"zarr_format": 2}')
            create_multiscale_metadata(
                ds_name=str(self.base_path),
                voxel_size=self.metadata["voxel_size"],
                translation=self.metadata["offset"],
                units=self.metadata["units"],
                axes=self.metadata["axes"],
                base_scale_level=self.scale_level,
                levels_to_add=0,
                out_path=str(UPath(self.base_path) / ".zattrs"),
            )
            spec = {
                "driver": "zarr",
                "kvstore": {"driver": "file", "path": self.path},
            }
            open_kwargs = {
                "read": True,
                "write": True,
                "create": True,
                "delete_existing": self.overwrite,
                "dtype": self.dtype,
                "shape": list(self.shape.values()),
                "fill_value": self.fill_value,
                "chunk_layout": tensorstore.ChunkLayout(
                    write_chunk_shape=self.chunk_shape
                ),
                "context": self.context,
            }
            array_future = tensorstore.open(
                spec,
                **open_kwargs,
            )
            try:
                array = array_future.result()
            except ValueError as e:
                if "ALREADY_EXISTS" in str(e):
                    raise FileExistsError(
                        f"Image already exists at {self.path}. Set overwrite=True to overwrite the image."
                    )
                Warning(e)
                UserWarning("Falling back to zarr3 driver")
                spec["driver"] = "zarr3"
                array_future = tensorstore.open(spec, **open_kwargs)
                array = array_future.result()
            from pydantic_ome_ngff.v04.axis import Axis
            from pydantic_ome_ngff.v04.transform import VectorScale, VectorTranslation
            from xarray_ome_ngff.v04.multiscale import coords_from_transforms

            data = xarray.DataArray(
                data=xt._TensorStoreAdapter(array),
                coords=coords_from_transforms(
                    axes=[
                        Axis(
                            name=c,
                            type="space" if c != "c" else "channel",
                            unit="nm" if c != "c" else "",
                        )
                        for c in self.axes
                    ],
                    transforms=(
                        VectorScale(scale=tuple(self.scale.values())),
                        VectorTranslation(translation=tuple(self.offset.values())),
                    ),
                    shape=tuple(self.shape.values()),
                ),
            )
            self._array = data
            with open(UPath(self.path) / ".zattrs", "w") as f:
                f.write("{}")
            return self._array

    @property
    def chunk_shape(self) -> Sequence[int]:
        try:
            return self._chunk_shape
        except AttributeError:
            self._chunk_shape = list(self.write_voxel_shape.values())
            return self._chunk_shape

    @property
    def world_shape(self) -> Mapping[str, float]:
        try:
            return self._world_shape
        except AttributeError:
            self._world_shape = {
                c: self.bounding_box[c][1] - self.bounding_box[c][0]
                for c in self.spatial_axes
            }
            return self._world_shape

    @property
    def shape(self) -> Mapping[str, int]:
        try:
            return self._shape
        except AttributeError:
            self._shape = {
                c: int(np.ceil(self.world_shape[c] / self.scale[c]))
                for c in self.spatial_axes
            }
            return self._shape

    @property
    def center(self) -> Mapping[str, float]:
        try:
            return self._center
        except AttributeError:
            self._center = {
                str(k): float(np.mean(v)) for k, v in self.array.coords.items()
            }
            return self._center

    @property
    def offset(self) -> Mapping[str, float]:
        try:
            return self._offset
        except AttributeError:
            self._offset = {c: self.bounding_box[c][0] for c in self.spatial_axes}
            return self._offset

    @property
    def full_coords(self) -> tuple[xarray.DataArray, ...]:
        try:
            return self._full_coords
        except AttributeError:
            self._full_coords = coords_from_transforms(
                axes=[
                    Axis(
                        name=c,
                        type="space" if c != "c" else "channel",
                        unit="nm" if c != "c" else "",
                    )
                    for c in self.axes
                ],
                transforms=(
                    VectorScale(scale=tuple(self.scale.values())),
                    VectorTranslation(translation=tuple(self.offset.values())),
                ),
                shape=tuple(self.shape.values()),
            )
            return self._full_coords

    def align_coords(
        self, coords: Mapping[str, tuple[Sequence, np.ndarray]]
    ) -> Mapping[str, tuple[Sequence, np.ndarray]]:
        aligned_coords = {}
        for c in self.spatial_axes:
            aligned_coords[c] = np.array(
                self.array.coords[c][
                    np.abs(np.array(self.array.coords[c])[:, None] - coords[c]).argmin(
                        axis=0
                    )
                ]
            ).squeeze()
        return aligned_coords

    def aligned_coords_from_center(self, center: Mapping[str, float]):
        coords = {}
        for c in self.axes:
            # Use center-of-voxel alignment
            start_requested = (
                center[c] - self.write_world_shape[c] / 2 + self.scale[c] / 2
            )
            start_aligned_idx = int(
                np.abs(self.array.coords[c] - start_requested).argmin()
            )
            coords[c] = self.array.coords[c][
                start_aligned_idx : start_aligned_idx + self.write_voxel_shape[c]
            ]
        return coords

    def __setitem__(
        self,
        coords: Union[Mapping[str, float], Mapping[str, tuple[Sequence, np.ndarray]]],
        data: Union[torch.Tensor, ArrayLike, float, int],
    ) -> None:
        """
        Set data at the specified coordinates.

        This method handles two types of coordinate inputs:
        1. Center coordinates: mapping axis names to float values
        2. Batch coordinates: mapping axis names to sequences of coordinates

        Args:
        ----
            coords: Either center coordinates or batch coordinates
            data: Data to write at the coordinates
        """
        first_coord_value = next(iter(coords.values()))

        if isinstance(first_coord_value, (int, float)):
            # Handle single item with center coordinates
            self._write_single_item(coords, data)  # type: ignore
        else:
            # Handle batch of items with coordinate sequences
            self._write_batch_items(coords, data)  # type: ignore

    def _write_single_item(
        self,
        center_coords: Mapping[str, float],
        data: Union[torch.Tensor, ArrayLike],
    ) -> None:
        """Write a single data item using center coordinates."""
        # Convert center coordinates to aligned array coordinates
        aligned_coords = self.aligned_coords_from_center(center_coords)

        # Convert data to numpy array with correct dtype
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        data_array = np.array(data).astype(self.dtype)

        # Remove batch dimension if present
        if data_array.ndim == len(self.axes) + 1 and data_array.shape[0] == 1:
            data_array = np.squeeze(data_array, axis=0)

        # Check for shape mismatches
        expected_shape = tuple(self.write_voxel_shape[c] for c in self.axes)
        if data_array.shape != expected_shape:
            if len(data_array.shape) < len(expected_shape) and 1 in expected_shape:
                # Try to expand dimensions to fit expected shape
                for axis, size in enumerate(expected_shape):
                    if size == 1:
                        data_array = np.expand_dims(data_array, axis=axis)
            else:
                raise ValueError(
                    f"Data shape {data_array.shape} does not match expected shape {expected_shape}."
                )
        coord_shape = tuple(len(aligned_coords[c]) for c in self.axes)
        if coord_shape != expected_shape:
            # Try to crop data to fit within bounds if necessary
            min_shape = tuple(min(c, e) for c, e in zip(coord_shape, expected_shape))
            slices = tuple(slice(0, s) for s in min_shape)
            data_array = data_array[slices]
            if data_array.shape != coord_shape:
                raise ValueError(
                    f"Aligned coordinates shape {coord_shape} does not match expected shape {expected_shape}."
                )
            UserWarning(
                f"Data shape cropped to {data_array.shape} to fit within bounds."
            )

        # Write to array
        self.array.loc[aligned_coords] = data_array

    def _write_batch_items(
        self,
        batch_coords: Mapping[str, tuple[Sequence, np.ndarray]],
        data: Union[torch.Tensor, ArrayLike],
    ) -> None:
        """Write multiple data items by iterating through coordinate batches."""
        # Do for each item in the batch
        for i in range(data.shape[0]):
            # Extract center coordinates for this item
            item_coords = {axis: batch_coords[axis][i] for axis in self.axes}

            # Extract data for this item
            item_data = data[i]  # type: ignore

            # Write this single item using center coordinates
            self._write_single_item(item_coords, item_data)

    def __repr__(self) -> str:
        return f"ImageWriter({self.path}: {self.label_class} @ {list(self.scale.values())} {self.metadata['units']})"

    def __getitem__(
        self, coords: Mapping[str, float] | Mapping[str, tuple[Sequence, np.ndarray]]
    ) -> torch.Tensor:
        """
        Get the image data at the specified center coordinates.

        Args:
        ----
            coords (Mapping[str, float] | Mapping[str, tuple[Sequence, np.ndarray]]): The center coordinates or aligned coordinates.

        Returns:
        -------
            torch.Tensor: The image data at the specified center.
        """
        # Check if center or coords are provided
        if isinstance(list(coords.values())[0], int | float):
            center = coords
            aligned_coords = self.aligned_coords_from_center(center)  # type: ignore
        else:
            # If coords are provided, align them
            aligned_coords = self.align_coords(coords)  # type: ignore
        return torch.tensor(self.array.loc[aligned_coords].data).squeeze()
