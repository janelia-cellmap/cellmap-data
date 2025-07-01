import os
import numpy as np
import torch
import xarray
import tensorstore
import xarray_tensorstore as xt
from typing import Any, Mapping, Optional, Sequence
from upath import UPath
from pydantic_ome_ngff.v04.axis import Axis
from pydantic_ome_ngff.v04.transform import VectorScale, VectorTranslation
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
        label_class: str,
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
        self.label_class = label_class
        if isinstance(scale, Sequence):
            if len(axis_order) > len(scale):
                scale = [scale[0]] * (len(axis_order) - len(scale)) + list(scale)
            scale = {c: s for c, s in zip(axis_order, scale)}
        if isinstance(write_voxel_shape, Sequence):
            if len(axis_order) > len(write_voxel_shape):
                write_voxel_shape = [1] * (
                    len(axis_order) - len(write_voxel_shape)
                ) + list(write_voxel_shape)
            write_voxel_shape = {c: t for c, t in zip(axis_order, write_voxel_shape)}
        self.scale = scale
        self.bounding_box = bounding_box
        self.write_voxel_shape = write_voxel_shape
        self.write_world_shape = {
            c: write_voxel_shape[c] * scale[c] for c in axis_order
        }
        self.axes = axis_order[: len(write_voxel_shape)]
        self.scale_level = scale_level
        self.context = context
        self.overwrite = overwrite
        self.dtype = dtype
        self.fill_value = fill_value
        dims = [c for c in axis_order]
        self.metadata = {
            "offset": list(self.offset.values()),
            "axes": dims,
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
            from xarray_ome_ngff.v04.multiscale import coords_from_transforms
            from pydantic_ome_ngff.v04.axis import Axis
            from pydantic_ome_ngff.v04.transform import VectorScale, VectorTranslation

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
                c: self.bounding_box[c][1] - self.bounding_box[c][0] for c in self.axes
            }
            return self._world_shape

    @property
    def shape(self) -> Mapping[str, int]:
        try:
            return self._shape
        except AttributeError:
            self._shape = {
                c: int(np.ceil(self.world_shape[c] / self.scale[c])) for c in self.axes
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
            self._offset = {c: self.bounding_box[c][0] for c in self.axes}
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
        for c in self.axes:
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
        coords: Mapping[str, float] | Mapping[str, tuple[Sequence, np.ndarray]],
        data: torch.Tensor | np.typing.ArrayLike | float | int,  # type: ignore
    ) -> None:
        if isinstance(list(coords.values())[0], int | float):
            center = coords
            coords = self.aligned_coords_from_center(center)  # type: ignore
            if isinstance(data, torch.Tensor):
                data = data.cpu().numpy()
            data = np.array(data).squeeze().astype(self.dtype)
            try:
                self.array.loc[coords] = data
            except ValueError as e:
                slices = []
                for coord in coords.values():
                    if len(coord) > 1:
                        slices.append(slice(None, len(coord)))
                data = data[*slices]
                self.array.loc[coords] = data
        else:
            for i in range(len(coords[self.axes[0]])):
                if isinstance(data, (int, float)):
                    this_data = data
                else:
                    this_data = data[i]
                self[{c: coords[c][i] for c in self.axes}] = this_data

    def __repr__(self) -> str:
        return f"ImageWriter({self.path}: {self.label_class} @ {list(self.scale.values())} {self.metadata['units']})"

    def __getitem__(
        self, coords: Mapping[str, float] | Mapping[str, tuple[Sequence, np.ndarray]]
    ) -> torch.Tensor:
        """
        Get the image data at the specified center coordinates.
        Args:
            coords (Mapping[str, float] | Mapping[str, tuple[Sequence, np.ndarray]]): The center coordinates or aligned coordinates.
        Returns:
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
