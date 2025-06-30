# %%
import os
from typing import Callable, Mapping, Sequence, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, DataLoader
import tensorstore
from upath import UPath

from .image import CellMapImage
from .image_writer import ImageWriter
from .utils import split_target_path
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# %%
class CellMapDatasetWriter(Dataset):
    """
    This class is used to write a dataset to disk in a format that can be read by the CellMapDataset class. It is useful, for instance, for writing predictions from a model to disk.
    """

    def __init__(
        self,
        raw_path: str,  # TODO: Switch "raw_path" to "input_path"
        target_path: str,
        classes: Sequence[str],
        input_arrays: Mapping[str, Mapping[str, Sequence[int | float]]],
        target_arrays: Mapping[str, Mapping[str, Sequence[int | float]]],
        target_bounds: Mapping[str, Mapping[str, list[float]]],
        raw_value_transforms: Optional[Callable] = None,
        axis_order: str = "zyx",
        context: Optional[tensorstore.Context] = None,  # type: ignore
        rng: Optional[torch.Generator] = None,
        empty_value: float | int = 0,
        overwrite: bool = False,
        device: Optional[str | torch.device] = None,
    ) -> None:
        """Initializes the CellMapDatasetWriter.

        Args:

            raw_path (str): The full path to the raw data zarr, excluding the mulstiscale level.
            target_path (str): The full path to the ground truth data zarr, excluding the mulstiscale level and the class name.
            classes (Sequence[str]): The classes in the dataset.
            input_arrays (Mapping[str, Mapping[str, Sequence[int | float]]]): The input arrays to return for processing. The dictionary should have the following structure::

                {
                    "array_name": {
                        "shape": tuple[int],
                        "scale": Sequence[float],

                        and optionally:
                        "scale_level": int,
                    },
                    ...
                }

            where 'array_name' is the name of the array, 'shape' is the shape of the array in voxels, and 'scale' is the scale of the array in world units. The 'scale_level' is the multiscale level to use for the array, otherwise set to 0 if not supplied.
            target_arrays (Mapping[str, Mapping[str, Sequence[int | float]]]): The target arrays to write to disk, with format matching that for input_arrays.
            target_bounds (Mapping[str, Mapping[str, list[float]]]): The bounding boxes for each target array, in world units. Example: {"array_1": {"x": [12.0, 102.0], "y": [12.0, 102.0], "z": [12.0, 102.0]}}.
            raw_value_transforms (Optional[Callable]): The value transforms to apply to the raw data.
            axis_order (str): The order of the axes in the data.
            context (Optional[tensorstore.Context]): The context to use for the tensorstore.
            rng (Optional[torch.Generator]): The random number generator to use.
            empty_value (float | int): The value to use for empty data in an array.
            overwrite (bool): Whether to overwrite existing data.
            device (Optional[str | torch.device]): The device to use for the dataset. If None, will default to "cuda" if available, then "mps", otherwise "cpu".
        """
        self.raw_path = raw_path
        self.target_path = target_path
        self.classes = classes
        self.input_arrays = input_arrays
        self.target_arrays = target_arrays
        self.target_bounds = target_bounds
        self.raw_value_transforms = raw_value_transforms
        self.axis_order = axis_order
        self.context = context
        self._rng = rng
        self.empty_value = empty_value
        self.overwrite = overwrite
        self._current_center = None
        self._current_idx = None
        self.input_sources: dict[str, CellMapImage] = {}
        for array_name, array_info in self.input_arrays.items():
            self.input_sources[array_name] = CellMapImage(
                self.raw_path,
                "raw",
                array_info["scale"],
                array_info["shape"],  # type: ignore
                value_transform=self.raw_value_transforms,
                context=self.context,
                pad=True,
                pad_value=0,  # inputs to the network should be padded with 0
                interpolation="linear",
            )
        self.target_array_writers: dict[str, dict[str, ImageWriter]] = {}
        for array_name, array_info in self.target_arrays.items():
            self.target_array_writers[array_name] = self.get_target_array_writer(
                array_name, array_info
            )
        if device is not None:
            self._device = device
        self.to(device, non_blocking=True)

    @property
    def center(self) -> Mapping[str, float] | None:
        """Returns the center of the dataset in world units."""
        try:
            return self._center
        except AttributeError:
            if self.bounding_box is None:
                self._center = None
            else:
                center = {}
                for c, (start, stop) in self.bounding_box.items():
                    center[c] = start + (stop - start) / 2
                self._center = center
            return self._center

    @property
    def smallest_voxel_sizes(self) -> Mapping[str, float]:
        """Returns the smallest voxel size of the dataset."""
        try:
            return self._smallest_voxel_sizes
        except AttributeError:
            smallest_voxel_size = {c: np.inf for c in self.axis_order}
            for source in list(self.input_sources.values()) + list(
                self.target_array_writers.values()
            ):
                if isinstance(source, dict):
                    for _, source in source.items():
                        if not hasattr(source, "scale") or source.scale is None:  # type: ignore
                            continue
                        for c, size in source.scale.items():  # type: ignore
                            smallest_voxel_size[c] = min(smallest_voxel_size[c], size)
                else:
                    if not hasattr(source, "scale") or source.scale is None:
                        continue
                    for c, size in source.scale.items():
                        smallest_voxel_size[c] = min(smallest_voxel_size[c], size)
            self._smallest_voxel_sizes = smallest_voxel_size

            return self._smallest_voxel_sizes

    @property
    def smallest_target_array(self) -> Mapping[str, float]:
        """Returns the smallest target array in world units."""
        try:
            return self._smallest_target_array
        except AttributeError:
            smallest_target_array = {c: np.inf for c in self.axis_order}
            for writer in self.target_array_writers.values():
                for _, writer in writer.items():
                    for c, size in writer.write_world_shape.items():
                        smallest_target_array[c] = min(smallest_target_array[c], size)
            self._smallest_target_array = smallest_target_array
            return self._smallest_target_array

    @property
    def bounding_box(self) -> Mapping[str, list[float]]:
        """Returns the bounding box inclusive of all the target images."""
        try:
            return self._bounding_box
        except AttributeError:
            bounding_box = None
            for current_box in self.target_bounds.values():
                bounding_box = self._get_box_union(current_box, bounding_box)
            if bounding_box is None:
                logger.warning(
                    "Bounding box is None. This may result in errors when trying to sample from the dataset."
                )
                bounding_box = {c: [-np.inf, np.inf] for c in self.axis_order}
            self._bounding_box = bounding_box
            return self._bounding_box

    @property
    def bounding_box_shape(self) -> Mapping[str, int]:
        """Returns the shape of the bounding box of the dataset in voxels of the smallest voxel size requested."""
        try:
            return self._bounding_box_shape
        except AttributeError:
            self._bounding_box_shape = self._get_box_shape(self.bounding_box)
            return self._bounding_box_shape

    @property
    def sampling_box(self) -> Mapping[str, list[float]]:
        """Returns the sampling box of the dataset (i.e. where centers should be drawn from and to fully sample within the bounding box)."""
        try:
            return self._sampling_box
        except AttributeError:
            sampling_box = None
            for array_name, array_info in self.target_arrays.items():
                padding = {c: np.ceil((shape * scale) / 2) for c, shape, scale in zip(self.axis_order, array_info["shape"], array_info["scale"])}  # type: ignore
                this_box = {
                    c: [bounds[0] + padding[c], bounds[1] - padding[c]]
                    for c, bounds in self.target_bounds[array_name].items()
                }
                sampling_box = self._get_box_union(this_box, sampling_box)
            if sampling_box is None:
                logger.warning(
                    "Sampling box is None. This may result in errors when trying to sample from the dataset."
                )
                sampling_box = {c: [-np.inf, np.inf] for c in self.axis_order}
            self._sampling_box = sampling_box
            return self._sampling_box

    @property
    def sampling_box_shape(self) -> dict[str, int]:
        """Returns the shape of the sampling box of the dataset in voxels of the smallest voxel size requested."""
        try:
            return self._sampling_box_shape
        except AttributeError:
            self._sampling_box_shape = self._get_box_shape(self.sampling_box)
            for c, size in self._sampling_box_shape.items():
                if size <= 0:
                    logger.debug(
                        f"Sampling box shape is <= 0 for axis {c} with size {size}. Setting to 1 and padding"
                    )
                    self._sampling_box_shape[c] = 1
            return self._sampling_box_shape

    @property
    def size(self) -> int:
        """Returns the size of the dataset in voxels of the smallest voxel size requested."""
        try:
            return self._size
        except AttributeError:
            self._size = np.prod(
                [stop - start for start, stop in self.bounding_box.values()]
            ).astype(int)
            return self._size

    @property
    def writer_indices(self) -> Sequence[int]:
        """Returns the indices of the dataset that will produce non-overlapping tiles for use in writer, based on the smallest requested target array."""
        try:
            return self._writer_indices
        except AttributeError:
            self._writer_indices = self.get_indices(self.smallest_target_array)
            return self._writer_indices

    @property
    def blocks(self) -> Subset:
        """A subset of the validation datasets, tiling the validation datasets with non-overlapping blocks."""
        try:
            return self._blocks
        except AttributeError:
            self._blocks = Subset(
                self,
                self.writer_indices,
            )
            return self._blocks

    def loader(
        self,
        batch_size: int = 1,
        num_workers: int = 0,
        **kwargs,
    ):
        """Returns a DataLoader for the dataset."""
        return DataLoader(
            self.blocks,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            **kwargs,
        )

    def collate_fn(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        """Combine a list of dictionaries from different sources into a single dictionary for output."""
        outputs = {}
        for b in batch:
            for key, value in b.items():
                if key not in outputs:
                    outputs[key] = []
                outputs[key].append(value)
        for key, value in outputs.items():
            outputs[key] = torch.stack(value)
        return outputs

    @property
    def device(self) -> torch.device:
        """Returns the device for the dataset."""
        try:
            return self._device
        except AttributeError:
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self._device = torch.device("mps")
            else:
                self._device = torch.device("cpu")
            self.to(self._device, non_blocking=True)
            return self._device

    def __len__(self) -> int:
        """Returns the length of the dataset, determined by the number of coordinates that could be sampled as the center for an array request."""
        try:
            return self._len
        except AttributeError:
            size = np.prod([self.sampling_box_shape[c] for c in self.axis_order])
            self._len = int(size)
            return self._len

    def get_center(self, idx: int) -> dict[str, float]:
        idx = np.array(idx)
        idx[idx < 0] = len(self) + idx[idx < 0]
        try:
            center = np.unravel_index(
                idx, [self.sampling_box_shape[c] for c in self.axis_order]
            )
        except ValueError:
            logger.error(
                f"Index {idx} out of bounds for dataset {self} of length {len(self)}"
            )
            logger.warning(f"Returning closest index in bounds")
            # TODO: This is a hacky temprorary fix. Need to figure out why this is happening
            center = [self.sampling_box_shape[c] - 1 for c in self.axis_order]
        center = {
            c: center[i] * self.smallest_voxel_sizes[c] + self.sampling_box[c][0]
            for i, c in enumerate(self.axis_order)
        }
        return center

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Returns a crop of the input and target data as PyTorch tensors, corresponding to the coordinate of the unwrapped index."""

        self._current_idx = idx
        self._current_center = self.get_center(idx)
        outputs = {}
        for array_name in self.input_arrays.keys():
            array = self.input_sources[array_name][self._current_center]  # type: ignore
            # TODO: Assumes 1 channel (i.e. grayscale)
            if array.shape[0] != 1:
                outputs[array_name] = array[None, ...]
            else:
                outputs[array_name] = array
        outputs["idx"] = torch.tensor(idx)

        return outputs

    def __setitem__(
        self,
        idx: int | torch.Tensor | np.ndarray | Sequence[int],
        arrays: dict[str, torch.Tensor | np.ndarray],
    ) -> None:
        """
        Writes the values for the given arrays at the given index.

        Args:
            idx (int | torch.Tensor | np.ndarray | Sequence[int]): The index or indices to write the arrays to.
            arrays (dict[str, torch.Tensor | np.ndarray]): The arrays to write to disk, with data either split by label class into a dictionary, or divided by class along the channel dimension of an array/tensor. The dictionary should have the following structure::

                {
                    "array_name": torch.Tensor | np.ndarray | dict[str, torch.Tensor | np.ndarray],
                    ...
                }
        """
        self._current_idx = idx
        self._current_center = self.get_center(self._current_idx)
        for array_name, array in arrays.items():
            if isinstance(array, int) or isinstance(array, float):
                for c, label in enumerate(self.classes):
                    self.target_array_writers[array_name][label][
                        self._current_center
                    ] = array
            elif isinstance(array, dict):
                for label, label_array in array.items():
                    self.target_array_writers[array_name][label][
                        self._current_center
                    ] = label_array
            else:
                for c, label in enumerate(self.classes):
                    self.target_array_writers[array_name][label][
                        self._current_center
                    ] = array[:, c, ...]

    def __repr__(self) -> str:
        """Returns a string representation of the dataset."""
        return f"CellMapDatasetWriter(\n\tRaw path: {self.raw_path}\n\tOutput path(s): {self.target_path}\n\tClasses: {self.classes})"

    def get_target_array_writer(
        self, array_name: str, array_info: Mapping[str, Sequence[int | float]]
    ) -> dict[str, ImageWriter]:
        """Returns a dictionary of ImageWriter for the target images (per class) for a given array."""
        target_image_writers = {}
        for label in self.classes:
            target_image_writers[label] = self.get_image_writer(
                array_name, label, array_info
            )

        return target_image_writers

    def get_image_writer(
        self,
        array_name: str,
        label: str,
        array_info: Mapping[str, Sequence[int | float] | int],
    ) -> ImageWriter:
        return ImageWriter(
            path=str(UPath(self.target_path) / label),
            label_class=label,
            scale=array_info["scale"],  # type: ignore
            bounding_box=self.target_bounds[array_name],
            write_voxel_shape=array_info["shape"],  # type: ignore
            scale_level=array_info.get("scale_level", 0),  # type: ignore
            axis_order=self.axis_order,
            context=self.context,
            fill_value=self.empty_value,
            overwrite=self.overwrite,
        )

    def _get_box_shape(self, source_box: Mapping[str, list[float]]) -> dict[str, int]:
        """Returns the shape of the box in voxels of the smallest voxel size requested."""
        box_shape = {}
        for c, (start, stop) in source_box.items():
            size = stop - start
            size /= self.smallest_voxel_sizes[c]
            box_shape[c] = int(np.floor(size))
        return box_shape

    def _get_box_union(
        self,
        source_box: Mapping[str, list[float]] | None,
        current_box: Mapping[str, list[float]] | None,
    ) -> Mapping[str, list[float]] | None:
        """Returns the union of the source and current boxes."""
        if source_box is not None:
            if current_box is None:
                return source_box
            for c, (start, stop) in source_box.items():
                assert stop > start, f"Invalid box: {start} to {stop}"
                current_box[c][0] = min(current_box[c][0], start)
                current_box[c][1] = max(current_box[c][1], stop)
        return current_box

    def _get_box_intersection(
        self,
        source_box: Mapping[str, list[float]] | None,
        current_box: Mapping[str, list[float]] | None,
    ) -> Mapping[str, list[float]] | None:
        """Returns the intersection of the source and current boxes."""
        if source_box is not None:
            if current_box is None:
                return source_box
            for c, (start, stop) in source_box.items():
                assert stop > start, f"Invalid box: {start} to {stop}"
                current_box[c][0] = max(current_box[c][0], start)
                current_box[c][1] = min(current_box[c][1], stop)
        return current_box

    def verify(self) -> bool:
        """Verifies that the dataset is valid to draw samples from."""
        # TODO: make more robust
        try:
            return len(self) > 0
        except Exception as e:
            logger.warning(f"Error: {e}")
            return False

    def get_indices(self, chunk_size: Mapping[str, float]) -> Sequence[int]:
        """Returns the indices of the dataset that will tile the dataset according to the chunk_size (supplied in world units)."""
        # TODO: ADD TEST

        # Convert the target chunk size in world units to voxel units
        chunk_size = {
            c: int(size // self.smallest_voxel_sizes[c])
            for c, size in chunk_size.items()
        }

        indices_dict = {}
        for c, size in chunk_size.items():
            indices_dict[c] = np.arange(0, self.sampling_box_shape[c], size, dtype=int)

            # Make sure the last index is included
            if indices_dict[c][-1] != self.sampling_box_shape[c] - 1:
                indices_dict[c] = np.append(
                    indices_dict[c], self.sampling_box_shape[c] - 1
                )

        indices = []
        # Generate linear indices by unraveling all combinations of axes indices
        for i in np.ndindex(*[len(indices_dict[c]) for c in self.axis_order]):
            index = [indices_dict[c][j] for c, j in zip(self.axis_order, i)]
            index = np.ravel_multi_index(index, list(self.sampling_box_shape.values()))
            indices.append(index)
        return indices

    def to(
        self, device: str | torch.device, non_blocking: bool = True
    ) -> "CellMapDatasetWriter":
        """Sets the device for the dataset."""
        if device is None:
            device = self.device
        self._device = torch.device(device)
        for source in self.input_sources.values():
            if isinstance(source, dict):
                for source in source.values():
                    if not hasattr(source, "to"):
                        continue
                    source.to(device, non_blocking=non_blocking)
            else:
                if not hasattr(source, "to"):
                    continue
                source.to(device, non_blocking=non_blocking)
        return self

    def set_raw_value_transforms(self, transforms: Callable) -> None:
        """Sets the raw value transforms for the dataset."""
        self.raw_value_transforms = transforms
        for source in self.input_sources.values():
            source.value_transform = transforms


# %%
