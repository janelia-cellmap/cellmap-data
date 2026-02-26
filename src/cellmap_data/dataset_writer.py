# %%
import logging
from functools import cached_property
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import tensorstore
import torch
from torch.utils.data import Dataset, Subset
from upath import UPath

from .image import CellMapImage
from .image_writer import ImageWriter

logger = logging.getLogger(__name__)
if logger.level == logging.NOTSET:
    logger.setLevel(logging.INFO)

# Special keys that should not be written to disk
_METADATA_KEYS = {"idx"}


# %%
class CellMapDatasetWriter(Dataset):
    """
    Writes a dataset to disk in a format readable by CellMapDataset.

    This is useful for saving model predictions to disk.
    """

    def __init__(
        self,
        raw_path: str,
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
        ----
            raw_path: Full path to the raw data Zarr, excluding multiscale level.
            target_path: Full path to the ground truth Zarr, excluding class name.
            classes: The classes in the dataset.
            input_arrays: Input arrays for processing, with shape, scale, and
                          optional scale_level.
            target_arrays: Target arrays to write, with the same format as input_arrays.
            target_bounds: Bounding boxes for each target array in world units.
            raw_value_transforms: Value transforms for raw data.
            axis_order: Order of axes (e.g., "zyx").
            context: TensorStore context.
            rng: Random number generator.
            empty_value: Value for empty data.
            overwrite: Whether to overwrite existing data.
            device: Device for torch tensors ("cuda", "mps", or "cpu").
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
                pad_value=0,
                interpolation="linear",
            )
        self.target_array_writers: dict[str, dict[str, ImageWriter]] = {}
        for array_name, array_info in self.target_arrays.items():
            self.target_array_writers[array_name] = self.get_target_array_writer(
                array_name, array_info
            )
        self._device: str | torch.device = device if device is not None else "cpu"
        if device is not None:
            self.to(device, non_blocking=True)

    @cached_property
    def center(self) -> Mapping[str, float] | None:
        """Returns the center of the dataset in world units."""
        if self.bounding_box is None:
            return None
        return {
            c: start + (stop - start) / 2
            for c, (start, stop) in self.bounding_box.items()
        }

    @cached_property
    def smallest_voxel_sizes(self) -> Mapping[str, float]:
        """Returns the smallest voxel size of the dataset."""
        smallest_voxel_size = {c: np.inf for c in self.axis_order}
        all_sources = list(self.input_sources.values()) + list(
            self.target_array_writers.values()
        )
        for source in all_sources:
            if isinstance(source, dict):
                for sub_source in source.values():
                    if hasattr(sub_source, "scale") and sub_source.scale is not None:
                        for c, size in sub_source.scale.items():
                            smallest_voxel_size[c] = min(smallest_voxel_size[c], size)
            elif hasattr(source, "scale") and source.scale is not None:
                for c, size in source.scale.items():
                    smallest_voxel_size[c] = min(smallest_voxel_size[c], size)
        return smallest_voxel_size

    @cached_property
    def smallest_target_array(self) -> Mapping[str, float]:
        """Returns the smallest target array in world units."""
        smallest_target_array = {c: np.inf for c in self.axis_order}
        for writer in self.target_array_writers.values():
            for _, writer in writer.items():
                for c, size in writer.write_world_shape.items():
                    smallest_target_array[c] = min(smallest_target_array[c], size)
        return smallest_target_array

    @cached_property
    def bounding_box(self) -> Mapping[str, list[float]]:
        """Returns the bounding box inclusive of all the target images."""
        bounding_box = None
        for current_box in self.target_bounds.values():
            bounding_box = self._get_box_union(current_box, bounding_box)
        if bounding_box is None:
            logger.warning(
                "Bounding box is None. This may cause errors during sampling."
            )
            bounding_box = {c: [-np.inf, np.inf] for c in self.axis_order}
        return bounding_box

    @cached_property
    def bounding_box_shape(self) -> Mapping[str, int]:
        """Returns the shape of the bounding box of the dataset in voxels of the smallest voxel size requested."""
        return self._get_box_shape(self.bounding_box)

    @cached_property
    def sampling_box(self) -> Mapping[str, list[float]]:
        """Returns the sampling box of the dataset (i.e. where centers should be drawn from and to fully sample within the bounding box)."""
        sampling_box = None
        for array_name, array_info in self.target_arrays.items():
            padding = {
                c: np.ceil((shape * scale) / 2)
                for c, shape, scale in zip(
                    self.axis_order, array_info["shape"], array_info["scale"]
                )
            }
            this_box = {
                c: [bounds[0] + padding[c], bounds[1] - padding[c]]
                for c, bounds in self.target_bounds[array_name].items()
            }
            sampling_box = self._get_box_union(this_box, sampling_box)
        if sampling_box is None:
            logger.warning(
                "Sampling box is None. This may cause errors during sampling."
            )
            sampling_box = {c: [-np.inf, np.inf] for c in self.axis_order}
        return sampling_box

    @cached_property
    def sampling_box_shape(self) -> dict[str, int]:
        """Returns the shape of the sampling box."""
        shape = self._get_box_shape(self.sampling_box)
        for c, size in shape.items():
            if size <= 0:
                logger.debug(
                    "Sampling box for axis %s has size %d <= 0. "
                    "Setting to 1 and padding.",
                    c,
                    size,
                )
                shape[c] = 1
        return shape

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return int(np.prod(list(self.sampling_box_shape.values())))

    @cached_property
    def size(self) -> int:
        """Returns the number of samples in the dataset."""
        return int(
            np.prod([stop - start for start, stop in self.bounding_box.values()])
        )

    @cached_property
    def writer_indices(self) -> Sequence[int]:
        """Returns the indices of the dataset that will produce non-overlapping tiles for use in writer, based on the smallest requested target array."""
        return self.get_indices(self.smallest_target_array)

    @cached_property
    def blocks(self) -> Subset:
        """A subset of the validation datasets, tiling the validation datasets with non-overlapping blocks."""
        return Subset(self, self.writer_indices)

    def loader(
        self,
        batch_size: int = 1,
        num_workers: int = 0,
        **kwargs,
    ):
        """Returns a CellMapDataLoader for the dataset."""
        from .dataloader import CellMapDataLoader
        from .subdataset import CellMapSubset

        return CellMapDataLoader(
            CellMapSubset(self, self.writer_indices),
            batch_size=batch_size,
            num_workers=num_workers,
            device=self.device,
            is_train=False,
            sampler=None,
            **kwargs,
        ).loader

    @property
    def device(self) -> str | torch.device:
        """Returns the device for the dataset."""
        return self._device

    def get_center(self, idx: int) -> dict[str, float]:
        """
        Gets the center coordinates for a given index.

        Args:
        ----
            idx: The index to get the center for.

        Returns:
        -------
            A dictionary of center coordinates.
        """
        if idx < 0:
            idx = len(self) + idx
        try:
            center_indices = np.unravel_index(
                idx, [self.sampling_box_shape[c] for c in self.axis_order]
            )
        except ValueError:
            logger.error(
                "Index %s out of bounds for dataset of length %d", idx, len(self)
            )
            logger.warning("Returning closest index in bounds")
            center_indices = [self.sampling_box_shape[c] - 1 for c in self.axis_order]
        center = {
            c: float(
                center_indices[i] * self.smallest_voxel_sizes[c]
                + self.sampling_box[c][0]
            )
            for i, c in enumerate(self.axis_order)
        }
        return center

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Returns a crop of the input and target data as PyTorch tensors, corresponding to the coordinate of the unwrapped index."""
        self._current_idx = idx
        self._current_center = self.get_center(idx)
        outputs = {}
        for array_name in self.input_arrays.keys():
            array = self.input_sources[array_name][self._current_center]
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
        Writes values for the given arrays at the given index.

        Args:
        ----
            idx: The index or indices to write to.
            arrays: Dictionary of arrays to write to disk. Data can be a
                    single array with channels for classes, or a dictionary
                    of arrays per class.
        """
        if isinstance(idx, (torch.Tensor, np.ndarray, Sequence)):
            if isinstance(idx, torch.Tensor):
                idx = idx.cpu().numpy()
            for batch_idx, i in enumerate(idx):
                # Extract the data for this specific item in the batch
                item_arrays = {}
                for array_name, array in arrays.items():
                    # Skip special metadata keys
                    if array_name in _METADATA_KEYS:
                        continue
                    if isinstance(array, (int, float)):
                        # Scalar values are the same for all items
                        item_arrays[array_name] = array
                    elif isinstance(array, dict):
                        # Dictionary of arrays - extract batch item from each
                        item_arrays[array_name] = {
                            label: label_array[batch_idx]
                            for label, label_array in array.items()
                        }
                    else:
                        # Regular array - extract the batch item
                        item_arrays[array_name] = array[batch_idx]
                self.__setitem__(i, item_arrays)
            return

        self._current_idx = idx
        self._current_center = self.get_center(self._current_idx)
        for array_name, array in arrays.items():
            # Skip special metadata keys
            if array_name in _METADATA_KEYS:
                continue
            if isinstance(array, (int, float)):
                for label in self.classes:
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
                    ] = array[c, ...]

    def __repr__(self) -> str:
        """Returns a string representation of the dataset."""
        return (
            f"CellMapDatasetWriter(\n\tRaw path: {self.raw_path}\n\t"
            f"Output path(s): {self.target_path}\n\tClasses: {self.classes})"
        )

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
        """Returns an ImageWriter for a specific target image."""
        scale = array_info["scale"]
        if not isinstance(scale, (Mapping, Sequence)):
            raise TypeError(f"Scale must be a Mapping or Sequence, not {type(scale)}")
        shape = array_info["shape"]
        if not isinstance(shape, (Mapping, Sequence)):
            raise TypeError(f"Shape must be a Mapping or Sequence, not {type(shape)}")
        if "n_channels" in array_info:
            shape = [array_info["n_channels"]] + list(shape)
            if "c" not in self.axis_order:
                self.axis_order = "c" + self.axis_order
        scale_level = array_info.get("scale_level", 0)
        if not isinstance(scale_level, int):
            raise TypeError(f"Scale level must be an int, not {type(scale_level)}")

        return ImageWriter(
            path=str(UPath(self.target_path) / label),
            target_class=label,
            scale=scale,  # type: ignore
            bounding_box=self.target_bounds[array_name],
            write_voxel_shape=shape,  # type: ignore
            scale_level=scale_level,
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
            box_shape[c] = int(np.ceil(size))
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
                if stop <= start:
                    raise ValueError(f"Invalid box: start={start}, stop={stop}")
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
                if stop <= start:
                    raise ValueError(f"Invalid box: start={start}, stop={stop}")
                current_box[c][0] = max(current_box[c][0], start)
                current_box[c][1] = min(current_box[c][1], stop)
        return current_box

    def verify(self) -> bool:
        """Verifies that the dataset is valid to draw samples from."""
        # TODO: make more robust
        try:
            return len(self) > 0
        except Exception as e:
            logger.warning("Dataset verification failed: %s", e)
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

            if indices_dict[c][-1] != self.sampling_box_shape[c] - 1:
                indices_dict[c] = np.append(
                    indices_dict[c], self.sampling_box_shape[c] - 1
                )

        indices = []
        shape_values = list(self.sampling_box_shape.values())
        for i in np.ndindex(*[len(indices_dict[c]) for c in self.axis_order]):
            index = [indices_dict[c][j] for c, j in zip(self.axis_order, i)]
            index = np.ravel_multi_index(index, shape_values)
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
                source.to(str(device), non_blocking=non_blocking)
        return self

    def set_raw_value_transforms(self, transforms: Callable) -> None:
        """Sets the raw value transforms for the dataset."""
        self.raw_value_transforms = transforms
        for source in self.input_sources.values():
            source.value_transform = transforms

    def get_weighted_sampler(
        self, batch_size: int = 1, rng: Optional[torch.Generator] = None
    ):
        raise NotImplementedError(
            "Weighted sampling is not typically used for writer datasets."
        )

    def get_subset_random_sampler(
        self, num_samples: int, rng: Optional[torch.Generator] = None
    ):
        raise NotImplementedError(
            "Random sampling is not typically used for writer datasets."
        )


# %%
