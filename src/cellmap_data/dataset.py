# %%
import csv
from typing import Callable, Dict, Sequence, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
import tensorstore as tswift
from fibsem_tools.io.core import read, read_xarray
from .image import CellMapImage, EmptyImage


def split_gt_path(path: str) -> tuple[str, list[str]]:
    """Splits a path to groundtruth data into the main path string, and the classes supplied for it."""
    path_prefix, path_rem = path.split("[")
    classes, path_suffix = path_rem.split("]")
    classes = classes.split(",")
    path_string = path_prefix + "{label}" + path_suffix
    return path_string, classes


# %%
class CellMapDataset(Dataset):
    """This subclasses PyTorch Dataset to load CellMap data for training. It maintains the same API as the Dataset class. Importantly, it maintains information about and handles for the sources for raw and groundtruth data. This information includes the path to the data, the classes for segmentation, and the arrays to input to the network and use as targets for the network. The dataset constructs the sources for the raw and groundtruth data, and retrieves the data from the sources. The dataset also provides methods to get the number of pixels for each class in the ground truth data, normalized by the resolution. Additionally, random crops of the data can be generated for training, because the CellMapDataset maintains information about the extents of its source arrays. This object additionally combines images for different classes into a single output array, which is useful for training segmentation networks."""

    raw_path: str
    gt_path: str
    classes: Sequence[str]
    input_arrays: dict[str, dict[str, Sequence[int | float]]]
    target_arrays: dict[str, dict[str, Sequence[int | float]]]
    input_sources: dict[str, CellMapImage]
    target_sources: dict[str, dict[str, CellMapImage | EmptyImage]]
    to_target: Callable
    transforms: RandomApply | None
    has_data: bool
    _bounding_box: Optional[Dict[str, list[int]]]
    _bounding_box_shape: Optional[Dict[str, int]]
    _sampling_box: Optional[Dict[str, list[int]]]
    _sampling_box_shape: Optional[Dict[str, int]]
    _class_counts: Optional[Dict[str, Dict[str, int]]]
    _largest_voxel_sizes: Optional[Dict[str, int]]
    _len: Optional[int]
    _iter_coords: Optional[...]

    def __init__(
        self,
        raw_path: str,
        gt_path: str,
        classes: Sequence[str],
        input_arrays: dict[str, dict[str, Sequence[int | float]]],
        target_arrays: dict[str, dict[str, Sequence[int | float]]],
        to_target: Callable,
        transforms: Optional[Sequence[Callable]] = None,
    ):
        """Initializes the CellMapDataset class.

        Args:
            raw_path (str): The path to the raw data.
            gt_path (str): The path to the ground truth data.
            classes (Sequence[str]): A list of classes for segmentation training. Class order will be preserved in the output arrays. Classes not contained in the dataset will be filled in with zeros.
            input_arrays (dict[str, dict[str, Sequence[int | float]]]): A dictionary containing the arrays of the dataset to input to the network. The dictionary should have the following structure:
                {
                    "array_name": {
                        "shape": typle[int],
                        "scale": Sequence[float],
                    },
                    ...
                }
            target_arrays (dict[str, dict[str, Sequence[int | float]]]): A dictionary containing the arrays of the dataset to use as targets for the network. The dictionary should have the following structure:
                {
                    "array_name": {
                        "shape": typle[int],
                        "scale": Sequence[float],
                    },
                    ...
                }
            to_target (Callable): A function to convert the ground truth data to target arrays. The function should have the following structure:
                def to_target(gt: torch.Tensor, classes: Sequence[str]) -> dict[str, torch.Tensor]:
            transforms (Optional[Sequence[Callable]], optional): A sequence of transformations to apply to the data. Defaults to None.
        """
        self.raw_path = raw_path
        self.gt_paths = gt_path
        self.gt_path_str, self.classes_with_path = split_gt_path(gt_path)
        self.classes = classes
        self.input_arrays = input_arrays
        self.target_arrays = target_arrays
        self.to_target = to_target
        self.transforms = transforms
        self.construct()

    def __len__(self):
        """Returns the length of the dataset, determined by the number of coordinates that could be sampled as the center for a cube."""
        if not self.has_data:
            return 0
        if self._len is None:
            size = 1
            for _, (start, stop) in self.sampling_box.items():
                size *= stop - start
            size /= np.prod(list(self.largest_voxel_sizes.values()))
            self._len = int(size)
        return self._len

    def __getitem__(self, idx):
        """Returns a crop of the input and target data as PyTorch tensors, corresponding to the coordinate of the unwrapped index."""

        center = np.unravel_index(idx, list(self.sampling_box_shape.values()))
        outputs = {}
        for array_name in self.input_arrays.keys():
            outputs[array_name] = self.input_sources[array_name][center][
                None, None, ...
            ]
        for array_name in self.target_arrays.keys():
            class_arrays = []
            for label in self.classes:
                class_arrays.append(self.target_sources[array_name][label][center])
            outputs[array_name] = torch.stack(class_arrays)[None, ...]
        return outputs

    def __iter__(self):
        """Iterates over the dataset, covering each section of the bounding box. For instance, for calculating validation scores."""
        # TODO
        if self._iter_coords is None:
            self._iter_coords = ...
        yield self.__getitem__(self._iter_coords)

    def construct(self):
        """Constructs the input and target sources for the dataset."""
        self.input_sources = {}
        for array_name, array_info in self.input_arrays.items():
            self.input_sources[array_name] = CellMapImage(
                self.raw_path,
                "raw",
                array_info["scale"],
                array_info["shape"],  # type: ignore
            )
        self.target_sources = {}
        self.has_data = False
        for array_name, array_info in self.target_arrays.items():
            self.target_sources[array_name] = {}
            empty_store = torch.zeros(array_info["shape"])  # type: ignore
            for label in self.classes:
                if label in self.classes_with_path:
                    self.target_sources[array_name][label] = CellMapImage(
                        self.gt_path_str.format(label=label),
                        label,
                        array_info["scale"],
                        array_info["shape"],  # type: ignore
                    )
                    self.has_data = True
                else:
                    self.target_sources[array_name][label] = EmptyImage(
                        label, array_info["shape"], empty_store  # type: ignore
                    )

        self._bounding_box = None
        self._bounding_box_shape = None
        self._sampling_box = None
        self._sampling_box_shape = None
        self._class_counts = None
        self._largest_voxel_sizes = None
        self._len = None
        self._iter_coords = None

    @property
    def largest_voxel_sizes(self):
        """Returns the largest voxel size of the dataset."""
        if self._largest_voxel_size is None:
            largest_voxel_size = {c: 0 for c in "zyx"}
            for source in [self.input_sources.values(), self.target_sources.values()]:
                if source.scale is None:
                    continue
                for c, size in zip("zyx", source.scale):
                    largest_voxel_size[c] = max(largest_voxel_size[c], size)
            self._largest_voxel_size = largest_voxel_size

        return self._largest_voxel_size

    @property
    def bounding_box(self):
        """Returns the bounding box of the dataset."""
        if self._bounding_box is None:
            bounding_box = {c: [0, 2**32] for c in "zyx"}
            for source in [self.input_sources.values(), self.target_sources.values()]:
                if source.bounding_box is None:
                    continue
                for c, (start, stop) in source.bounding_box.items():
                    bounding_box[c][0] = max(bounding_box[c][0], start)
                    bounding_box[c][1] = min(bounding_box[c][1], stop)
            self._bounding_box = bounding_box
        return self._bounding_box

    @property
    def bounding_box_shape(self):
        """Returns the shape of the bounding box of the dataset in voxels of the largest voxel size."""
        if self._bounding_box_shape is None:
            bounding_box_shape = {c: 0 for c in "zyx"}
            for c, (start, stop) in self.bounding_box.items():
                size = stop - start
                size /= self.largest_voxel_sizes[c]
                bounding_box_shape[c] = int(size)
            self._bounding_box_shape = bounding_box_shape
        return self._bounding_box_shape

    @property
    def sampling_box(self):
        """Returns the sampling box of the dataset (i.e. where centers can be drawn from and still have full samples drawn from within the bounding box)."""
        if self._sampling_box is None:
            sampling_box = {c: [0, 2**32] for c in "zyx"}
            for source in [self.input_sources.values(), self.target_sources.values()]:
                if source.sampling_box is None:
                    continue
                for c, (start, stop) in source.sampling_box.items():
                    sampling_box[c][0] = max(sampling_box[c][0], start)
                    sampling_box[c][1] = min(sampling_box[c][1], stop)
            self._sampling_box = sampling_box
        return self._sampling_box

    @property
    def sampling_box_shape(self):
        """Returns the shape of the sampling box of the dataset in voxels of the largest voxel size."""
        if self._sampling_box_shape is None:
            sampling_box_shape = {c: 0 for c in "zyx"}
            for c, (start, stop) in self.sampling_box.items():
                size = stop - start
                size /= self.largest_voxel_sizes[c]
                sampling_box_shape[c] = int(size)
            self._sampling_box_shape = sampling_box_shape
        return self._sampling_box_shape

    @property
    def class_counts(self) -> Dict[str, Dict[str, int]]:
        """Returns the number of pixels for each class in the ground truth data, normalized by the resolution."""
        if self._class_counts is None:
            class_counts = {}
            for array_name, sources in self.target_sources.items():
                class_counts[array_name] = {}
                for label, source in sources.items():
                    class_counts[array_name][label] = source.class_counts
            self._class_counts = class_counts
        return self._class_counts


# Example input arrays:
# {'0_input': {'shape': (90, 90, 90), 'scale': (32, 32, 32)},
#  '1_input': {'shape': (114, 114, 114), 'scale': (8, 8, 8)}}

# Example output arrays:
# {'0_output': {'shape': (28, 28, 28), 'scale': (32, 32, 32)},
#  '1_output': {'shape': (32, 32, 32), 'scale': (8, 8, 8)}}
