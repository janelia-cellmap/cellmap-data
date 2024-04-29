# %%
import math
import os
from typing import Callable, Dict, Sequence, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info
import tensorstore
from fibsem_tools.io.core import read, read_xarray
from .image import CellMapImage, EmptyImage


def split_gt_path(path: str) -> tuple[str, list[str]]:
    """Splits a path to groundtruth data into the main path string, and the classes supplied for it."""
    try:
        path_prefix, path_rem = path.split("[")
        classes, path_suffix = path_rem.split("]")
        classes = classes.split(",")
        path_string = path_prefix + "{label}" + path_suffix
    except ValueError:
        path_string = path
        classes = [path.split(os.path.sep)[-1]]
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
    spatial_transforms: Optional[dict[str, any]]  # type: ignore
    raw_value_transforms: Optional[Callable]
    gt_value_transforms: Optional[Callable | Sequence[Callable] | dict[str, Callable]]
    has_data: bool
    is_train: bool
    axis_order: str
    context: Optional[tensorstore.Context]  # type: ignore
    _bounding_box: Optional[Dict[str, list[int]]]
    _bounding_box_shape: Optional[Dict[str, int]]
    _sampling_box: Optional[Dict[str, list[int]]]
    _sampling_box_shape: Optional[Dict[str, int]]
    _class_counts: Optional[Dict[str, Dict[str, int]]]
    _largest_voxel_sizes: Optional[Dict[str, int]]
    _len: Optional[int]
    _iter_coords: Optional[Sequence[float]]

    def __init__(
        self,
        raw_path: str,  # TODO: Switch "raw_path" to "input_path"
        gt_path: str,
        classes: Sequence[str],
        input_arrays: dict[str, dict[str, Sequence[int | float]]],
        target_arrays: dict[str, dict[str, Sequence[int | float]]],
        spatial_transforms: Optional[dict[str, any]] = None,  # type: ignore
        raw_value_transforms: Optional[Callable] = None,
        gt_value_transforms: Optional[
            Callable | Sequence[Callable] | dict[str, Callable]
        ] = None,
        is_train: bool = False,
        axis_order: str = "zyx",
        context: Optional[tensorstore.Context] = None,  # type: ignore
        rng: Optional[np.random.Generator] = None,
        force_has_data: bool = False,
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
            spatial_transforms (Optional[Sequence[dict[str, any]]], optional): A sequence of dictionaries containing the spatial transformations to apply to the data. The dictionary should have the following structure:
                {transform_name: {transform_args}}
                Defaults to None.
            raw_value_transforms (Optional[Callable], optional): A function to apply to the raw data. Defaults to None. Example is to normalize the raw data.
            gt_value_transforms (Optional[Callable | Sequence[Callable] | dict[str, Callable]], optional): A function to convert the ground truth data to target arrays. Defaults to None. Example is to convert the ground truth data to a signed distance transform. May be a single function, a list of functions, or a dictionary of functions for each class. In the case of a list of functions, it is assumed that the functions correspond to each class in the classes list in order.
            is_train (bool, optional): Whether the dataset is for training. Defaults to False.
            context (Optional[tensorstore.Context], optional): The context for the image data. Defaults to None.
            rng (Optional[np.random.Generator], optional): A random number generator. Defaults to None.
            force_has_data (bool, optional): Whether to force the dataset to report that it has data. Defaults to False.
        """
        self.raw_path = raw_path
        self.gt_paths = gt_path
        self.gt_path_str, self.classes_with_path = split_gt_path(gt_path)
        self.classes = classes
        self.input_arrays = input_arrays
        self.target_arrays = target_arrays
        self.spatial_transforms = spatial_transforms
        self.raw_value_transforms = raw_value_transforms
        self.gt_value_transforms = gt_value_transforms
        self.is_train = is_train
        self.axis_order = axis_order
        self.context = context
        self._rng = rng
        self.construct()
        self.force_has_data = force_has_data

    def __len__(self):
        """Returns the length of the dataset, determined by the number of coordinates that could be sampled as the center for a cube."""
        if not self.has_data and not self.force_has_data:
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
        center = {
            c: center[i] * self.largest_voxel_sizes[c] + self.sampling_box[c][0]
            for i, c in enumerate(self.axis_order)
        }
        self._current_center = center
        spatial_transforms = self.generate_spatial_transforms()
        outputs = {}
        for array_name in self.input_arrays.keys():
            self.input_sources[array_name].set_spatial_transforms(spatial_transforms)
            array = self.input_sources[array_name][center]
            if array.shape[0] != 1:
                outputs[array_name] = array[None, ...]
            else:
                outputs[array_name] = array
        # TODO: Allow for distribtion of array gathering to multiple threads
        for array_name in self.target_arrays.keys():
            class_arrays = []
            for label in self.classes:
                self.target_sources[array_name][label].set_spatial_transforms(
                    spatial_transforms
                )
                class_arrays.append(self.target_sources[array_name][label][center])
            outputs[array_name] = torch.stack(class_arrays)
        return outputs

    def __iter__(self):
        """Iterates over the dataset, covering each section of the bounding box. For instance, for calculating validation scores."""
        # TODO : determine if this is right
        raise NotImplementedError("Iterating over the dataset is not implemented.")
        # We need to iterate over idx's that are non-overlapping from within the sample_box
        idxs = ...

    def __repr__(self):
        """Returns a string representation of the dataset."""
        return f"CellMapDataset(\n\tRaw path: {self.raw_path}\n\tGT path(s): {self.gt_paths}\n\tClasses: {self.classes})"

    def to(self, device):
        """Sets the device for the dataset."""
        for source in list(self.input_sources.values()) + list(
            self.target_sources.values()
        ):
            if isinstance(source, dict):
                for source in source.values():
                    source.to(device)
            else:
                source.to(device)
        return self

    def construct(self):
        """Constructs the input and target sources for the dataset."""
        self._bounding_box = None
        self._bounding_box_shape = None
        self._sampling_box = None
        self._sampling_box_shape = None
        self._class_counts = None
        self._largest_voxel_sizes = None
        self._len = None
        self._iter_coords = None
        self._current_center = None
        self._current_spatial_transforms = None
        self.input_sources = {}
        for array_name, array_info in self.input_arrays.items():
            self.input_sources[array_name] = CellMapImage(
                self.raw_path,
                "raw",
                array_info["scale"],
                array_info["shape"],  # type: ignore
                value_transform=self.raw_value_transforms,
                context=self.context,
            )
        self.target_sources = {}
        self.has_data = False
        for array_name, array_info in self.target_arrays.items():
            self.target_sources[array_name] = {}
            empty_store = torch.zeros(array_info["shape"])  # type: ignore
            for i, label in enumerate(self.classes):  # type: ignore
                if label in self.classes_with_path:
                    if isinstance(self.gt_value_transforms, dict):
                        value_transform: Callable = self.gt_value_transforms[label]
                    elif isinstance(self.gt_value_transforms, list):
                        value_transform: Callable = self.gt_value_transforms[i]
                    else:
                        value_transform: Callable = self.gt_value_transforms  # type: ignore
                    self.target_sources[array_name][label] = CellMapImage(
                        self.gt_path_str.format(label=label),
                        label,
                        array_info["scale"],
                        array_info["shape"],  # type: ignore
                        value_transform=value_transform,
                        context=self.context,
                    )
                    self.has_data = (
                        self.has_data
                        or self.target_sources[array_name][label].class_counts != 0
                    )
                else:
                    self.target_sources[array_name][label] = EmptyImage(
                        label, array_info["shape"], empty_store  # type: ignore
                    )

    def generate_spatial_transforms(self) -> Optional[dict[str, any]]:
        """Generates spatial transforms for the dataset."""
        # TODO: use torch random number generator so accerlerators can synchronize across workers
        if self._rng is None:
            self._rng = np.random.default_rng()
        rng = self._rng

        if not self.is_train or self.spatial_transforms is None:
            return None
        spatial_transforms = {}
        for transform, params in self.spatial_transforms.items():
            if transform == "mirror":
                # input: "mirror": {"axes": {"x": 0.5, "y": 0.5, "z":0.1}}
                # output: {"mirror": ["x", "y"]}
                spatial_transforms[transform] = []
                for axis, prob in params["axes"].items():
                    if rng.random() < prob:
                        spatial_transforms[transform].append(axis)
            elif transform == "transpose":
                # only reorder axes specified in params
                # input: "transpose": {"axes": ["x", "z"]}
                # output: {"transpose": {"x": 2, "y": 1, "z": 0}}
                axes = {axis: i for i, axis in enumerate(self.axis_order)}
                shuffled_axes = rng.permutation(
                    [axes[a] for a in params["axes"]]
                )  # shuffle indices
                shuffled_axes = {
                    axis: shuffled_axes[i] for i, axis in enumerate(params["axes"])
                }  # reassign axes
                axes.update(shuffled_axes)
                spatial_transforms[transform] = axes
            else:
                raise ValueError(f"Unknown spatial transform: {transform}")
        self._current_spatial_transforms = spatial_transforms
        return spatial_transforms

    @property
    def largest_voxel_sizes(self):
        """Returns the largest voxel size of the dataset."""
        if self._largest_voxel_sizes is None:
            largest_voxel_size = {c: 0 for c in self.axis_order}
            for source in list(self.input_sources.values()) + list(
                self.target_sources.values()
            ):
                if isinstance(source, dict):
                    for label, source in source.items():
                        if source.scale is None:
                            continue
                        for c, size in source.scale.items():
                            largest_voxel_size[c] = max(largest_voxel_size[c], size)
                else:
                    if source.scale is None:
                        continue
                    for c, size in source.scale.items():
                        largest_voxel_size[c] = max(largest_voxel_size[c], size)
            self._largest_voxel_sizes = largest_voxel_size

        return self._largest_voxel_sizes

    @property
    def bounding_box(self):
        """Returns the bounding box of the dataset."""
        if self._bounding_box is None:
            bounding_box = {c: [0, 2**32] for c in self.axis_order}
            for source in list(self.input_sources.values()) + list(
                self.target_sources.values()
            ):
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
            bounding_box_shape = {c: 0 for c in self.axis_order}
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
            sampling_box = {c: [0, 2**32] for c in self.axis_order}
            for source in list(self.input_sources.values()) + list(
                self.target_sources.values()
            ):
                if isinstance(source, dict):
                    for label, source in source.items():
                        if source.sampling_box is None:
                            continue
                        for c, (start, stop) in source.sampling_box.items():
                            sampling_box[c][0] = max(sampling_box[c][0], start)
                            sampling_box[c][1] = min(sampling_box[c][1], stop)
                else:
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
            sampling_box_shape = {c: 0 for c in self.axis_order}
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
            class_counts = {"totals": {c: 0 for c in self.classes}}
            for array_name, sources in self.target_sources.items():
                class_counts[array_name] = {}
                for label, source in sources.items():
                    class_counts[array_name][label] = source.class_counts
                    class_counts["totals"][label] += source.class_counts
            self._class_counts = class_counts
        return self._class_counts


# Example input arrays:
# {'0_input': {'shape': (90, 90, 90), 'scale': (32, 32, 32)},
#  '1_input': {'shape': (114, 114, 114), 'scale': (8, 8, 8)}}

# Example output arrays:
# {'0_output': {'shape': (28, 28, 28), 'scale': (32, 32, 32)},
#  '1_output': {'shape': (32, 32, 32), 'scale': (8, 8, 8)}}
