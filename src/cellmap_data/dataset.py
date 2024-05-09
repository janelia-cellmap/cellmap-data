# %%
import os
from typing import Any, Callable, Dict, Sequence, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
import tensorstore
from .image import CellMapImage, EmptyImage

import logging

logger = logging.getLogger(__name__)


def split_target_path(path: str) -> tuple[str, list[str]]:
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
    target_path: str
    classes: Sequence[str]
    input_arrays: dict[str, dict[str, Sequence[int | float]]]
    target_arrays: dict[str, dict[str, Sequence[int | float]]]
    input_sources: dict[str, CellMapImage]
    target_sources: dict[str, dict[str, CellMapImage | EmptyImage]]
    spatial_transforms: Optional[dict[str, any]]  # type: ignore
    raw_value_transforms: Optional[Callable]
    target_value_transforms: Optional[
        Callable | Sequence[Callable] | dict[str, Callable]
    ]
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
        target_path: str,
        classes: Sequence[str],
        input_arrays: dict[str, dict[str, Sequence[int | float]]],
        target_arrays: dict[str, dict[str, Sequence[int | float]]],
        spatial_transforms: Optional[dict[str, any]] = None,  # type: ignore
        raw_value_transforms: Optional[Callable] = None,
        target_value_transforms: Optional[
            Callable | Sequence[Callable] | dict[str, Callable]
        ] = None,
        is_train: bool = False,
        axis_order: str = "zyx",
        context: Optional[tensorstore.Context] = None,  # type: ignore
        rng: Optional[torch.Generator] = None,
        force_has_data: bool = False,
    ):
        """Initializes the CellMapDataset class.

        Args:
            raw_path (str): The path to the raw data.
            target_path (str): The path to the ground truth data.
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
            target_value_transforms (Optional[Callable | Sequence[Callable] | dict[str, Callable]], optional): A function to convert the ground truth data to target arrays. Defaults to None. Example is to convert the ground truth data to a signed distance transform. May be a single function, a list of functions, or a dictionary of functions for each class. In the case of a list of functions, it is assumed that the functions correspond to each class in the classes list in order.
            is_train (bool, optional): Whether the dataset is for training. Defaults to False.
            context (Optional[tensorstore.Context], optional): The context for the image data. Defaults to None.
            rng (Optional[torch.Generator], optional): A random number generator. Defaults to None.
            force_has_data (bool, optional): Whether to force the dataset to report that it has data. Defaults to False.
        """
        self.raw_path = raw_path
        self.target_paths = target_path
        self.target_path_str, self.classes_with_path = split_target_path(target_path)
        self.classes = classes
        self.input_arrays = input_arrays
        self.target_arrays = target_arrays
        self.spatial_transforms = spatial_transforms
        self.raw_value_transforms = raw_value_transforms
        self.target_value_transforms = target_value_transforms
        self.is_train = is_train
        self.axis_order = axis_order
        self.context = context
        self.rng = rng
        self.force_has_data = force_has_data
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
            # TODO: This approach to empty store doesn't work for multiple classes, at least with cross entropy loss in cellmap-train
            empty_store = torch.zeros(
                array_info["shape"]
            )  # * torch.nan  # type: ignore
            for i, label in enumerate(self.classes):  # type: ignore
                if label in self.classes_with_path:
                    if isinstance(self.target_value_transforms, dict):
                        value_transform: Callable = self.target_value_transforms[label]
                    elif isinstance(self.target_value_transforms, list):
                        value_transform: Callable = self.target_value_transforms[i]
                    else:
                        value_transform: Callable = self.target_value_transforms  # type: ignore
                    self.target_sources[array_name][label] = CellMapImage(
                        self.target_path_str.format(label=label),
                        label,
                        array_info["scale"],
                        array_info["shape"],  # type: ignore
                        value_transform=value_transform,
                        context=self.context,
                    )
                    if not self.has_data:
                        self.has_data = (
                            self.has_data
                            or self.target_sources[array_name][label].class_counts != 0
                        )
                else:
                    self.target_sources[array_name][label] = EmptyImage(
                        label, array_info["shape"], empty_store  # type: ignore
                    )

    def __len__(self):
        """Returns the length of the dataset, determined by the number of coordinates that could be sampled as the center for a cube."""
        if not self.has_data and not self.force_has_data:
            return 0
        if self._len is None:
            size = np.prod([self.sampling_box_shape[c] for c in self.axis_order])
            self._len = int(size)
        return self._len

    def __getitem__(self, idx):
        """Returns a crop of the input and target data as PyTorch tensors, corresponding to the coordinate of the unwrapped index."""
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
            center = [self.sampling_box_shape[c] for c in self.axis_order]
        center = {
            c: center[i] * self.largest_voxel_sizes[c] + self.sampling_box[c][0]
            for i, c in enumerate(self.axis_order)
        }
        self._current_idx = idx
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
                array = self.target_sources[array_name][label][center]
                class_arrays.append(array)
            outputs[array_name] = torch.stack(class_arrays)
        return outputs

    def __repr__(self):
        """Returns a string representation of the dataset."""
        return f"CellMapDataset(\n\tRaw path: {self.raw_path}\n\tGT path(s): {self.target_paths}\n\tClasses: {self.classes})"

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
                if isinstance(source, dict):
                    for label, source in source.items():
                        bounding_box = self._get_box(source.bounding_box, bounding_box)
                else:
                    bounding_box = self._get_box(source.bounding_box, bounding_box)
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
                        sampling_box = self._get_box(source.sampling_box, sampling_box)
                else:
                    sampling_box = self._get_box(source.sampling_box, sampling_box)
            self._sampling_box = sampling_box
        return self._sampling_box

    @property
    def sampling_box_shape(self):
        """Returns the shape of the sampling box of the dataset in voxels of the largest voxel size."""
        if self._sampling_box_shape is None:
            sampling_box_shape = {}
            for c, (start, stop) in self.sampling_box.items():
                size = stop - start
                size /= self.largest_voxel_sizes[c]
                sampling_box_shape[c] = int(np.floor(size))
            self._sampling_box_shape = sampling_box_shape
        return self._sampling_box_shape

    @property
    def class_weights(self):
        """
        Returns the class weights for the multi-dataset based on the number of samples in each class.
        """
        if len(self.classes) > 1:
            class_counts = {c: 0 for c in self.classes}
            class_count_sum = 0
            for c in self.classes:
                class_counts[c] += self.class_counts["totals"][c]
                class_count_sum += self.class_counts["totals"][c]

            class_weights = {
                c: (
                    1 - (class_counts[c] / class_count_sum)
                    if class_counts[c] != class_count_sum
                    else 0.1
                )
                for c in self.classes
            }
        else:
            class_weights = {self.classes[0]: 0.1}  # less than 1 to avoid overflow
        return class_weights

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

    @property
    def validation_indices(self) -> Sequence[int]:
        """Returns the indices of the dataset that will tile the dataset for validation."""
        chunk_size = {}
        for c, size in self.bounding_box_shape.items():
            chunk_size[c] = np.ceil(size - self.sampling_box_shape[c]).astype(int)
        return self.get_indices(chunk_size)

    def _get_box(
        self, source_box: dict[str, list[int]], current_box: dict[str, list[int]]
    ) -> dict[str, list[int]]:
        if source_box is not None:
            for c, (start, stop) in source_box.items():
                assert stop > start
                current_box[c][0] = max(current_box[c][0], start)
                current_box[c][1] = min(current_box[c][1], stop)
        return current_box

    def verify(self):
        """Verifies that the dataset is valid."""
        # TODO: make more robust
        try:
            length = len(self)
            return True
        except Exception as e:
            # print(e)
            return False

    def get_indices(self, chunk_size: dict[str, int]) -> Sequence[int]:
        # TODO: ADD TEST
        """Returns the indices of the dataset that will tile the dataset according to the chunk_size."""
        # Get padding per axis
        indices_dict = {}
        for c, size in chunk_size.items():
            indices_dict[c] = np.arange(0, self.sampling_box_shape[c], size, dtype=int)

        indices = []
        # Generate linear indices by unraveling all combinations of axes indices
        for i in np.ndindex(*[len(indices_dict[c]) for c in self.axis_order]):
            index = [indices_dict[c][j] for c, j in zip(self.axis_order, i)]
            index = np.ravel_multi_index(index, list(self.sampling_box_shape.values()))
            indices.append(index)

        return indices

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

    def generate_spatial_transforms(self) -> Optional[dict[str, Any]]:
        """Generates spatial transforms for the dataset."""
        # *TODO: use torch random number generator so accerlerators can synchronize across workers

        if not self.is_train or self.spatial_transforms is None:
            return None
        spatial_transforms = {}
        for transform, params in self.spatial_transforms.items():
            if transform == "mirror":
                # input: "mirror": {"axes": {"x": 0.5, "y": 0.5, "z":0.1}}
                # output: {"mirror": ["x", "y"]}
                spatial_transforms[transform] = []
                for axis, prob in params["axes"].items():
                    if torch.rand(1, generator=self.rng).item() < prob:
                        spatial_transforms[transform].append(axis)
            elif transform == "transpose":
                # only reorder axes specified in params
                # input: "transpose": {"axes": ["x", "z"]}
                # params["axes"] = ["x", "z"]
                # axes = {"x": 0, "y": 1, "z": 2}
                axes = {axis: i for i, axis in enumerate(self.axis_order)}
                # shuffled_axes = [0, 2]
                shuffled_axes = [axes[a] for a in params["axes"]]
                # shuffled_axes = [2, 0]
                shuffled_axes = shuffled_axes[
                    torch.randperm(len(shuffled_axes), generator=self.rng)
                ]  # shuffle indices
                # shuffled_axes = {"x": 2, "z": 0}
                shuffled_axes = {
                    axis: shuffled_axes[i] for i, axis in enumerate(params["axes"])
                }  # reassign axes
                # axes = {"x": 2, "y": 1, "z": 0}
                axes.update(shuffled_axes)
                # output: {"transpose": {"x": 2, "y": 1, "z": 0}}
                spatial_transforms[transform] = axes
            else:
                raise ValueError(f"Unknown spatial transform: {transform}")
        self._current_spatial_transforms = spatial_transforms
        return spatial_transforms

    def set_raw_value_transforms(self, transforms: Callable):
        """Sets the raw value transforms for the dataset."""
        self.raw_value_transforms = transforms
        for source in self.input_sources.values():
            source.value_transform = transforms

    def set_target_value_transforms(self, transforms: Callable):
        """Sets the ground truth value transforms for the dataset."""
        self.target_value_transforms = transforms
        for sources in self.target_sources.values():
            for source in sources.values():
                if isinstance(source, CellMapImage):
                    source.value_transform = transforms


# Example input arrays:
# {'0_input': {'shape': (90, 90, 90), 'scale': (32, 32, 32)},
#  '1_input': {'shape': (114, 114, 114), 'scale': (8, 8, 8)}}

# Example output arrays:
# {'0_output': {'shape': (28, 28, 28), 'scale': (32, 32, 32)},
#  '1_output': {'shape': (32, 32, 32), 'scale': (8, 8, 8)}}
