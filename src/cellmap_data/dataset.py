# %%
import os
from typing import Any, Callable, Dict, Mapping, Sequence, Optional
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
    target_sources: dict[str, dict[str, CellMapImage | EmptyImage | Sequence[str]]]
    spatial_transforms: Optional[dict[str, any]]  # type: ignore
    raw_value_transforms: Optional[Callable]
    target_value_transforms: Optional[
        Callable | Sequence[Callable] | dict[str, Callable]
    ]
    class_relation_dict: Optional[Mapping[str, Sequence[str]]]
    empty_value: float | int | str
    has_data: bool
    is_train: bool
    axis_order: str
    context: Optional[tensorstore.Context]  # type: ignore

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
        class_relation_dict: Optional[Mapping[str, Sequence[str]]] = None,
        is_train: bool = False,
        axis_order: str = "zyx",
        context: Optional[tensorstore.Context] = None,  # type: ignore
        rng: Optional[torch.Generator] = None,
        force_has_data: bool = False,
        empty_value: float | int | str = 0,
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
            empty_value (float | int | str, optional): The value to fill in for empty data. If set to "mask" will also produce training masks for empty data (data in target arrays will be 0). Defaults to 0.
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
        self.class_relation_dict = class_relation_dict
        self.is_train = is_train
        self.axis_order = axis_order
        self.context = context
        self._rng = rng
        self.force_has_data = force_has_data
        self.empty_value = empty_value
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
            self.target_sources[array_name] = self.get_target_array(array_info)

    def get_empty_store(self, array_info):
        if self.empty_value == "mask":
            empty_store = torch.ones(array_info["shape"]) * -100  # type: ignore
        else:
            assert isinstance(
                self.empty_value, (float, int)
            ), "Empty value must be `mask` or a number."
            empty_store = torch.ones(array_info["shape"]) * self.empty_value  # type: ignore
        return empty_store

    def get_target_array(self, array_info):
        empty_store = self.get_empty_store(array_info)
        target_array = {}
        for i, label in enumerate(self.classes):  # type: ignore
            target_array[label] = self.get_label_array(
                label, i, array_info, empty_store
            )
        # Check to make sure we aren't trying to define true negatives with non-existent images
        for label in self.classes:
            if isinstance(target_array[label], (CellMapImage, EmptyImage)):
                continue
            is_empty = True
            for other_label in target_array[label]:
                if isinstance(target_array[other_label], (CellMapImage, EmptyImage)):
                    is_empty = False
                    break
            if is_empty:
                target_array[label] = empty_store

        return target_array

    def get_label_array(self, label, i, array_info, empty_store):
        if label in self.classes_with_path:
            if isinstance(self.target_value_transforms, dict):
                value_transform: Callable = self.target_value_transforms[label]
            elif isinstance(self.target_value_transforms, list):
                value_transform: Callable = self.target_value_transforms[i]
            else:
                value_transform: Callable = self.target_value_transforms  # type: ignore
            array = CellMapImage(
                self.target_path_str.format(label=label),
                label,
                array_info["scale"],
                array_info["shape"],  # type: ignore
                value_transform=value_transform,
                context=self.context,
            )
            if not self.has_data:
                self.has_data = self.has_data or array.class_counts != 0  # type: ignore
        else:
            if (
                self.class_relation_dict is not None
                and label in self.class_relation_dict
            ):
                # Add lookup of source images for true-negatives in absence of annotations
                array = self.class_relation_dict[label]
            else:
                array = EmptyImage(
                    label, array_info["shape"], empty_store  # type: ignore
                )
        return array

    def __len__(self):
        """Returns the length of the dataset, determined by the number of coordinates that could be sampled as the center for a cube."""
        if not self.has_data and not self.force_has_data:
            return 0
        if not hasattr(self, "_len"):
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
            center = [self.sampling_box_shape[c] - 1 for c in self.axis_order]
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
            # TODO: Assumes 1 channel (i.e. grayscale)
            if array.shape[0] != 1:
                outputs[array_name] = array[None, ...]
            else:
                outputs[array_name] = array
        # TODO: Allow for distribution of array gathering to multiple threads
        for array_name in self.target_arrays.keys():
            class_arrays = {}
            mask_arrays = {}
            inferred_arrays = []
            for label in self.classes:
                self.target_sources[array_name][label].set_spatial_transforms(
                    spatial_transforms
                )
                if isinstance(
                    self.target_sources[array_name][label], (CellMapImage, EmptyImage)
                ):
                    array = self.target_sources[array_name][label][center].squeeze()
                else:
                    # Add to list of arrays to infer
                    inferred_arrays.append(label)
                    array = None
                class_arrays[label] = array

            for label in inferred_arrays:
                # Make array of true negatives
                array = self.get_empty_store(self.target_arrays[array_name])
                for other_label in self.target_sources[array_name][label]:  # type: ignore
                    array[class_arrays[other_label] > 0] = 0
                class_arrays[label] = array

            if self.masked:
                for label in self.classes:
                    # TODO: Find a better way to do this
                    # TODO: This will break with inferred arrays
                    mask = (
                        class_arrays[label] == -100
                    )  # Get all places where the array is empty
                    mask = mask.cpu()  # Convert to CPU to avoid memory issues
                    class_arrays[label][mask] = 0  # Set all empty places to 0
                    mask_arrays[label] = (
                        mask == 0
                    )  # Take the inverse of the empty places mask to use for masking the loss
                outputs[array_name + "_mask"] = torch.stack(list(mask_arrays.values()))
            outputs[array_name] = torch.stack(list(class_arrays.values()))
        return outputs

    def __repr__(self):
        """Returns a string representation of the dataset."""
        return f"CellMapDataset(\n\tRaw path: {self.raw_path}\n\tGT path(s): {self.target_paths}\n\tClasses: {self.classes})"

    @property
    def masked(self):
        """Returns whether the dataset returns training masks, alongside input and target arrays."""
        if not hasattr(self, "_masked"):
            self._masked = self.empty_value == "mask"
            if self._masked:
                DeprecationWarning(
                    "The `masked` attribute is deprecated, and not recommended due to increased memory overhead."
                )
        return self._masked

    @property
    def largest_voxel_sizes(self) -> dict[str, float]:
        """Returns the largest voxel size of the dataset."""
        if not hasattr(self, "_largest_voxel_sizes"):
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
    def bounding_box(self) -> dict[str, list[float]]:
        """Returns the bounding box of the dataset."""
        if not hasattr(self, "_bounding_box"):
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
    def bounding_box_shape(self) -> dict[str, int]:
        """Returns the shape of the bounding box of the dataset in voxels of the largest voxel size."""
        if not hasattr(self, "_bounding_box_shape"):
            self._bounding_box_shape = self._get_box_shape(self.bounding_box)
        return self._bounding_box_shape

    @property
    def sampling_box(self) -> dict[str, list[float]]:
        """Returns the sampling box of the dataset (i.e. where centers can be drawn from and still have full samples drawn from within the bounding box)."""
        if not hasattr(self, "_sampling_box"):
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
    def sampling_box_shape(self) -> dict[str, int]:
        """Returns the shape of the sampling box of the dataset in voxels of the largest voxel size."""
        if not hasattr(self, "_sampling_box_shape"):
            self._sampling_box_shape = self._get_box_shape(self.sampling_box)
        return self._sampling_box_shape

    @property
    def class_weights(self) -> dict[str, float]:
        """
        Returns the class weights for the dataset based on the number of samples in each class.
        """
        if not hasattr(self, "_class_weights"):
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
            self._class_weights = class_weights
        return self._class_weights

    @property
    def class_counts(self) -> Dict[str, Dict[str, int]]:
        """Returns the number of pixels for each class in the ground truth data, normalized by the resolution."""
        if not hasattr(self, "_class_counts"):
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
        if not hasattr(self, "_validation_indices"):
            chunk_size = {}
            for c, size in self.bounding_box_shape.items():
                chunk_size[c] = np.ceil(size - self.sampling_box_shape[c]).astype(int)
            self._validation_indices = self.get_indices(chunk_size)
        return self._validation_indices

    def _get_box_shape(self, source_box: dict[str, list[float]]) -> dict[str, int]:
        box_shape = {}
        for c, (start, stop) in source_box.items():
            size = stop - start
            size /= self.largest_voxel_sizes[c]
            box_shape[c] = int(np.floor(size))
        return box_shape

    def _get_box(
        self,
        source_box: dict[str, list[float]],
        current_box: dict[str, list[float]],
    ) -> dict[str, list[float]]:
        if source_box is not None:
            for c, (start, stop) in source_box.items():
                assert stop > start
                current_box[c][0] = max(current_box[c][0], start)
                current_box[c][1] = min(current_box[c][1], stop)
        return current_box

    def verify(self):
        """Verifies that the dataset is valid to draw samples from."""
        # TODO: make more robust
        try:
            return len(self) > 0
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

        if not self.is_train or self.spatial_transforms is None:
            return None
        spatial_transforms = {}
        for transform, params in self.spatial_transforms.items():
            if transform == "mirror":
                # input: "mirror": {"axes": {"x": 0.5, "y": 0.5, "z":0.1}}
                # output: {"mirror": ["x", "y"]}
                spatial_transforms[transform] = []
                for axis, prob in params["axes"].items():
                    if torch.rand(1, generator=self._rng).item() < prob:
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
                shuffled_axes = [
                    shuffled_axes[i]
                    for i in torch.randperm(len(shuffled_axes), generator=self._rng)
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
