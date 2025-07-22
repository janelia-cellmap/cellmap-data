# %%
from concurrent.futures import ThreadPoolExecutor, as_completed
import functools
import os
from typing import Any, Callable, Mapping, Sequence, Optional
import numpy as np
from numpy.typing import ArrayLike
import torch
from torch.utils.data import Dataset
import tensorstore

from .mutable_sampler import MutableSubsetRandomSampler
from .utils import min_redundant_inds, split_target_path, is_array_2D, get_sliced_shape
from .image import CellMapImage
from .empty_image import EmptyImage
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# %%
class CellMapDataset(Dataset):
    """
    This subclasses PyTorch Dataset to load CellMap data for training. It maintains the same API as the Dataset class. Importantly, it maintains information about and handles for the sources for raw and groundtruth data. This information includes the path to the data, the classes for segmentation, and the arrays to input to the network and use as targets for the network predictions. The dataset constructs the sources for the raw and groundtruth data, and retrieves the data from the sources. The dataset also provides methods to get the number of pixels for each class in the ground truth data, normalized by the resolution. Additionally, random crops of the data can be generated for training, because the CellMapDataset maintains information about the extents of its source arrays. This object additionally combines images for different classes into a single output array, which is useful for training multiclass segmentation networks.

    """

    def __init__(
        self,
        raw_path: str,  # TODO: Switch "raw_path" to "input_path"
        target_path: str,
        classes: Sequence[str] | None,
        input_arrays: Mapping[str, Mapping[str, Sequence[int | float]]],
        target_arrays: Mapping[str, Mapping[str, Sequence[int | float]]] | None = None,
        spatial_transforms: Optional[Mapping[str, Mapping]] = None,  # type: ignore
        raw_value_transforms: Optional[Callable] = None,
        target_value_transforms: Optional[
            Callable | Sequence[Callable] | Mapping[str, Callable]
        ] = None,
        class_relation_dict: Optional[Mapping[str, Sequence[str]]] = None,
        is_train: bool = False,
        axis_order: str = "zyx",
        context: Optional[tensorstore.Context] = None,  # type: ignore
        rng: Optional[torch.Generator] = None,
        force_has_data: bool = False,
        empty_value: float | int = torch.nan,
        pad: bool = True,
        device: Optional[str | torch.device] = None,
        max_workers: Optional[int] = None,
    ) -> None:
        """Initializes the CellMapDataset class.

        Args:
            raw_path (str): The path to the raw data.
            target_path (str): The path to the ground truth data.
            classes (Sequence[str]): A list of classes for segmentation training. Class order will be preserved in the output arrays. Classes not contained in the dataset will be filled in with zeros.
            input_arrays (Mapping[str, Mapping[str, Sequence[int | float]]]): A dictionary containing the arrays of the dataset to input to the network. The dictionary should have the following structure::
            max_workers (Optional[int], optional): The maximum number of worker threads to use for parallel data loading. If not specified, defaults to the minimum of the number of CPU cores and the value of the CELLMAP_MAX_WORKERS environment variable (default 4).

                {
                    "array_name": {
                        "shape": tuple[int],
                        "scale": Sequence[float],
                    },
                    ...
                }

            where 'array_name' is the name of the array, 'shape' is the shape of the array in voxels, and 'scale' is the scale of the array in world units.
            target_arrays (Mapping[str, Mapping[str, Sequence[int | float]]]): A dictionary containing the arrays of the dataset to use as targets for the network. The dictionary should have the same structure as 'input_arrays'.
            spatial_transforms (Optional[Mapping[str, Any]] = None, optional): A sequence of dictionaries containing the spatial transformations to apply to the data. Defaults to None. The dictionary should have the following structure::

                {transform_name: {transform_args}}

            raw_value_transforms (Optional[Callable], optional): A function to apply to the raw data. Defaults to None. Example is to normalize the raw data.
            target_value_transforms (Optional[Callable | Sequence[Callable] | Mapping[str, Callable]], optional): A function to convert the ground truth data to target arrays. Defaults to None. Example is to convert the ground truth data to a signed distance transform. May be a single function, a list of functions, or a dictionary of functions for each class. In the case of a list of functions, it is assumed that the functions correspond to each class in the classes list in order. If the function is a dictionary, the keys should correspond to the classes in the 'classes' list. The function should return a tensor of the same shape as the input tensor. Note that target transforms are applied to the ground truth data and should generally not be used with use of true-negative data inferred using the 'class_relation_dict'.
            is_train (bool, optional): Whether the dataset is for training. Defaults to False.
            context (Optional[tensorstore.Context], optional): The context for the image data. Defaults to None.
            rng (Optional[torch.Generator], optional): A random number generator. Defaults to None.
            force_has_data (bool, optional): Whether to force the dataset to report that it has data. Defaults to False.
            empty_value (float | int, optional): The value to fill in for empty data. Defaults to torch.nan.
            pad (bool, optional): Whether to pad the image data to match requested arrays. Defaults to False.
            device (Optional[str | torch.device], optional): The device for the dataset. Defaults to None. If None, the device will be set to "cuda" if available, "mps" if available, or "cpu" if neither are available.

        """
        super().__init__()
        self.raw_path = raw_path
        self.target_path = target_path
        self.target_path_str, self.classes_with_path = split_target_path(target_path)
        self.classes = classes if classes is not None else []
        self.raw_only = classes is None
        self.input_arrays = input_arrays
        self.target_arrays = target_arrays if target_arrays is not None else {}
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
        self.pad = pad
        self._current_center = None
        self._current_spatial_transforms = None
        self.input_sources: dict[str, CellMapImage] = {}
        if device is not None:
            self._device = torch.device(device)
        for array_name, array_info in self.input_arrays.items():
            self.input_sources[array_name] = CellMapImage(
                self.raw_path,
                "raw",
                array_info["scale"],
                array_info["shape"],  # type: ignore
                value_transform=self.raw_value_transforms,
                context=self.context,
                pad=self.pad,
                pad_value=0,  # inputs to the network should be padded with 0
                interpolation="linear",
            )
        self.target_sources = {}
        self.has_data = (
            False if (len(self.target_arrays) > 0 and len(self.classes) > 0) else True
        )
        for array_name, array_info in self.target_arrays.items():
            if classes is None:
                self.target_sources[array_name] = CellMapImage(
                    self.raw_path,
                    "raw",
                    array_info["scale"],
                    array_info["shape"],  # type: ignore
                    value_transform=self.target_value_transforms,
                    context=self.context,
                    pad=self.pad,
                    pad_value=0,  # inputs to the network should be padded with 0
                    interpolation="linear",
                )
            else:
                self.target_sources[array_name] = self.get_target_array(array_info)

        # Initialize persistent ThreadPoolExecutor for performance
        # This eliminates the major performance bottleneck of creating new executors per __getitem__ call
        self._executor = None
        if max_workers is not None:
            self._max_workers = max_workers
        else:
            self._max_workers = min(
                os.cpu_count() or 1, int(os.environ.get("CELLMAP_MAX_WORKERS", 4))
            )

        logger.debug(
            f"CellMapDataset initialized with {len(self.input_arrays)} input arrays, "
            f"{len(self.target_arrays)} target arrays, {len(self.classes)} classes. "
            f"Using persistent ThreadPoolExecutor with {self._max_workers} workers for performance."
        )

    @property
    def executor(self) -> ThreadPoolExecutor:
        """
        Lazy initialization of persistent ThreadPoolExecutor.
        This eliminates the performance bottleneck of creating new executors per __getitem__ call.
        """
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self._max_workers)
        return self._executor

    def __del__(self):
        """Cleanup ThreadPoolExecutor to prevent resource leaks."""
        if hasattr(self, "_executor") and self._executor is not None:
            self._executor.shutdown(wait=False)

    def __new__(
        cls,
        raw_path: str,  # TODO: Switch "raw_path" to "input_path"
        target_path: str,
        classes: Sequence[str] | None,
        input_arrays: Mapping[str, Mapping[str, Sequence[int | float]]],
        target_arrays: Mapping[str, Mapping[str, Sequence[int | float]]] | None = None,
        spatial_transforms: Optional[Mapping[str, Mapping]] = None,  # type: ignore
        raw_value_transforms: Optional[Callable] = None,
        target_value_transforms: Optional[
            Callable | Sequence[Callable] | Mapping[str, Callable]
        ] = None,
        class_relation_dict: Optional[Mapping[str, Sequence[str]]] = None,
        is_train: bool = False,
        axis_order: str = "zyx",
        context: Optional[tensorstore.Context] = None,  # type: ignore
        rng: Optional[torch.Generator] = None,
        force_has_data: bool = False,
        empty_value: float | int = torch.nan,
        pad: bool = True,
        device: Optional[str | torch.device] = None,
    ):
        # Need to determine if 2D arrays are requested without slicing axis specified
        # If so, turn into a multidataset with 3 datasets each 2D arrays sliced along one axis
        if is_array_2D(input_arrays) or is_array_2D(target_arrays):
            from cellmap_data.multidataset import CellMapMultiDataset

            logger.warning(
                "2D arrays requested without slicing axis specified. Creating datasets that each slice along one axis. If this is not intended, please specify the slicing axis in the input and target arrays."
            )
            datasets = []
            for axis in range(3):
                logger.debug(f"Creating dataset for axis {axis}")
                input_arrays_2d = {
                    name: {
                        "shape": get_sliced_shape(array_info["shape"], axis),
                        "scale": array_info["scale"],
                    }
                    for name, array_info in input_arrays.items()
                }
                target_arrays_2d = (
                    {
                        name: {
                            "shape": get_sliced_shape(array_info["shape"], axis),
                            "scale": array_info["scale"],
                        }
                        for name, array_info in target_arrays.items()
                    }
                    if target_arrays is not None
                    else None
                )
                logger.debug(f"Input arrays for axis {axis}: {input_arrays_2d}")
                logger.debug(f"Target arrays for axis {axis}: {target_arrays_2d}")
                datasets.append(
                    CellMapDataset(
                        raw_path,
                        target_path,
                        classes,
                        input_arrays_2d,
                        target_arrays_2d,
                        spatial_transforms=spatial_transforms,
                        raw_value_transforms=raw_value_transforms,
                        target_value_transforms=target_value_transforms,
                        class_relation_dict=class_relation_dict,
                        is_train=is_train,
                        axis_order=axis_order,
                        context=context,
                        rng=rng,
                        force_has_data=force_has_data,
                        empty_value=empty_value,
                        pad=pad,
                        device=device,
                    )
                )
            return CellMapMultiDataset(
                classes=classes,
                input_arrays=input_arrays,
                target_arrays=target_arrays,
                datasets=datasets,
            )
        # If not, return the standard CellMapDataset
        else:
            instance = super().__new__(cls)
            return instance

    def __reduce__(self):
        """
        Support pickling for multiprocessing DataLoader and spawned processes.
        """
        # These are the args __init__ needs:
        args = (
            self.raw_path,
            self.target_path,
            self.classes,
            self.input_arrays,
            self.target_arrays,
            self.spatial_transforms,
            self.raw_value_transforms,
            self.target_value_transforms,
            self.class_relation_dict,
            self.is_train,
            self.axis_order,
            self.context,
            self._rng,
            self.force_has_data,
            self.empty_value,
            self.pad,
            self.device,
        )
        # Return: (callable, args_for_constructor, state_dict)
        return (self.__class__, args, self.__dict__)

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
    def largest_voxel_sizes(self) -> Mapping[str, float]:
        """Returns the largest voxel size of the dataset."""
        try:
            return self._largest_voxel_sizes
        except AttributeError:
            largest_voxel_size = {c: 0.0 for c in self.axis_order}
            for source in list(self.input_sources.values()) + list(
                self.target_sources.values()
            ):
                if isinstance(source, dict):
                    for _, source in source.items():
                        if not hasattr(source, "scale") or source.scale is None:  # type: ignore
                            continue
                        for c, size in source.scale.items():  # type: ignore
                            largest_voxel_size[c] = max(largest_voxel_size[c], size)
                else:
                    if not hasattr(source, "scale") or source.scale is None:
                        continue
                    for c, size in source.scale.items():
                        largest_voxel_size[c] = max(largest_voxel_size[c], size)
            self._largest_voxel_sizes = largest_voxel_size

            return self._largest_voxel_sizes

    @property
    def bounding_box(self) -> Mapping[str, list[float]]:
        """Returns the bounding box of the dataset."""
        try:
            return self._bounding_box
        except AttributeError:
            bounding_box = None
            for source in list(self.input_sources.values()) + list(
                self.target_sources.values()
            ):
                if isinstance(source, dict):
                    for source in source.values():
                        if not hasattr(source, "bounding_box"):
                            continue
                        bounding_box = self._get_box_intersection(
                            source.bounding_box, bounding_box  # type: ignore
                        )
                else:
                    if not hasattr(source, "bounding_box"):
                        continue
                    bounding_box = self._get_box_intersection(
                        source.bounding_box, bounding_box
                    )
            if bounding_box is None:
                logger.warning(
                    "Bounding box is None. This may result in errors when trying to sample from the dataset."
                )
                bounding_box = {c: [-np.inf, np.inf] for c in self.axis_order}
            self._bounding_box = bounding_box
            return self._bounding_box

    @property
    def bounding_box_shape(self) -> Mapping[str, int]:
        """Returns the shape of the bounding box of the dataset in voxels of the largest voxel size requested."""
        try:
            return self._bounding_box_shape
        except AttributeError:
            self._bounding_box_shape = self._get_box_shape(self.bounding_box)
            return self._bounding_box_shape

    @property
    def sampling_box(self) -> Mapping[str, list[float]]:
        """Returns the sampling box of the dataset (i.e. where centers can be drawn from and still have full samples drawn from within the bounding box)."""
        try:
            return self._sampling_box
        except AttributeError:
            sampling_box = None
            for source in list(self.input_sources.values()) + list(
                self.target_sources.values()
            ):
                if isinstance(source, dict):
                    for source in source.values():
                        if not hasattr(source, "sampling_box"):
                            continue
                        sampling_box = self._get_box_intersection(
                            source.sampling_box, sampling_box  # type: ignore
                        )
                else:
                    if not hasattr(source, "sampling_box"):
                        continue
                    sampling_box = self._get_box_intersection(
                        source.sampling_box, sampling_box
                    )
            if sampling_box is None:
                logger.warning(
                    "Sampling box is None. This may result in errors when trying to sample from the dataset."
                )
                sampling_box = {c: [-np.inf, np.inf] for c in self.axis_order}
            self._sampling_box = sampling_box
            return self._sampling_box

    @property
    def sampling_box_shape(self) -> dict[str, int]:
        """Returns the shape of the sampling box of the dataset in voxels of the largest voxel size requested."""
        try:
            return self._sampling_box_shape
        except AttributeError:
            self._sampling_box_shape = self._get_box_shape(self.sampling_box)
            if self.pad:
                for c, size in self._sampling_box_shape.items():
                    if size <= 0:
                        logger.debug(
                            f"Sampling box shape is <= 0 for axis {c} with size {size}. Setting to 1 and padding."
                        )
                        self._sampling_box_shape[c] = 1
            return self._sampling_box_shape

    @property
    def size(self) -> int:
        """Returns the size of the dataset in voxels of the largest voxel size requested."""
        try:
            return self._size
        except AttributeError:
            self._size = np.prod(
                [stop - start for start, stop in self.bounding_box.values()]
            ).astype(int)
            return self._size

    @property
    def class_counts(self) -> Mapping[str, Mapping[str, float]]:
        """Returns the number of pixels for each class in the ground truth data, normalized by the resolution."""
        try:
            return self._class_counts
        except AttributeError:
            class_counts = {"totals": {c: 0.0 for c in self.classes}}
            class_counts["totals"].update({c + "_bg": 0.0 for c in self.classes})
            for array_name, sources in self.target_sources.items():
                class_counts[array_name] = {}
                for label, source in sources.items():
                    if not isinstance(source, CellMapImage):
                        class_counts[array_name][label] = 0.0
                        class_counts[array_name][label + "_bg"] = 0.0
                    else:
                        class_counts[array_name][label] = source.class_counts
                        class_counts[array_name][label + "_bg"] = source.bg_count
                        class_counts["totals"][label] += source.class_counts
                        class_counts["totals"][label + "_bg"] += source.bg_count
            self._class_counts = class_counts
            return self._class_counts

    @property
    def class_weights(self) -> Mapping[str, float]:
        """Returns the class weights for the dataset based on the number of samples in each class. Classes without any samples will have a weight of NaN."""
        try:
            return self._class_weights
        except AttributeError:
            class_weights = {
                c: (
                    self.class_counts["totals"][c + "_bg"]
                    / self.class_counts["totals"][c]
                    if self.class_counts["totals"][c] != 0
                    else 1
                )
                for c in self.classes
            }
            self._class_weights = class_weights
            return self._class_weights

    @property
    def validation_indices(self) -> Sequence[int]:
        """Returns the indices of the dataset that will produce non-overlapping tiles for use in validation, based on the largest requested voxel size."""
        try:
            return self._validation_indices
        except AttributeError:
            chunk_size = {}
            for c, size in self.bounding_box_shape.items():
                chunk_size[c] = np.ceil(size - self.sampling_box_shape[c]).astype(int)
            self._validation_indices = self.get_indices(chunk_size)
            return self._validation_indices

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
        if not self.has_data and not self.force_has_data:
            return 0
        try:
            return self._len
        except AttributeError:
            size = np.prod([self.sampling_box_shape[c] for c in self.axis_order])
            self._len = int(size)
            return self._len

    def __getitem__(self, idx: ArrayLike) -> dict[str, torch.Tensor]:
        """Returns a crop of the input and target data as PyTorch tensors, corresponding to the coordinate of the unwrapped index."""
        idx = np.array(idx)
        idx[idx < 0] = len(self) + idx[idx < 0]
        try:
            center = np.unravel_index(
                idx, [self.sampling_box_shape[c] for c in self.axis_order]
            )
        except ValueError:
            # TODO: This is a hacky temprorary fix. Need to figure out why this is happening
            logger.error(
                f"Index {idx} out of bounds for dataset {self} of length {len(self)}"
            )
            logger.warning(f"Returning closest index in bounds")
            center = [self.sampling_box_shape[c] - 1 for c in self.axis_order]
        center = {
            c: center[i] * self.largest_voxel_sizes[c] + self.sampling_box[c][0]
            for i, c in enumerate(self.axis_order)
        }
        self._current_idx = idx
        self._current_center = center
        spatial_transforms = self.generate_spatial_transforms()

        # TODO: Should do as many coordinate transformations as possible at the dataset level (duplicate reference frame images should have the same coordinate transformations) --> do this per array, perhaps with CellMapArray object

        # For input arrays
        def get_input_array(array_name: str) -> tuple[str, torch.Tensor]:
            self.input_sources[array_name].set_spatial_transforms(spatial_transforms)
            array = self.input_sources[array_name][center]  # type: ignore
            return array_name, array.squeeze()[None, ...]  # Add channel dimension

        # Use persistent executor instead of creating new one (MAJOR PERFORMANCE FIX)
        futures = [
            self.executor.submit(get_input_array, array_name)
            for array_name in self.input_arrays.keys()
        ]

        # For target arrays
        if self.raw_only:

            def get_target_array(array_name: str) -> tuple[str, torch.Tensor]:
                self.target_sources[array_name].set_spatial_transforms(
                    spatial_transforms
                )
                array = self.target_sources[array_name][center]
                return array_name, array.squeeze()[None, ...]  # Add channel dimension

        else:

            def get_target_array(array_name: str) -> tuple[str, torch.Tensor]:
                class_arrays = {
                    label: None for label in self.classes
                }  # Force order of classes
                inferred_arrays = []

                # 1) Get images with gt data
                def get_label_array(
                    label: str,
                ) -> tuple[str, torch.Tensor | None]:
                    if isinstance(
                        self.target_sources[array_name][label],
                        (CellMapImage, EmptyImage),
                    ):
                        self.target_sources[array_name][
                            label
                        ].set_spatial_transforms(  # type: ignore
                            spatial_transforms
                        )
                        array = self.target_sources[array_name][label][
                            center
                        ].squeeze()  # type: ignore
                    else:
                        # Add to list of arrays to infer
                        array = None
                    return label, array

                futures = [
                    self.executor.submit(get_label_array, label)
                    for label in self.classes
                ]
                for future in as_completed(futures):
                    label, array = future.result()
                    if array is not None:
                        class_arrays[label] = array
                    else:
                        inferred_arrays.append(label)

                # 2) Infer true negatives from mutually exclusive classes in gt
                # Use the dataset device to match the device of tensors returned by CellMapImage
                empty_array = self.get_empty_store(
                    self.target_arrays[array_name], device=self.device
                )  # type: ignore

                def infer_label_array(label: str) -> tuple[str, torch.Tensor]:
                    # Make array of true negatives
                    array = empty_array.clone()
                    for other_label in self.target_sources[array_name][label]:  # type: ignore
                        if class_arrays[other_label] is not None:
                            mask = class_arrays[other_label] > 0
                            array[mask] = 0
                    return label, array

                futures = [
                    self.executor.submit(infer_label_array, label)
                    for label in inferred_arrays
                ]
                for future in as_completed(futures):
                    label, array = future.result()
                    class_arrays[label] = array
                # Ensure all tensors are on the correct device before stacking, and filter out None
                array = torch.stack(
                    [
                        (
                            arr
                            if arr.device == self.device
                            else arr.to(self.device, non_blocking=True)
                        )
                        for arr in class_arrays.values()
                        if arr is not None
                    ]
                )
                assert array.shape[0] == len(
                    self.classes
                ), f"Number of classes in target array {array_name} does not match number of classes in dataset: {len(self.classes)} != {array.shape[0]}"
                return array_name, array

        futures += [
            self.executor.submit(get_target_array, array_name)
            for array_name in self.target_arrays.keys()
        ]

        outputs = {
            "__metadata__": self.metadata,
        }
        for future in as_completed(futures):
            array_name, array = future.result()
            outputs[array_name] = array

        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        """Returns metadata about the dataset."""
        metadata = {
            "raw_path": self.raw_path,
            "current_center": self._current_center,
            "current_idx": self._current_idx,
        }

        if self._current_spatial_transforms is not None:
            metadata["current_spatial_transforms"] = self._current_spatial_transforms
        if not self.raw_only:
            metadata["target_path_str"] = self.target_path_str
            metadata["class_weights"] = self.class_weights
        return metadata

    def __repr__(self) -> str:
        """Returns a string representation of the dataset."""
        return f"CellMapDataset(\n\tRaw path: {self.raw_path}\n\tGT path(s): {self.target_path}\n\tClasses: {self.classes})"

    def get_empty_store(
        self, array_info: Mapping[str, Sequence[int]], device: torch.device
    ) -> torch.Tensor:
        """Returns an empty store, based on the requested array."""
        empty_store = torch.ones(array_info["shape"], device=device) * self.empty_value
        return empty_store.squeeze()

    def get_target_array(
        self, array_info: Mapping[str, Sequence[int | float]]
    ) -> dict[str, CellMapImage | EmptyImage | Sequence[str]]:
        """Returns a target array source for the dataset. Creates a dictionary of image sources for each class in the dataset. For classes that are not present in the ground truth data, the data can be inferred from the other classes in the dataset. This is useful for training segmentation networks with mutually exclusive classes."""
        # Use CPU device to match the device of tensors returned by CellMapImage
        empty_store = self.get_empty_store(array_info, device=torch.device("cpu"))  # type: ignore
        target_array = {}
        for i, label in enumerate(self.classes):
            target_array[label] = self.get_label_array(
                label, i, array_info, empty_store
            )
        # Check to make sure we aren't trying to define true negatives with non-existent images
        for label in self.classes:
            if isinstance(target_array[label], (CellMapImage, EmptyImage)):
                continue
            is_empty = True
            for other_label in target_array[label]:
                if other_label in target_array and isinstance(
                    target_array[other_label], CellMapImage
                ):
                    is_empty = False
                    break
            if is_empty:
                target_array[label] = EmptyImage(
                    label, array_info["scale"], array_info["shape"], empty_store  # type: ignore
                )

        return target_array

    def get_label_array(
        self,
        label: str,
        i: int,
        array_info: Mapping[str, Sequence[int | float]],
        empty_store: torch.Tensor,
    ) -> CellMapImage | EmptyImage | Sequence[str]:
        """Returns a target array source for a specific class in the dataset."""
        if label in self.classes_with_path:
            if isinstance(self.target_value_transforms, dict):
                value_transform: Callable = self.target_value_transforms[label]
            elif isinstance(self.target_value_transforms, list):
                value_transform = self.target_value_transforms[i]
            else:
                value_transform = self.target_value_transforms  # type: ignore
            array = CellMapImage(
                self.target_path_str.format(label=label),
                label,
                array_info["scale"],
                array_info["shape"],  # type: ignore
                value_transform=value_transform,
                context=self.context,
                pad=self.pad,
                pad_value=self.empty_value,
                interpolation="nearest",
            )
            if not self.has_data:
                self.has_data = array.class_counts != 0
        else:
            if (
                self.class_relation_dict is not None
                and label in self.class_relation_dict
            ):
                # Add lookup of source images for true-negatives in absence of annotations
                array = self.class_relation_dict[label]
            else:
                array = EmptyImage(
                    label, array_info["scale"], array_info["shape"], empty_store  # type: ignore
                )
        return array

    def _get_box_shape(self, source_box: Mapping[str, list[float]]) -> dict[str, int]:
        """Returns the shape of the box in voxels of the largest voxel size requested."""
        box_shape = {}
        for c, (start, stop) in source_box.items():
            size = stop - start
            size /= self.largest_voxel_sizes[c]
            box_shape[c] = int(np.floor(size))
        return box_shape

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

    def get_indices(self, chunk_size: Mapping[str, int]) -> Sequence[int]:
        """Returns the indices of the dataset that will tile the dataset according to the chunk_size."""
        # TODO: ADD TEST
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

    def to(
        self, device: str | torch.device, non_blocking: bool = True
    ) -> "CellMapDataset":
        """Sets the device for the dataset."""
        self._device = torch.device(device)
        for source in list(self.input_sources.values()) + list(
            self.target_sources.values()
        ):
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

    def generate_spatial_transforms(self) -> Optional[Mapping[str, Any]]:
        """When 'self.is_train' is True, generates random spatial transforms for the dataset, based on the user specified transforms.

        Available spatial transforms:
            - "mirror": Mirrors the data along the specified axes. Parameters are the probabilities of mirroring along each axis, formatted as a dictionary of axis: probability pairs. Example: {"mirror": {"axes": {"x": 0.5, "y": 0.5, "z":0.1}}} will mirror the data along the x and y axes with a 50% probability, and along the z axis with a 10% probability.
            - "transpose": Transposes the data along the specified axes. Parameters are the axes to transpose, formatted as a list. Example: {"transpose": {"axes": ["x", "z"]}} will randomly transpose the data along the x and z axes.
            - "rotate": Rotates the data around the specified axes within the specified angle ranges. Parameters are the axes to rotate and the angle ranges, formatted as a dictionary of axis: [min_angle, max_angle] pairs. Example: {"rotate": {"axes": {"x": [-180,180], "y": [-180,180], "z":[-180,180]}} will rotate the data around the x, y, and z axes from 180 to -180 degrees.
        """

        if not self.is_train or self.spatial_transforms is None:
            return None
        spatial_transforms: dict[str, Any] = {}
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
            elif transform == "rotate":
                # input: "rotate": {"axes": {"x": [-180,180], "y": [-180,180], "z":[-180,180]}}
                # output: {"rotate": {"x": 45, "y": 90, "z": 0}}
                spatial_transforms[transform] = {}
                for axis, limits in params["axes"].items():
                    spatial_transforms[transform][axis] = torch.rand(
                        1, generator=self._rng
                    ).item()
                    spatial_transforms[transform][axis] = (
                        spatial_transforms[transform][axis] * (limits[1] - limits[0])
                        + limits[0]
                    )
            else:
                raise ValueError(f"Unknown spatial transform: {transform}")
        self._current_spatial_transforms = spatial_transforms
        return spatial_transforms

    def set_raw_value_transforms(self, transforms: Callable) -> None:
        """Sets the raw value transforms for the dataset."""
        self.raw_value_transforms = transforms
        for source in self.input_sources.values():
            source.value_transform = transforms

    def set_target_value_transforms(self, transforms: Callable) -> None:
        """Sets the ground truth value transforms for the dataset."""
        self.target_value_transforms = transforms
        for sources in self.target_sources.values():
            for source in sources.values():
                if isinstance(source, CellMapImage):
                    source.value_transform = transforms

    def reset_arrays(self, type: str = "target") -> None:
        """Sets the arrays for the dataset to return."""
        if type.lower() == "input":
            self.input_sources = {}
            for array_name, array_info in self.input_arrays.items():
                self.input_sources[array_name] = CellMapImage(
                    self.raw_path,
                    "raw",
                    array_info["scale"],
                    array_info["shape"],  # type: ignore
                    value_transform=self.raw_value_transforms,
                    context=self.context,
                    pad=self.pad,
                    pad_value=0,  # inputs to the network should be padded with 0
                )
        elif type.lower() == "target":
            self.target_sources = {}
            self.has_data = False
            for array_name, array_info in self.target_arrays.items():
                self.target_sources[array_name] = self.get_target_array(array_info)
        else:
            raise ValueError(f"Unknown dataset array type: {type}")

    def get_random_subset_indices(
        self, num_samples: int, rng: Optional[torch.Generator] = None, **kwargs: Any
    ) -> Sequence[int]:
        return min_redundant_inds(len(self), num_samples, rng=rng).tolist()

    def get_subset_random_sampler(
        self,
        num_samples: int,
        rng: Optional[torch.Generator] = None,
        **kwargs: Any,
    ) -> MutableSubsetRandomSampler:
        """
        Returns a random sampler that yields exactly `num_samples` indices from this subset.
        - If `num_samples` ≤ total number of available indices, samples without replacement.
        - If `num_samples` > total number of available indices, samples with replacement using repeated shuffles to minimize duplicates.
        """

        indices_generator = functools.partial(
            self.get_random_subset_indices, num_samples, rng, **kwargs
        )

        return MutableSubsetRandomSampler(
            indices_generator,
            rng=rng,
        )

    @staticmethod
    def empty() -> "CellMapDataset":
        """Creates an empty dataset."""
        empty_dataset = CellMapDataset("", "", [], {}, {})
        empty_dataset.classes = []
        empty_dataset._class_counts = {}
        empty_dataset._class_weights = {}
        empty_dataset._validation_indices = []
        empty_dataset.has_data = False
        empty_dataset._len = 0

        return empty_dataset


# %%
