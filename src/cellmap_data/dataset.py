# %%
import atexit
import functools
from functools import cached_property
import logging
import os
import platform
from concurrent.futures import Future as _ConcurrentFuture
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import tensorstore
import torch
from numpy.typing import ArrayLike
from torch.utils.data import Dataset

from .base_dataset import CellMapBaseDataset
from .empty_image import EmptyImage
from .image import CellMapImage
from .mutable_sampler import MutableSubsetRandomSampler
from .utils.read_limiter import MAX_CONCURRENT_READS, limit_tensorstore_reads
from .utils import get_sliced_shape, is_array_2D, min_redundant_inds, split_target_path

logger = logging.getLogger(__name__)
if logger.level == logging.NOTSET:
    logger.setLevel(logging.INFO)

# Cache system values to avoid repeated calls during dataset instantiation
_OS_NAME = platform.system()
_DATA_BACKEND = os.environ.get("CELLMAP_DATA_BACKEND", "tensorstore")

# On Windows + TensorStore, calling tensorstore's .read().result() from a
# Python ThreadPoolExecutor worker thread causes a hard native crash
# (STATUS_STACK_BUFFER_OVERRUN / abort, exit code 0xC0000409).  The
# limit_tensorstore_reads semaphore only prevents *concurrent* Python reads
# but does not fix the per-thread crash.  The safest fix is to run all
# dataset __getitem__ work synchronously in the calling thread so that
# TensorStore is never invoked from a ThreadPoolExecutor worker on Windows.
_USE_IMMEDIATE_EXECUTOR = (
    _OS_NAME == "Windows" and _DATA_BACKEND.lower() == "tensorstore"
)


class _ImmediateExecutor:
    """Drop-in for ThreadPoolExecutor that runs tasks in the calling thread.

    On Windows + TensorStore the real ThreadPoolExecutor causes native crashes.
    This executor avoids that by executing every submitted callable synchronously
    before returning, so the returned Future is already resolved.
    ``as_completed`` handles pre-resolved futures correctly (yields immediately).
    ``shutdown`` is a no-op because there are no threads to join.
    """

    def submit(self, fn, /, *args, **kwargs):
        f = _ConcurrentFuture()
        try:
            f.set_result(fn(*args, **kwargs))
        except Exception as exc:  # noqa: BLE001
            f.set_exception(exc)
        return f

    def shutdown(self, wait=True, *, cancel_futures=False):
        pass  # nothing to shut down


_IMMEDIATE_EXECUTOR: _ImmediateExecutor | None = (
    _ImmediateExecutor() if _USE_IMMEDIATE_EXECUTOR else None
)


# %%
class CellMapDataset(CellMapBaseDataset, Dataset):
    """
    Subclasses PyTorch Dataset to load CellMap data for training.

    This class handles data sources for raw and ground truth data, including paths,
    segmentation classes, and input/target array configurations. It retrieves data,
    calculates class-specific pixel counts, and generates random crops for training.
    It also combines images for different classes into a single output array,
    which is useful for training multi-class segmentation networks.
    """

    def __init__(
        self,
        raw_path: str,
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
        device: Optional[str | torch.device] = "cpu",
        max_workers: Optional[int] = None,
    ) -> None:
        """Initializes the CellMapDataset class.

        Args:
        ----
            raw_path: Path to the raw data.
            target_path: Path to the ground truth data.
            classes: List of classes for segmentation training.
            input_arrays: Dictionary of input arrays with shape and scale.
            target_arrays: Dictionary of target arrays with shape and scale.
            spatial_transforms: Spatial transformations to apply.
            raw_value_transforms: Transforms for raw data (e.g., normalization).
            target_value_transforms: Transforms for target data (e.g., distance transform).
            class_relation_dict: Defines mutual exclusivity between classes.
            is_train: Whether the dataset is for training.
            axis_order: The order of axes (e.g., "zyx").
            context: TensorStore context.
            rng: Random number generator.
            force_has_data: If True, forces the dataset to report having data.
            empty_value: Value for empty data.
            pad: Whether to pad data to match requested array shapes.
            device: The device for torch tensors. Defaults to CPU.
            max_workers: Max worker threads for data loading.
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
        self._device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        for array_name, array_info in self.input_arrays.items():
            self.input_sources[array_name] = CellMapImage(
                self.raw_path,
                "raw",
                array_info["scale"],  # type: ignore
                tuple(map(int, array_info["shape"])),
                value_transform=self.raw_value_transforms,
                context=self.context,
                pad=self.pad,
                pad_value=0,
                interpolation="linear",
                device=self._device,
            )
        self.target_sources = {}
        self.has_data = force_has_data or (
            False if (len(self.target_arrays) > 0 and len(self.classes) > 0) else True
        )
        for array_name, array_info in self.target_arrays.items():
            if classes is None:
                self.target_sources[array_name] = CellMapImage(
                    self.raw_path,
                    "raw",
                    array_info["scale"],  # type: ignore
                    tuple(map(int, array_info["shape"])),
                    value_transform=self.target_value_transforms,
                    context=self.context,
                    pad=self.pad,
                    pad_value=0,
                    interpolation="linear",
                    device=self._device,
                )
            else:
                self.target_sources[array_name] = self.get_target_array(array_info)

        self._executor = None
        self._executor_pid = None
        if max_workers is not None:
            self._max_workers = max_workers
        else:
            # For HPC with I/O lag: prioritize I/O parallelism over CPU count
            # Estimate based on number of concurrent I/O operations needed
            estimated_concurrent_io = len(self.input_arrays) + len(self.target_arrays)
            # Use at least 2 workers (input + target), cap at reasonable limit
            # to avoid thread overhead while allowing parallel I/O requests
            self._max_workers = min(
                max(estimated_concurrent_io, 2),  # At least 2 workers
                int(os.environ.get("CELLMAP_MAX_WORKERS", 8)),  # Cap at 8 by default
            )

        logger.info(
            "CellMapDataset: OS=%s backend=%s max_workers=%d max_concurrent_reads=%s "
            "inputs=%d targets=%d classes=%d",
            _OS_NAME,
            _DATA_BACKEND,
            self._max_workers,
            (
                str(MAX_CONCURRENT_READS)
                if MAX_CONCURRENT_READS is not None
                else "unlimited"
            ),
            len(self.input_arrays),
            len(self.target_arrays),
            len(self.classes),
        )
        atexit.register(self.close)

    @property
    def executor(self) -> ThreadPoolExecutor | _ImmediateExecutor:
        """
        Lazy initialization of persistent executor.

        On Windows + TensorStore returns a module-level ``_ImmediateExecutor``
        that runs every submitted callable synchronously in the calling thread.
        This avoids the native crash (0xC0000409 / STATUS_STACK_BUFFER_OVERRUN)
        that occurs when TensorStore's ``.read().result()`` is called from a
        Python ``ThreadPoolExecutor`` worker thread on Windows.

        On all other platforms returns the usual persistent ``ThreadPoolExecutor``.

        In both cases ``self._executor`` and ``self._executor_pid`` are kept in
        sync so that ``close()``, ``__del__``, and tests can inspect them
        consistently regardless of platform.
        """
        current_pid = os.getpid()

        if _USE_IMMEDIATE_EXECUTOR:
            # Use the module-level singleton but still track state so that
            # _executor / _executor_pid are never left as None after first access.
            if self._executor is None or self._executor_pid != current_pid:
                self._executor = _IMMEDIATE_EXECUTOR
                self._executor_pid = current_pid
            return self._executor  # type: ignore[return-value]

        # Non-Windows path: detect process forking and create a fresh executor.
        if self._executor_pid != current_pid:
            self._executor = None
            self._executor_pid = current_pid

        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self._max_workers)
        return self._executor

    def __str__(self) -> str:
        return f"CellMapDataset(raw_path={self.raw_path}, target_path={self.target_path}, classes={self.classes})"

    def __del__(self):
        """Cleanup ThreadPoolExecutor to prevent resource leaks."""
        if hasattr(self, "_executor") and self._executor is not None:
            self._executor.shutdown(wait=True)

    def close(self) -> None:
        """Shut down the ThreadPoolExecutor and release resources.

        Called automatically via atexit to ensure clean shutdown at interpreter
        exit, regardless of whether __del__ is invoked.
        """
        if hasattr(self, "_executor") and self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

    def __new__(
        cls,
        raw_path: str,
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
    ):
        # If 2D arrays are requested without a slicing axis, create a
        # multidataset with 3 datasets, each slicing along one axis.
        if is_array_2D(input_arrays, summary=any) or is_array_2D(
            target_arrays, summary=any
        ):
            from cellmap_data.multidataset import CellMapMultiDataset

            logger.info(
                "2D arrays requested without slicing axis. Creating datasets "
                "that each slice along one axis. If this is not intended, "
                "specify the slicing axis in the input and target arrays."
            )
            datasets = []
            for axis in range(3):
                logger.debug("Creating dataset for axis %d", axis)
                input_arrays_2d = {
                    name: {
                        "shape": get_sliced_shape(
                            tuple(map(int, array_info["shape"])), axis
                        ),
                        "scale": array_info["scale"],
                    }
                    for name, array_info in input_arrays.items()
                }
                target_arrays_2d = (
                    {
                        name: {
                            "shape": get_sliced_shape(
                                tuple(map(int, array_info["shape"])), axis
                            ),
                            "scale": array_info["scale"],
                        }
                        for name, array_info in target_arrays.items()
                    }
                    if target_arrays is not None
                    else None
                )
                logger.debug("Input arrays for axis %d: %s", axis, input_arrays_2d)
                logger.debug("Target arrays for axis %d: %s", axis, target_arrays_2d)
                dataset_instance = super(CellMapDataset, cls).__new__(cls)
                dataset_instance.__init__(
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
                    max_workers=max_workers,
                )
                datasets.append(dataset_instance)
            return CellMapMultiDataset(
                classes=classes,
                input_arrays=input_arrays,
                target_arrays=target_arrays,
                datasets=datasets,
            )
        else:
            return super().__new__(cls)

    def __reduce__(self):
        """
        Support pickling for multiprocessing DataLoader.
        """
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
            self.device.type if hasattr(self.device, "type") else self.device,
            self._max_workers,
        )
        return (self.__class__, args, self.__dict__)

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
    def largest_voxel_sizes(self) -> Mapping[str, float]:
        """Returns the largest voxel size of the dataset."""
        largest_voxel_size = dict.fromkeys(self.axis_order, 0.0)
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
        return largest_voxel_size

    @cached_property
    def bounding_box(self) -> Mapping[str, list[float]]:
        """Returns the bounding box of the dataset."""
        all_sources = list(self.input_sources.values()) + list(
            self.target_sources.values()
        )
        # Flatten to individual CellMapImage objects
        flat_sources = []
        for source in all_sources:
            if isinstance(source, dict):
                flat_sources.extend(
                    s for s in source.values() if hasattr(s, "bounding_box")
                )
            elif hasattr(source, "bounding_box"):
                flat_sources.append(source)

        # Prefetch bounding boxes in parallel (each triggers a zarr group open)
        # Use self.executor to respect Windows+TensorStore immediate executor handling
        boxes = list(self.executor.map(lambda s: s.bounding_box, flat_sources))

        bounding_box: dict[str, list[float]] | None = None
        for box in boxes:
            bounding_box = self._get_box_intersection(box, bounding_box)

        if bounding_box is None:
            logger.warning(
                "Bounding box is None. This may cause errors during sampling."
            )
            bounding_box = {c: [-np.inf, np.inf] for c in self.axis_order}
        return bounding_box

    @cached_property
    def bounding_box_shape(self) -> Mapping[str, int]:
        """Returns the shape of the bounding box of the dataset in voxels of the largest voxel size requested."""
        return self._get_box_shape(self.bounding_box)

    @cached_property
    def sampling_box(self) -> Mapping[str, list[float]]:
        """Returns the sampling box of the dataset (i.e. where centers can be drawn from and still have full samples drawn from within the bounding box)."""
        all_sources = list(self.input_sources.values()) + list(
            self.target_sources.values()
        )
        flat_sources = []
        for source in all_sources:
            if isinstance(source, dict):
                flat_sources.extend(
                    s for s in source.values() if hasattr(s, "sampling_box")
                )
            elif hasattr(source, "sampling_box"):
                flat_sources.append(source)

        # Prefetch sampling boxes in parallel; bounding_box is already cached
        # from the bounding_box property so these are cheap if called after it.
        # Use self.executor to respect Windows+TensorStore immediate executor handling
        boxes = list(self.executor.map(lambda s: s.sampling_box, flat_sources))

        sampling_box: dict[str, list[float]] | None = None
        for box in boxes:
            sampling_box = self._get_box_intersection(box, sampling_box)

        if sampling_box is None:
            logger.warning(
                "Sampling box is None. This may cause errors during sampling."
            )
            sampling_box = {c: [-np.inf, np.inf] for c in self.axis_order}
        return sampling_box

    @cached_property
    def sampling_box_shape(self) -> dict[str, int]:
        """Returns the shape of the sampling box of the dataset in voxels of the largest voxel size requested."""
        shape = self._get_box_shape(self.sampling_box)
        if self.pad:
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

    @cached_property
    def size(self) -> int:
        """Returns the size of the dataset in voxels of the largest voxel size requested."""
        return int(
            np.prod([stop - start for start, stop in self.bounding_box.values()])
        )

    @cached_property
    def class_counts(self) -> Mapping[str, Mapping[str, float]]:
        """Returns the number of pixels for each class in the ground truth data, normalized by the resolution."""
        class_counts = {"totals": dict.fromkeys(self.classes, 0.0)}
        class_counts["totals"].update({c + "_bg": 0.0 for c in self.classes})
        for array_name, sources in self.target_sources.items():
            class_counts[array_name] = {}
            for label, source in sources.items():
                if isinstance(source, CellMapImage):
                    class_counts[array_name][label] = source.class_counts
                    class_counts[array_name][label + "_bg"] = source.bg_count
                    class_counts["totals"][label] += source.class_counts
                    class_counts["totals"][label + "_bg"] += source.bg_count
                else:
                    class_counts[array_name][label] = 0.0
                    class_counts[array_name][label + "_bg"] = 0.0
        return class_counts

    @cached_property
    def class_weights(self) -> dict[str, float]:
        """Returns the class weights for the dataset based on the number of samples in each class. Classes without any samples will have a weight of 1."""
        if self.classes is None:
            return {}
        return {
            c: (
                self.class_counts["totals"][c + "_bg"] / self.class_counts["totals"][c]
                if self.class_counts["totals"][c] != 0
                else 1
            )
            for c in self.classes
        }

    @cached_property
    def validation_indices(self) -> Sequence[int]:
        """Returns the indices of the dataset that will produce non-overlapping tiles for use in validation, based on the largest requested voxel size."""
        chunk_size = {
            c: np.ceil(size - self.sampling_box_shape[c]).astype(int)
            for c, size in self.bounding_box_shape.items()
        }
        return self.get_indices(chunk_size)

    @property
    def device(self) -> torch.device:
        """Returns the device for the dataset."""
        return self._device

    def __len__(self) -> int:
        """Returns the number of unique patches in the dataset."""
        if not self.has_data and not self.force_has_data:
            return 0
        # Return at least 1 if the dataset has data, so that samplers can be initialized
        return int(max(np.prod(list(self.sampling_box_shape.values())), 1))

    def __getitem__(self, idx: ArrayLike) -> dict[str, torch.Tensor]:
        """Returns a crop of the input and target data as PyTorch tensors, corresponding to the coordinate of the unwrapped index."""
        try:
            idx_arr = np.array(idx)
            if np.any(idx_arr < 0):
                idx_arr[idx_arr < 0] = len(self) + idx_arr[idx_arr < 0]

            center_indices = np.unravel_index(
                idx_arr, [self.sampling_box_shape[c] for c in self.axis_order]
            )
        except ValueError:
            logger.error(
                "Index %s out of bounds for dataset of length %d", idx, len(self)
            )
            logger.warning("Returning closest index in bounds")
            center_indices = [self.sampling_box_shape[c] - 1 for c in self.axis_order]
        center = {
            c: float(
                center_indices[i] * self.largest_voxel_sizes[c]
                + self.sampling_box[c][0]
            )
            for i, c in enumerate(self.axis_order)
        }

        self._current_idx = idx
        self._current_center = center
        spatial_transforms = self.generate_spatial_transforms()

        def get_input_array(array_name: str) -> tuple[str, torch.Tensor]:
            self.input_sources[array_name].set_spatial_transforms(spatial_transforms)
            with limit_tensorstore_reads():
                array = self.input_sources[array_name][center]
            return array_name, array.squeeze()[None, ...]

        futures = [
            self.executor.submit(get_input_array, array_name)
            for array_name in self.input_arrays.keys()
        ]

        if self.raw_only:

            def get_target_array(array_name: str) -> tuple[str, torch.Tensor]:
                self.target_sources[array_name].set_spatial_transforms(
                    spatial_transforms
                )
                with limit_tensorstore_reads():
                    array = self.target_sources[array_name][center]
                return array_name, array.squeeze()[None, ...]

        else:

            def get_target_array(array_name: str) -> tuple[str, torch.Tensor]:
                class_arrays = dict.fromkeys(self.classes)  # Force order of classes
                inferred_arrays = []

                def get_label_array(
                    label: str,
                ) -> tuple[str, torch.Tensor | None]:
                    source = self.target_sources[array_name].get(label)
                    if isinstance(source, (CellMapImage, EmptyImage)):
                        source.set_spatial_transforms(spatial_transforms)
                        with limit_tensorstore_reads():
                            array = source[center].squeeze()
                    else:
                        array = None
                    return label, array

                label_futures = [
                    self.executor.submit(get_label_array, label)
                    for label in self.classes
                ]
                for future in as_completed(label_futures):
                    label, array = future.result()
                    if array is not None:
                        class_arrays[label] = array
                    else:
                        inferred_arrays.append(label)

                empty_array = self.get_empty_store(
                    self.target_arrays[array_name], device=self.device
                )

                def infer_label_array(label: str) -> tuple[str, torch.Tensor]:
                    array = empty_array.clone()
                    other_labels = self.target_sources[array_name].get(label, [])
                    for other_label in other_labels:
                        other_array = class_arrays.get(other_label)
                        if other_array is not None:
                            mask = other_array > 0
                            array[mask] = 0
                    return label, array

                infer_futures = [
                    self.executor.submit(infer_label_array, label)
                    for label in inferred_arrays
                ]
                for future in as_completed(infer_futures):
                    label, array = future.result()
                    class_arrays[label] = array

                stacked_arrays = []
                for label in self.classes:
                    arr = class_arrays.get(label)
                    if arr is not None:
                        stacked_arrays.append(
                            arr.to(self.device, non_blocking=True)
                            if arr.device != self.device
                            else arr
                        )

                array = torch.stack(stacked_arrays)
                if array.shape[0] != len(self.classes):
                    raise ValueError(
                        f"Target array {array_name} has {array.shape[0]} classes, "
                        f"but {len(self.classes)} were expected."
                    )
                return array_name, array

        futures += [
            self.executor.submit(get_target_array, array_name)
            for array_name in self.target_arrays.keys()
        ]

        outputs: dict[str, Any] = {
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
        return (
            f"CellMapDataset(\n\tRaw path: {self.raw_path}\n\t"
            f"GT path(s): {self.target_path}\n\tClasses: {self.classes})"
        )

    def get_empty_store(
        self, array_info: Mapping[str, Sequence[int | float]], device: torch.device
    ) -> torch.Tensor:
        """Returns an empty store, based on the requested array."""
        shape = tuple(map(int, array_info["shape"]))
        empty_store = torch.ones(shape, device=device) * self.empty_value
        return empty_store.squeeze()

    def get_target_array(
        self, array_info: Mapping[str, Sequence[int | float]]
    ) -> dict[str, CellMapImage | EmptyImage | Sequence[str]]:
        """
        Returns a target array source for the dataset.

        Creates a dictionary of image sources for each class. If ground truth
        data is missing for a class, it can be inferred from other mutually
        exclusive classes.
        """
        empty_store = self.get_empty_store(array_info, device=torch.device("cpu"))
        target_array = {}
        for i, label in enumerate(self.classes):
            target_array[label] = self.get_label_array(
                label, i, array_info, empty_store
            )

        for label in self.classes:
            if isinstance(target_array.get(label), (CellMapImage, EmptyImage)):
                continue

            is_empty = True
            related_labels = target_array.get(label)
            if isinstance(related_labels, list):
                for other_label in related_labels:
                    if isinstance(target_array.get(other_label), CellMapImage):
                        is_empty = False
                        break
            if is_empty:
                shape = tuple(map(int, array_info["shape"]))
                target_array[label] = EmptyImage(
                    label, array_info["scale"], shape, empty_store  # type: ignore
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
            value_transform: Callable | None = None
            if isinstance(self.target_value_transforms, dict):
                value_transform = self.target_value_transforms.get(label)
            elif isinstance(self.target_value_transforms, list):
                value_transform = self.target_value_transforms[i]
            elif callable(self.target_value_transforms):
                value_transform = self.target_value_transforms

            array = CellMapImage(
                self.target_path_str.format(label=label),
                label,
                array_info["scale"],  # type: ignore
                tuple(map(int, array_info["shape"])),
                value_transform=value_transform,
                context=self.context,
                pad=self.pad,
                pad_value=self.empty_value,
                interpolation="nearest",
                device=self._device,
            )
            if not self.has_data and not self.force_has_data:
                self.has_data = array.class_counts > 0
            logger.debug(f"{str(self)} has data: {self.has_data}")
        else:
            if (
                self.class_relation_dict is not None
                and label in self.class_relation_dict
            ):
                array = self.class_relation_dict[label]
            else:
                shape = tuple(map(int, array_info["shape"]))
                array = EmptyImage(
                    label, array_info["scale"], shape, empty_store, device=self._device  # type: ignore
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
        current_box: dict[str, list[float]] | None,
    ) -> dict[str, list[float]] | None:
        """Returns the intersection of the source and current boxes."""
        if source_box is None:
            return current_box
        if current_box is None:
            return {k: v[:] for k, v in source_box.items()}

        result_box = {k: v[:] for k, v in current_box.items()}
        for c, (start, stop) in source_box.items():
            if stop <= start:
                raise ValueError(f"Invalid box: start={start}, stop={stop}")
            result_box[c][0] = max(result_box[c][0], start)
            result_box[c][1] = min(result_box[c][1], stop)
        return result_box

    def verify(self) -> bool:
        """Verifies that the dataset is valid to draw samples from."""
        try:
            return len(self) > 0
        except Exception as e:
            logger.warning("Dataset verification failed: %s", e)
            return False

    def get_indices(self, chunk_size: Mapping[str, int]) -> Sequence[int]:
        """Returns the indices of the dataset that will tile the dataset according to the chunk_size."""
        # TODO: ADD TEST
        # Get padding per axis
        indices_dict = {}
        for c, size in chunk_size.items():
            if size <= 0:
                indices_dict[c] = np.array([0], dtype=int)
            else:
                indices_dict[c] = np.arange(
                    0, self.sampling_box_shape[c], size, dtype=int
                )

        indices = []
        shape_values = [self.sampling_box_shape[c] for c in self.axis_order]
        for i in np.ndindex(*[len(indices_dict[c]) for c in self.axis_order]):
            index = [indices_dict[c][j] for c, j in zip(self.axis_order, i)]
            index = np.ravel_multi_index(index, shape_values)
            indices.append(index)
        return indices

    def to(
        self, device: str | torch.device, non_blocking: bool = True
    ) -> "CellMapDataset":
        """Sets the device for the dataset."""
        self._device = torch.device(device)
        device_str = str(self._device)
        all_sources = list(self.input_sources.values()) + list(
            self.target_sources.values()
        )
        for source in all_sources:
            if isinstance(source, dict):
                for sub_source in source.values():
                    if hasattr(sub_source, "to"):
                        sub_source.to(device_str, non_blocking=non_blocking)
            elif hasattr(source, "to"):
                source.to(device_str, non_blocking=non_blocking)
        return self

    def generate_spatial_transforms(self) -> Optional[Mapping[str, Any]]:
        """
        Generates random spatial transforms for training.

        Available transforms:
        - "mirror": {"axes": {"x": 0.5, "y": 0.5}}
        - "transpose": {"axes": ["x", "z"]}
        - "rotate": {"axes": {"z": [-90, 90]}}
        """
        if not self.is_train or self.spatial_transforms is None:
            return None

        spatial_transforms: dict[str, Any] = {}
        for transform, params in self.spatial_transforms.items():
            if transform == "mirror":
                mirrored_axes = [
                    axis
                    for axis, prob in params["axes"].items()
                    if torch.rand(1, generator=self._rng).item() < prob
                ]
                if mirrored_axes:
                    spatial_transforms[transform] = mirrored_axes
            elif transform == "transpose":
                axes = {axis: i for i, axis in enumerate(self.axis_order)}
                permuted_axes = [axes[a] for a in params["axes"]]
                permuted_indices = torch.randperm(
                    len(permuted_axes), generator=self._rng
                )
                shuffled_axes = [permuted_axes[i] for i in permuted_indices]
                axes.update(
                    {axis: shuffled_axes[i] for i, axis in enumerate(params["axes"])}
                )
                spatial_transforms[transform] = axes
            elif transform == "rotate":
                rotated_axes = {}
                for axis, limits in params["axes"].items():
                    angle = (
                        torch.rand(1, generator=self._rng).item()
                        * (limits[1] - limits[0])
                        + limits[0]
                    )
                    rotated_axes[axis] = angle
                if rotated_axes:
                    spatial_transforms[transform] = rotated_axes
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

    def reset_arrays(self, array_type: str = "target") -> None:
        """Resets the specified arrays for the dataset."""
        if array_type.lower() == "input":
            self.input_sources = {}
            for array_name, array_info in self.input_arrays.items():
                self.input_sources[array_name] = CellMapImage(
                    self.raw_path,
                    "raw",
                    array_info["scale"],  # type: ignore
                    tuple(map(int, array_info["shape"])),
                    value_transform=self.raw_value_transforms,
                    context=self.context,
                    pad=self.pad,
                    pad_value=0,
                )
        elif array_type.lower() == "target":
            self.target_sources = {}
            self.has_data = False
            for array_name, array_info in self.target_arrays.items():
                self.target_sources[array_name] = self.get_target_array(array_info)
        else:
            raise ValueError(f"Unknown dataset array type: {array_type}")

    def get_random_subset_sampler(
        self, num_samples: int, rng: Optional[torch.Generator] = None, **kwargs: Any
    ) -> MutableSubsetRandomSampler:
        """
        Returns a random sampler that yields exactly `num_samples` indices from this subset.
        - If `num_samples` ≤ total number of available indices, samples without replacement.
        - If `num_samples` > total number of available indices, samples with replacement using repeated shuffles to minimize duplicates.
        """
        indices_generator = functools.partial(
            self.get_random_subset_indices, num_samples, rng, **kwargs
        )

        return MutableSubsetRandomSampler(indices_generator)

    def get_random_subset_indices(
        self, num_samples: int, rng: Optional[torch.Generator] = None, **kwargs: Any
    ) -> Sequence[int]:
        inds = min_redundant_inds(len(self), num_samples, rng=rng)
        return inds.tolist()

    def get_subset_random_sampler(
        self,
        num_samples: int,
        weighted: bool = False,
        rng: Optional[torch.Generator] = None,
    ) -> MutableSubsetRandomSampler:
        """
        Returns a subset random sampler for the dataset.

        Args:
        ----
            num_samples: The number of samples.
            weighted: Whether to use weighted sampling.
            rng: The random number generator.

        Returns:
        -------
            A subset random sampler.
        """
        if num_samples is None:
            num_samples = len(self) * 2

        if weighted:
            raise NotImplementedError("Weighted sampling is not yet implemented.")
        else:
            indices_generator = lambda: min_redundant_inds(
                len(self), num_samples, rng=rng
            )

        return MutableSubsetRandomSampler(indices_generator, rng=rng)

    @staticmethod
    def empty() -> "CellMapDataset":
        """Creates an empty dataset."""
        # Directly instantiate to bypass __new__ logic
        instance = super(CellMapDataset, CellMapDataset).__new__(CellMapDataset)
        instance.__init__("", "", [], {}, {}, force_has_data=False)
        instance.has_data = False
        # Set cached_property value directly in __dict__ to bypass computation
        instance.__dict__["sampling_box_shape"] = {c: 0 for c in instance.axis_order}
        return instance
