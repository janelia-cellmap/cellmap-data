import functools
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

import multiprocessing as mp
import sys

from .mutable_sampler import MutableSubsetRandomSampler
from .subdataset import CellMapSubset
from .dataset import CellMapDataset
from .multidataset import CellMapMultiDataset
from .dataset_writer import CellMapDatasetWriter
from .utils.logging_config import get_logger
from .utils.error_handling import ValidationError, ErrorMessages
from typing import Callable, Optional, Sequence

logger = get_logger("dataloader")

# Stream optimization settings
MIN_BATCH_MEMORY_FOR_STREAMS_MB = float(
    os.environ.get("MIN_BATCH_MEMORY_FOR_STREAMS_MB", 100.0)
)
MAX_CONCURRENT_CUDA_STREAMS = int(os.environ.get("MAX_CONCURRENT_CUDA_STREAMS", 8))


class CellMapDataLoader:
    """Optimized DataLoader wrapper for CellMap datasets with advanced batching and streaming.

    This class provides an enhanced DataLoader interface specifically designed for
    CellMap datasets with support for weighted sampling, CUDA stream optimization,
    automatic device placement, and efficient batch processing. It wraps PyTorch's
    DataLoader with CellMap-specific optimizations and memory management.

    The class automatically handles device placement, stream optimization for large
    batches, multiprocessing configuration, and provides specialized collation
    functions for CellMap data structures. It supports both training and inference
    workflows with configurable sampling strategies.

    Attributes
    ----------
    dataset : CellMapDataset or CellMapMultiDataset or CellMapSubset or CellMapDatasetWriter
        The underlying dataset providing data samples.
    classes : sequence of str
        List of segmentation classes being loaded.
    batch_size : int
        Number of samples per batch.
    num_workers : int
        Number of worker processes for data loading.
    device : str or torch.device
        Target device for batch tensors.
    loader : torch.utils.data.DataLoader
        The underlying PyTorch DataLoader instance.
    sampler : Sampler or callable or None
        Custom sampling strategy for batch generation.
    is_train : bool
        Whether this loader is used for training (affects shuffling).

    Methods
    -------
    __iter__()
        Iterate over batches with optimized loading and streaming.
    __len__()
        Return number of batches per epoch.
    to(device)
        Move dataset operations to specified device.
    refresh()
        Refresh sampler state for dynamic sampling strategies.
    collate_fn(batch)
        Combine individual samples into optimized batch tensors.

    Examples
    --------
    Basic training loader:

    >>> dataset = CellMapDataset(...)
    >>> loader = CellMapDataLoader(
    ...     dataset=dataset,
    ...     batch_size=4,
    ...     num_workers=2,
    ...     is_train=True,
    ...     device="cuda"
    ... )
    >>> for batch in loader:
    ...     # Process batch
    ...     print(batch.keys())
    dict_keys(['raw', 'labels'])

    With weighted sampling for imbalanced classes:

    >>> loader = CellMapDataLoader(
    ...     dataset=dataset,
    ...     batch_size=8,
    ...     weighted_sampler=True,
    ...     num_workers=4,
    ...     is_train=True
    ... )

    Multi-dataset loader with custom sampler:

    >>> from torch.utils.data import RandomSampler
    >>> multidataset = CellMapMultiDataset([dataset1, dataset2])
    >>> custom_sampler = RandomSampler(multidataset)
    >>> loader = CellMapDataLoader(
    ...     dataset=multidataset,
    ...     sampler=custom_sampler,
    ...     batch_size=2,
    ...     num_workers=8
    ... )

    Notes
    -----
    The loader automatically optimizes memory usage and streaming based on batch
    size and available hardware. CUDA streams are enabled for large batches to
    overlap data transfer and computation.

    Multiprocessing context is automatically configured for optimal performance
    across different platforms (spawn on Windows, forkserver on Linux/macOS).

    For very large datasets with weighted sampling, specify iterations_per_epoch
    to avoid memory issues with sample index generation.

    See Also
    --------
    CellMapDataset : Core dataset implementation
    CellMapMultiDataset : Multi-dataset training support
    MutableSubsetRandomSampler : Dynamic subset sampling
    """

    def __init__(
        self,
        dataset: (
            CellMapMultiDataset | CellMapDataset | CellMapSubset | CellMapDatasetWriter
        ),
        classes: Sequence[str] | None = None,
        batch_size: int = 1,
        num_workers: int = 0,
        weighted_sampler: bool = False,
        sampler: Sampler | Callable | None = None,
        is_train: bool = True,
        rng: Optional[torch.Generator] = None,
        device: Optional[str | torch.device] = None,
        iterations_per_epoch: Optional[int] = None,
        **kwargs,
    ):
        """Initialize CellMapDataLoader with optimized configuration for batch loading.

        Creates a DataLoader wrapper with CellMap-specific optimizations including
        CUDA stream management, weighted sampling for imbalanced datasets, and
        automatic device placement with multiprocessing support.

        Args:
            dataset: The dataset instance providing data samples. Supports single datasets,
                multi-dataset training, subsets, and dataset writers.
            classes: List of segmentation classes to load. Defaults to None.
                If None, uses all classes defined in the dataset.
            batch_size: Number of samples per batch. Defaults to 1.
                Larger batches enable CUDA stream optimization for GPU acceleration.
            num_workers: Number of worker processes for parallel data loading. Defaults to 0.
                Set to 0 for single-threaded loading, >0 for multiprocessing.
            weighted_sampler: Whether to use weighted sampling for class balancing. Defaults to False.
                Automatically weights samples based on class frequency in dataset.
            sampler: Custom sampling strategy for batch generation. Defaults to None.
                Can be PyTorch Sampler instance or callable returning sampler.
            is_train: Whether this loader is for training (enables shuffling). Defaults to True.
                Training mode enables data shuffling and augmentation-friendly settings.
            rng: Random number generator for reproducible sampling. Defaults to None.
                If None, uses default PyTorch random state.
            device: Target device for tensor operations. Defaults to None.
                If None, automatically selects: "cuda" > "mps" > "cpu".
            iterations_per_epoch: Number of iterations per epoch for large datasets with weighted sampling.
                Defaults to None. Required when dataset size exceeds 2^24 samples.
            **kwargs: Additional keyword arguments passed to PyTorch DataLoader constructor.
                Common options include pin_memory, persistent_workers, prefetch_factor.

        Raises:
            ValidationError: If dataset type is not supported or configuration is invalid.
            RuntimeError: If multiprocessing context cannot be initialized.
            MemoryError: If weighted sampling requires too much memory without iterations_per_epoch limit.
        """
        self.dataset = dataset
        self.classes = classes if classes is not None else dataset.classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.weighted_sampler = weighted_sampler
        self.sampler = sampler
        self.is_train = is_train
        self.rng = rng
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.iterations_per_epoch = iterations_per_epoch

        # Initialize stream optimization settings
        self._use_streams = None  # Determined once, cached
        self._streams = None  # Created once, reused
        self._stream_assignments = None  # Cached key assignments
        if num_workers == 0:
            self.dataset.to(device, non_blocking=True)
            mp_kwargs = {}
        else:
            if (
                sys.platform.startswith("win")
                or "forkserver" not in mp.get_all_start_methods()
            ):
                ctx = "spawn"
            else:
                ctx = "forkserver"
            torch.multiprocessing.set_start_method(ctx, force=True)
            torch.multiprocessing.set_sharing_strategy("file_system")
            mp_kwargs = {
                "num_workers": num_workers,
                "multiprocessing_context": ctx,
                "persistent_workers": True,
                "pin_memory": True,
            }
        if self.sampler is None:
            if iterations_per_epoch is not None or (
                weighted_sampler and len(self.dataset) > 2**24
            ):
                assert (
                    iterations_per_epoch is not None
                ), "If the dataset has more than 2^24 samples, iterations_per_epoch must be specified to allow for subset selection. In between epochs, run `refresh()` to update the sampler."
                assert not isinstance(
                    self.dataset, CellMapDatasetWriter
                ), "CellMapDatasetWriter does not support random sampling."
                self.sampler = self.dataset.get_subset_random_sampler(
                    num_samples=iterations_per_epoch * batch_size,
                    weighted=weighted_sampler,
                    rng=self.rng,
                )
            elif weighted_sampler and isinstance(self.dataset, CellMapMultiDataset):
                self.sampler = self.dataset.get_weighted_sampler(
                    self.batch_size, self.rng
                )

        self.default_kwargs = mp_kwargs
        self.default_kwargs.update(kwargs)
        self.refresh()

    def __getitem__(self, indices: Sequence[int]) -> dict:
        """Get an item from the DataLoader."""
        if isinstance(indices, int):
            indices = [indices]
        return self.collate_fn([self.loader.dataset[index] for index in indices])

    def to(self, device: str | torch.device, non_blocking: bool = True):
        """Move the dataset to the specified device."""
        self.dataset.to(device, non_blocking=non_blocking)
        self.device = device
        # Reset stream optimization for new device
        self._use_streams = None
        self._streams = None
        self._stream_assignments = None

    def refresh(self):
        """If the sampler is a Callable, refresh the DataLoader with the current sampler."""
        if isinstance(self.sampler, MutableSubsetRandomSampler):
            self.sampler.refresh()
            if not hasattr(self, "loader"):
                kwargs = self.default_kwargs.copy()
                pin_memory = (
                    (self.device != "cpu" and self.num_workers > 0)
                    or kwargs.get("pin_memory", False)
                    or "pin_memory_device" in kwargs
                )
                kwargs.update(
                    {
                        "dataset": self.dataset,
                        "batch_size": self.batch_size,
                        "num_workers": self.num_workers,
                        "pin_memory": pin_memory,
                        "collate_fn": self.collate_fn,
                        "sampler": self.sampler,
                    }
                )
                self.loader = DataLoader(**kwargs)
        else:
            kwargs = self.default_kwargs.copy()
            kwargs.update(
                {
                    "dataset": self.dataset,
                    "batch_size": self.batch_size,
                    "num_workers": self.num_workers,
                    "pin_memory": (self.device != "cpu" and self.num_workers > 0)
                    or self.default_kwargs.get("pin_memory", False),
                    "collate_fn": self.collate_fn,
                }
            )
            if self.sampler is not None:
                if isinstance(self.sampler, Callable):
                    kwargs["sampler"] = self.sampler()
                else:
                    kwargs["sampler"] = self.sampler
            elif self.is_train:
                kwargs["shuffle"] = True
            else:
                kwargs["shuffle"] = False
            self.loader = DataLoader(**kwargs)

    def _calculate_batch_memory_mb(self) -> float:
        """Calculate the expected memory usage for a batch in MB."""
        try:
            input_arrays = getattr(self.dataset, "input_arrays", {})
            target_arrays = getattr(self.dataset, "target_arrays", {})

            if not input_arrays and not target_arrays:
                return 0.0

            total_elements = 0

            # Calculate input array elements
            for array_name, array_info in input_arrays.items():
                if "shape" not in array_info:
                    raise ValidationError(
                        ErrorMessages.ARRAY_INFO_MISSING_KEY,
                        array_name=array_name,
                        key="shape",
                    )
                # Input arrays: batch_size * elements_per_sample
                total_elements += self.batch_size * np.prod(array_info["shape"])

            # Calculate target array elements
            for array_name, array_info in target_arrays.items():
                if "shape" not in array_info:
                    raise ValidationError(
                        ErrorMessages.ARRAY_INFO_MISSING_KEY,
                        array_name=array_name,
                        key="shape",
                    )
                # Target arrays: batch_size * elements_per_sample * num_classes
                elements_per_sample = np.prod(array_info["shape"])
                num_classes = len(self.classes) if self.classes else 1
                total_elements += self.batch_size * elements_per_sample * num_classes

            # Convert to MB (assume float32 = 4 bytes per element)
            bytes_total = total_elements * 4  # float32
            mb_total = bytes_total / (1024 * 1024)  # Convert bytes to MB
            return mb_total

        except (AttributeError, KeyError, TypeError) as e:
            # Fallback: if we can't calculate, return 0 to disable memory-based decision
            logger.debug(f"Could not calculate batch memory size: {e}")
            return 0.0

    def _initialize_stream_optimization(self, sample_batch: dict) -> None:
        """Initialize stream optimization settings once based on dataset characteristics."""
        if self._use_streams is not None:
            return  # Already initialized

        # Calculate expected batch memory usage
        batch_memory_mb = self._calculate_batch_memory_mb()

        # Determine if streams should be used based on static conditions
        self._use_streams = (
            str(self.device).startswith("cuda")
            and torch.cuda.is_available()
            and batch_memory_mb >= MIN_BATCH_MEMORY_FOR_STREAMS_MB
        )

        if not self._use_streams:
            if batch_memory_mb > 0:
                logger.debug(
                    f"CUDA streams disabled: batch_size={self.batch_size}, "
                    f"memory={batch_memory_mb:.1f}MB (min: {MIN_BATCH_MEMORY_FOR_STREAMS_MB}MB)"
                )
            return

        # Get data keys from sample batch
        data_keys = [key for key in sample_batch if key != "__metadata__"]
        num_keys = len(data_keys)

        # Create persistent streams with error handling
        max_streams = min(num_keys, MAX_CONCURRENT_CUDA_STREAMS)
        try:
            self._streams = [torch.cuda.Stream() for _ in range(max_streams)]

            # Pre-compute stream assignments for efficiency
            self._stream_assignments = {}
            for i, key in enumerate(data_keys):
                stream_idx = i % max_streams
                self._stream_assignments[key] = stream_idx

            logger.debug(
                f"CUDA streams enabled: {max_streams} streams, "
                f"batch_size={self.batch_size}, memory={batch_memory_mb:.1f}MB"
            )

        except RuntimeError as e:
            logger.warning(
                f"Failed to create CUDA streams, falling back to sequential: {e}"
            )
            self._use_streams = False
            self._streams = None
            self._stream_assignments = None

    def collate_fn(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        """Combine a list of dictionaries from different sources into a single dictionary for output."""
        outputs = {}
        for b in batch:
            for key, value in b.items():
                if key not in outputs:
                    outputs[key] = []
                outputs[key].append(value)

        # Initialize stream optimization on first batch
        self._initialize_stream_optimization(outputs)

        if (
            self._use_streams
            and self._streams is not None
            and self._stream_assignments is not None
        ):
            # Use pre-allocated streams with cached assignments
            for key, value in outputs.items():
                if key != "__metadata__":
                    stream_idx = self._stream_assignments.get(key, 0)
                    stream = self._streams[stream_idx]
                    with torch.cuda.stream(stream):
                        outputs[key] = torch.stack(value).to(
                            self.device, non_blocking=True
                        )

            # Synchronization barrier
            for stream in self._streams:
                stream.synchronize()
        else:
            # Sequential processing
            for key, value in outputs.items():
                if key != "__metadata__":
                    outputs[key] = torch.stack(value).to(self.device, non_blocking=True)

        return outputs
