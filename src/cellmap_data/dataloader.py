import functools
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
import logging

import multiprocessing as mp
import sys

from .mutable_sampler import MutableSubsetRandomSampler
from .subdataset import CellMapSubset
from .dataset import CellMapDataset
from .multidataset import CellMapMultiDataset
from .dataset_writer import CellMapDatasetWriter
from typing import Callable, Optional, Sequence

logger = logging.getLogger(__name__)

# Stream optimization settings
MIN_BATCH_MEMORY_FOR_STREAMS_MB = float(
    os.environ.get("MIN_BATCH_MEMORY_FOR_STREAMS_MB", 100.0)
)
MAX_CONCURRENT_CUDA_STREAMS = int(os.environ.get("MAX_CONCURRENT_CUDA_STREAMS", 8))


class CellMapDataLoader:
    """
    Utility class to create a DataLoader for a CellMapDataset or CellMapMultiDataset.

    Attributes:
        dataset (CellMapMultiDataset | CellMapDataset | CellMapSubset): The dataset to load.
        classes (Iterable[str]): The classes to load.
        batch_size (int): The batch size.
        num_workers (int): The number of workers to use.
        weighted_sampler (bool): Whether to use a weighted sampler.
        sampler (Sampler | Callable | None): The sampler to use.
        is_train (bool): Whether the data is for training and thus should be shuffled.
        rng (Optional[torch.Generator]): The random number generator to use.
        loader (DataLoader): The PyTorch DataLoader.
        default_kwargs (dict): The default arguments to pass to the PyTorch DataLoader.

    Methods:
        refresh: If the sampler is a Callable, refresh the DataLoader with the current sampler.
        collate_fn: Combine a list of dictionaries from different sources into a single dictionary for output.

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
        """
        Initialize the CellMapDataLoader

        Args:
            dataset (CellMapMultiDataset | CellMapDataset | CellMapSubset): The dataset to load.
            classes (Iterable[str]): The classes to load.
            batch_size (int): The batch size.
            num_workers (int): The number of workers to use.
            weighted_sampler (bool): Whether to use a weighted sampler. Defaults to False.
            sampler (Sampler | Callable | None): The sampler to use.
            is_train (bool): Whether the data is for training and thus should be shuffled.
            rng (Optional[torch.Generator]): The random number generator to use.
            device (Optional[str | torch.device]): The device to use. Defaults to "cuda" or "mps" if available, else "cpu".
            iterations_per_epoch (Optional[int]): Number of iterations per epoch, only necessary when a subset is used with a weighted sampler (i.e. if total samples in the dataset are > 2^24).
            `**kwargs`: Additional arguments to pass to the DataLoader.

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
                    raise ValueError(
                        f"Input array info for {array_name} must include 'shape'"
                    )
                # Input arrays: batch_size * elements_per_sample
                total_elements += self.batch_size * np.prod(array_info["shape"])

            # Calculate target array elements
            for array_name, array_info in target_arrays.items():
                if "shape" not in array_info:
                    raise ValueError(
                        f"Target array info for {array_name} must include 'shape'"
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
