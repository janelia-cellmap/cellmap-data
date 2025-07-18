import functools
import os
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

    # Streaming settings
    MIN_ELEMENTS_FOR_STREAMS = os.environ.get("MIN_ELEMENTS_FOR_STREAMS", 1_000_000)
    MAX_CONCURRENT_CUDA_STREAMS = int(os.environ.get(
        "MAX_CONCURRENT_CUDA_STREAMS", 8
    ))
    GPU_MEMORY_LOG_THRESHOLD_GB = float(os.environ.get("GPU_MEMORY_LOG_THRESHOLD_GB", 2.0))

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

    def collate_fn(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        """Combine a list of dictionaries from different sources into a single dictionary for output."""
        outputs = {}
        for b in batch:
            for key, value in b.items():
                if key not in outputs:
                    outputs[key] = []
                outputs[key].append(value)

        # Determine if CUDA streams would be beneficial
        use_streams = self._should_use_cuda_streams(outputs)

        if use_streams:
            # Limit number of streams to avoid resource exhaustion
            data_keys = [key for key in outputs if key != "__metadata__"]
            max_streams = min(
                len(data_keys), self.MAX_CONCURRENT_CUDA_STREAMS
            )  # Limit to MAX_CONCURRENT_STREAMS

            # Create streams with proper error handling
            streams = []
            try:
                streams = [torch.cuda.Stream() for _ in range(max_streams)]
            except RuntimeError as e:
                # Fallback to sequential processing if stream creation fails
                use_streams = False

            if use_streams:
                # Track which keys are assigned to which streams to avoid conflicts
                key_to_stream = {}
                stream_assignments = {i: [] for i in range(max_streams)}

                # Assign keys to streams in round-robin fashion
                for i, key in enumerate(data_keys):
                    stream_idx = i % max_streams
                    key_to_stream[key] = stream_idx
                    stream_assignments[stream_idx].append(key)

                # Process each stream's keys sequentially within the stream
                for stream_idx, assigned_keys in stream_assignments.items():
                    if assigned_keys:  # Only use streams that have keys assigned
                        stream = streams[stream_idx]
                        with torch.cuda.stream(stream):
                            for key in assigned_keys:
                                if key in outputs:
                                    outputs[key] = torch.stack(outputs[key]).to(
                                        self.device, non_blocking=True
                                    )

                # Create synchronization barrier - wait for all streams to complete
                for stream in streams:
                    stream.synchronize()

        # Log memory usage for large batches (minimal overhead)
        if use_streams and torch.cuda.is_available():
            try:
                memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                if memory_allocated > self.GPU_MEMORY_LOG_THRESHOLD_GB:  # Only log if > 2GB
                    logger.debug(
                        f"Large batch GPU memory usage: {memory_allocated:.2f} GB"
                    )
            except Exception:
                pass  # Ignore memory monitoring failures

        if not use_streams:
            # Sequential processing for small/single tensors
            for key, value in outputs.items():
                if key != "__metadata__":
                    outputs[key] = torch.stack(value).to(self.device, non_blocking=True)
        return outputs

    def _should_use_cuda_streams(self, outputs: dict) -> bool:
        """Determine if CUDA streams would be beneficial for this batch."""
        if not (str(self.device).startswith("cuda") and torch.cuda.is_available()):
            return False

        # Count data keys (excluding metadata)
        data_keys = [key for key in outputs if key != "__metadata__"]
        num_keys = len(data_keys)

        # Only use streams if we have multiple keys
        if num_keys < 2:
            return False

        # Estimate tensor sizes to determine if transfer is worth parallelizing
        total_elements = 0
        # Use streams if we have multiple keys and sufficient data volume
        # Threshold: at least 1M elements total to justify stream overhead
        min_elements_threshold = self.MIN_ELEMENTS_FOR_STREAMS
        return num_keys >= 2 and total_elements >= min_elements_threshold
                    batch_elements = sample_tensor.numel() * len(outputs[key])
                    total_elements += batch_elements

        # Use streams if we have multiple keys and sufficient data volume
        # Threshold: at least 1M elements total to justify stream overhead
        min_elements_threshold = 1_000_000
        return num_keys >= 2 and total_elements >= min_elements_threshold
