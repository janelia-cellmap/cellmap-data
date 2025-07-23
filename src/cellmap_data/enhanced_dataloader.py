"""
Enhanced CellMapDataLoader with plugin system support.
Extends the original dataloader with extensible plugin hooks.
"""

import functools
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
import logging
import multiprocessing as mp
import sys
from typing import Callable, Optional, Sequence, Dict, Any

from .mutable_sampler import MutableSubsetRandomSampler
from .subdataset import CellMapSubset
from .dataset import CellMapDataset
from .multidataset import CellMapMultiDataset
from .dataset_writer import CellMapDatasetWriter
from .plugins import (
    PluginManager,
    PrefetchPlugin,
    AugmentationPlugin,
    DeviceTransferPlugin,
    MemoryOptimizationPlugin,
)
from .device.device_manager import DeviceManager

logger = logging.getLogger(__name__)

# Stream optimization settings
MIN_BATCH_MEMORY_FOR_STREAMS_MB = float(
    os.environ.get("MIN_BATCH_MEMORY_FOR_STREAMS_MB", 100.0)
)
MAX_CONCURRENT_CUDA_STREAMS = int(os.environ.get("MAX_CONCURRENT_CUDA_STREAMS", 8))


class EnhancedCellMapDataLoader:
    """
    Enhanced CellMapDataLoader with plugin system support.

    Provides extensible hooks for data processing pipeline stages:
    - pre_batch: Before batch creation
    - post_sample: After individual sample loading
    - post_collate: After batch collation
    - post_batch: After batch processing

    Built-in plugins:
    - PrefetchPlugin: Asynchronous data prefetching
    - AugmentationPlugin: Data augmentation pipeline
    - DeviceTransferPlugin: Framework-aware device transfer
    - MemoryOptimizationPlugin: Memory pooling and optimization
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
        # Plugin system options
        enable_plugins: bool = True,
        enable_prefetch: bool = True,
        enable_augmentation: bool = True,
        enable_device_transfer: bool = True,
        enable_memory_optimization: bool = True,
        max_prefetch: int = 8,
        transforms: Optional[list] = None,
        **kwargs,
    ):
        """
        Initialize Enhanced CellMapDataLoader with plugin system.

        Args:
            dataset: Dataset to load from
            classes: Classes to load (defaults to dataset.classes)
            batch_size: Batch size for loading
            num_workers: Number of worker processes
            weighted_sampler: Use weighted sampling
            sampler: Custom sampler
            is_train: Training mode (affects shuffling)
            rng: Random number generator
            device: Target device for tensors
            iterations_per_epoch: Iterations per epoch for large datasets
            enable_plugins: Enable the plugin system
            enable_prefetch: Enable prefetching plugin
            enable_augmentation: Enable augmentation plugin
            enable_device_transfer: Enable device transfer plugin
            enable_memory_optimization: Enable memory optimization plugin
            max_prefetch: Maximum items to prefetch
            transforms: List of augmentation transforms
            **kwargs: Additional DataLoader arguments
        """
        # Core attributes
        self.dataset = dataset
        self.classes = classes if classes is not None else dataset.classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.weighted_sampler = weighted_sampler
        self.sampler = sampler
        self.is_train = is_train
        self.rng = rng
        self.iterations_per_epoch = iterations_per_epoch

        # Device setup
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        # Initialize DeviceManager
        self.device_manager = DeviceManager(device=device)

        # Plugin system setup
        self.plugin_manager = PluginManager() if enable_plugins else None
        self._setup_plugins(
            enable_prefetch=enable_prefetch,
            enable_augmentation=enable_augmentation,
            enable_device_transfer=enable_device_transfer,
            enable_memory_optimization=enable_memory_optimization,
            max_prefetch=max_prefetch,
            transforms=transforms,
        )

        # Stream optimization settings
        self._use_streams = None
        self._streams = None
        self._stream_assignments = None

        # Multiprocessing setup
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

        # Sampler setup
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

    def _setup_plugins(
        self,
        enable_prefetch: bool,
        enable_augmentation: bool,
        enable_device_transfer: bool,
        enable_memory_optimization: bool,
        max_prefetch: int,
        transforms: Optional[list],
    ):
        """Setup built-in plugins based on configuration."""
        if self.plugin_manager is None:
            return

        if enable_prefetch:
            prefetch_plugin = PrefetchPlugin(max_prefetch=max_prefetch)
            self.plugin_manager.register_plugin(prefetch_plugin)

        if enable_augmentation:
            augmentation_plugin = AugmentationPlugin(transforms=transforms)
            self.plugin_manager.register_plugin(augmentation_plugin)

        if enable_device_transfer:
            device_plugin = DeviceTransferPlugin(device_manager=self.device_manager)
            self.plugin_manager.register_plugin(device_plugin)

        if enable_memory_optimization:
            memory_plugin = MemoryOptimizationPlugin(device_manager=self.device_manager)
            self.plugin_manager.register_plugin(memory_plugin)

    def add_plugin(self, plugin):
        """Add a custom plugin to the loader."""
        if self.plugin_manager is not None:
            self.plugin_manager.register_plugin(plugin)

    def remove_plugin(self, plugin_name: str):
        """Remove a plugin by name."""
        if self.plugin_manager is not None:
            self.plugin_manager.unregister_plugin(plugin_name)

    def get_plugin(self, plugin_name: str):
        """Get a plugin by name."""
        if self.plugin_manager is not None:
            return self.plugin_manager.get_plugin(plugin_name)
        return None

    def __getitem__(self, indices: Sequence[int]) -> dict:
        """Get an item from the DataLoader with plugin hooks."""
        if isinstance(indices, int):
            indices = [indices]

        # Pre-batch hook
        if self.plugin_manager:
            self.plugin_manager.execute_hook("pre_batch", self.dataset, indices)

        # Load samples with post_sample hooks
        samples = []
        for index in indices:
            sample = self.loader.dataset[index]

            # Post-sample hook (for augmentation)
            if self.plugin_manager:
                sample = self.plugin_manager.execute_hook("post_sample", sample)

            samples.append(sample)

        # Collate and apply post-collate hooks
        batch = self.collate_fn(samples)

        # Post-batch hook
        if self.plugin_manager:
            self.plugin_manager.execute_hook("post_batch")

        return batch

    def to(self, device: str | torch.device, non_blocking: bool = True):
        """Move the dataset to the specified device."""
        self.dataset.to(device, non_blocking=non_blocking)
        self.device = device
        self.device_manager.device = device

        # Reset stream optimization for new device
        self._use_streams = None
        self._streams = None
        self._stream_assignments = None

    def refresh(self):
        """Refresh the DataLoader with current sampler."""
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
                total_elements += self.batch_size * np.prod(array_info["shape"])

            # Calculate target array elements
            for array_name, array_info in target_arrays.items():
                if "shape" not in array_info:
                    raise ValueError(
                        f"Target array info for {array_name} must include 'shape'"
                    )
                elements_per_sample = np.prod(array_info["shape"])
                num_classes = len(self.classes) if self.classes else 1
                total_elements += self.batch_size * elements_per_sample * num_classes

            # Convert to MB (assume float32 = 4 bytes per element)
            bytes_total = total_elements * 4
            mb_total = bytes_total / (1024 * 1024)
            return mb_total

        except (AttributeError, KeyError, TypeError) as e:
            logger.debug(f"Could not calculate batch memory size: {e}")
            return 0.0

    def _initialize_stream_optimization(self, sample_batch: dict) -> None:
        """Initialize stream optimization settings once based on dataset characteristics."""
        if self._use_streams is not None:
            return

        batch_memory_mb = self._calculate_batch_memory_mb()

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

        data_keys = [key for key in sample_batch if key != "__metadata__"]
        num_keys = len(data_keys)

        max_streams = min(num_keys, MAX_CONCURRENT_CUDA_STREAMS)
        try:
            self._streams = [torch.cuda.Stream() for _ in range(max_streams)]

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
        """Combine a list of dictionaries with plugin hooks for optimization."""
        outputs = {}
        for b in batch:
            for key, value in b.items():
                if key not in outputs:
                    outputs[key] = []
                outputs[key].append(value)

        # Initialize stream optimization on first batch
        self._initialize_stream_optimization(outputs)

        # Standard collation
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

        # Post-collate hook (for device transfer and memory optimization)
        if self.plugin_manager:
            outputs = self.plugin_manager.execute_hook(
                "post_collate", outputs, device=self.device
            )

        return outputs


# Backward compatibility alias
CellMapDataLoaderWithPlugins = EnhancedCellMapDataLoader
