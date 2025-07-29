import functools
import os
import logging
import time
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
from .utils.memory_manager import (
    MemoryProfiler,
    MemoryOptimizedResourceManager,
    memory_profiled_execution,
    optimize_cuda_memory_allocation,
    get_memory_optimization_recommendations,
)
from .utils.memory_optimization import (
    MemoryOptimizationConfig,
    AdvancedMemoryManager,
    StreamingDataBuffer,
)
from .utils.thread_safety import (
    get_global_executor_manager,
    get_global_cuda_manager,
    thread_safe_execution,
    ThreadSafetyConfig,
    thread_safe,
)
from .utils.enhanced_thread_safety import (
    get_global_thread_safety_enhancements,
    EnhancedThreadSafetyConfig,
    enhanced_thread_safety_execution,
)
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
        enable_memory_profiling: bool = False,
        memory_limit_mb: Optional[float] = None,
        **kwargs,
    ):
        """Initialize CellMapDataLoader with optimized configuration for batch loading.

        Creates a DataLoader wrapper with CellMap-specific optimizations including
        CUDA stream management, weighted sampling for imbalanced datasets, automatic
        device placement, multiprocessing support, and enhanced memory management.

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
            enable_memory_profiling: Whether to enable memory usage profiling. Defaults to False.
                When enabled, tracks memory usage patterns for optimization analysis.
            memory_limit_mb: Optional memory limit in MB for resource allocation. Defaults to None.
                When set, prevents allocation that would exceed this limit.
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

        # Initialize memory management
        self.enable_memory_profiling = enable_memory_profiling
        self.memory_limit_mb = memory_limit_mb
        self.memory_profiler = MemoryProfiler() if enable_memory_profiling else None
        self.memory_manager = (
            MemoryOptimizedResourceManager(memory_limit_mb) if memory_limit_mb else None
        )

        # Initialize advanced memory management if needed
        if enable_memory_profiling or memory_limit_mb:
            memory_config = MemoryOptimizationConfig(
                max_memory_mb=memory_limit_mb,
                enable_gpu_memory_optimization=str(device).startswith("cuda"),
                enable_detailed_profiling=enable_memory_profiling,
            )
            self.advanced_memory_manager = AdvancedMemoryManager(memory_config)
        else:
            self.advanced_memory_manager = None

        # Initialize enhanced thread safety framework
        enhanced_thread_config = EnhancedThreadSafetyConfig(
            max_workers=num_workers if num_workers > 0 else 2,
            enable_concurrent_profiling=enable_memory_profiling,
            max_concurrent_operations=max(4, batch_size),
            enable_memory_aware_threading=True,
            integrate_with_memory_manager=self.advanced_memory_manager is not None,
            enable_cuda_memory_threading=str(device).startswith("cuda"),
            enable_detailed_profiling=enable_memory_profiling,
        )

        # Original thread safety manager for compatibility
        thread_safety_config = ThreadSafetyConfig(
            max_workers=num_workers if num_workers > 0 else 2,
            enable_concurrent_profiling=enable_memory_profiling,
            max_concurrent_operations=max(4, batch_size),
        )
        self._thread_safety_manager = get_global_executor_manager(thread_safety_config)
        self._cuda_stream_manager = get_global_cuda_manager(MAX_CONCURRENT_CUDA_STREAMS)

        # Enhanced thread safety manager
        self._enhanced_thread_safety = get_global_thread_safety_enhancements(
            enhanced_thread_config
        )

        # Register this dataloader for resource tracking
        self._thread_safety_manager.register_resource(f"dataloader_{id(self)}", self)

        # Register with enhanced concurrent loader manager
        self._enhanced_thread_safety.concurrent_loader_manager.register_dataloader(self)

        # Apply CUDA memory optimizations if using GPU
        if str(device).startswith("cuda"):
            optimize_cuda_memory_allocation()

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
        """Calculate the expected memory usage for a batch in MB with enhanced precision."""
        try:
            # Record memory profiling snapshot if enabled
            if self.memory_profiler:
                self.memory_profiler.record_snapshot("batch_memory_calculation_start")

            input_arrays = getattr(self.dataset, "input_arrays", {})
            target_arrays = getattr(self.dataset, "target_arrays", {})

            if not input_arrays and not target_arrays:
                return 0.0

            total_elements = 0
            memory_breakdown = {}

            # Calculate input array elements with detailed tracking
            for array_name, array_info in input_arrays.items():
                if "shape" not in array_info:
                    raise ValidationError(
                        ErrorMessages.ARRAY_INFO_MISSING_KEY,
                        array_name=array_name,
                        key="shape",
                    )
                # Input arrays: batch_size * elements_per_sample
                elements_per_sample = np.prod(array_info["shape"])
                batch_elements = self.batch_size * elements_per_sample
                total_elements += batch_elements
                memory_breakdown[f"input_{array_name}"] = batch_elements

            # Calculate target array elements with class multiplier
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
                batch_elements = self.batch_size * elements_per_sample * num_classes
                total_elements += batch_elements
                memory_breakdown[f"target_{array_name}"] = batch_elements

            # Enhanced memory calculation with dtype consideration
            dtype_info = getattr(self.dataset, "dtype", torch.float32)
            if isinstance(dtype_info, torch.dtype):
                element_size = torch.tensor(0, dtype=dtype_info).element_size()
            else:
                element_size = 4  # Default to float32 = 4 bytes

            # Calculate memory with overhead factor for PyTorch operations
            overhead_factor = 1.2  # 20% overhead for intermediate operations
            bytes_total = total_elements * element_size * overhead_factor
            mb_total = bytes_total / (1024 * 1024)

            if self.enable_memory_profiling:
                logger.debug(
                    f"Batch memory calculation: {mb_total:.1f}MB total "
                    f"({total_elements:,} elements * {element_size} bytes * {overhead_factor:.1f} overhead)"
                )
                logger.debug(f"Memory breakdown: {memory_breakdown}")

            return mb_total

        except (AttributeError, KeyError, TypeError) as e:
            # Fallback: if we can't calculate, return 0 to disable memory-based decision
            logger.debug(f"Could not calculate batch memory size: {e}")
            return 0.0
        finally:
            # Record completion snapshot if profiling enabled
            if self.memory_profiler:
                self.memory_profiler.record_snapshot("batch_memory_calculation_end")

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

    @thread_safe(lock_name="dataloader_collate")
    def collate_fn(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        """Combine a list of dictionaries from different sources into a single dictionary for output."""
        # Start memory profiling if enabled
        if self.memory_profiler:
            self.memory_profiler.record_snapshot("collate_start")

        outputs = {}
        for b in batch:
            for key, value in b.items():
                if key not in outputs:
                    outputs[key] = []
                outputs[key].append(value)

        # Initialize stream optimization on first batch
        self._initialize_stream_optimization(outputs)

        # Profile tensor allocation phase
        if self.memory_profiler:
            self.memory_profiler.record_snapshot("tensor_allocation_start")

        # Process all keys and convert lists to tensors
        try:
            # Use thread-safe CUDA stream context if available
            with self._cuda_stream_manager.cuda_stream_context() as cuda_stream:
                for key, value in outputs.items():
                    if key != "__metadata__":
                        # Skip non-tensor values
                        if not value or not isinstance(value[0], torch.Tensor):
                            continue

                        # Try advanced memory management if available and streams are enabled
                        tensor_created = False

                        if (
                            cuda_stream is not None
                            and self._use_streams
                            and self.memory_manager
                        ):
                            try:
                                # Advanced memory-managed allocation with CUDA streams
                                batch_shape = (len(value),) + value[0].shape
                                tensor_name = f"batch_{key}"
                                device_obj = (
                                    torch.device(self.device)
                                    if isinstance(self.device, str)
                                    else self.device
                                )

                                outputs[key] = self.memory_manager.allocate_tensor(
                                    shape=batch_shape,
                                    dtype=value[0].dtype,
                                    device=device_obj,
                                    name=tensor_name,
                                )

                                # Copy data into allocated tensor with non-blocking transfers
                                for i, v in enumerate(value):
                                    outputs[key][i] = v.to(
                                        self.device, non_blocking=True
                                    )

                                tensor_created = True
                                logger.debug(
                                    f"Created tensor '{key}' using memory manager with CUDA stream"
                                )

                            except Exception as e:
                                logger.debug(
                                    f"Memory manager allocation failed for '{key}': {e}"
                                )
                                # Fall back to standard allocation

                        # Standard tensor stacking (fallback or default path)
                        if not tensor_created:
                            outputs[key] = torch.stack(value).to(
                                self.device, non_blocking=True
                            )

        except Exception as e:
            # If any advanced processing fails, fall back to basic tensor stacking
            logger.warning(f"Advanced collate processing failed, using fallback: {e}")
            for key, value in outputs.items():
                if (
                    key != "__metadata__"
                    and value
                    and isinstance(value[0], torch.Tensor)
                ):
                    outputs[key] = torch.stack(value).to(self.device, non_blocking=True)

        # Complete memory profiling
        if self.memory_profiler:
            self.memory_profiler.record_snapshot("collate_end")

            # Generate memory recommendations if needed
            current_usage = self.memory_profiler._get_memory_info()
            recommendations = get_memory_optimization_recommendations(current_usage)

            if recommendations and self.enable_memory_profiling:
                logger.info("Memory optimization recommendations:")
                for rec in recommendations:
                    logger.info(f"  - {rec}")

        return outputs

    def get_memory_report(self) -> str:
        """Generate a comprehensive memory usage report.

        Returns:
            Formatted memory usage report including profiling data and resource allocation
        """
        if not self.memory_profiler:
            return (
                "Memory profiling is disabled. Enable with enable_memory_profiling=True"
            )

        report_lines = [
            "=== CellMapDataLoader Memory Report ===",
            f"Batch Size: {self.batch_size}",
            f"Device: {self.device}",
            (
                f"Memory Limit: {self.memory_limit_mb} MB"
                if self.memory_limit_mb
                else "Memory Limit: None"
            ),
            "",
            self.memory_profiler.generate_report(),
        ]

        # Add resource manager info if available
        if self.memory_manager:
            usage_summary = self.memory_manager.get_memory_usage_summary()
            report_lines.extend(
                [
                    "",
                    "=== Resource Manager Summary ===",
                    f"Tracked Resources: {usage_summary['total_tracked_resources']}",
                    f"Allocated Memory: {usage_summary['total_allocated_mb']:.1f} MB",
                    f"Resource Breakdown: {usage_summary['resource_breakdown']}",
                ]
            )

        return "\n".join(report_lines)

    def cleanup_memory_resources(self) -> int:
        """Clean up memory resources and perform garbage collection.

        Returns:
            Number of resources cleaned up
        """
        cleanup_count = 0

        if self.memory_manager:
            cleanup_count = self.memory_manager.cleanup_resources(force_gc=True)
            logger.info(f"Cleaned up {cleanup_count} tracked memory resources")

        # Clear CUDA cache if using GPU
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("Cleared CUDA memory cache")

        return cleanup_count

    def __del__(self):
        """Cleanup resources when dataloader is destroyed."""
        try:
            # Clean up memory manager resources
            if hasattr(self, "memory_manager") and self.memory_manager:
                self.memory_manager.cleanup_resources(force_gc=False)

            # Clean up thread safety resources
            self.cleanup_thread_safety_resources()

        except Exception as e:
            # Don't raise exceptions in destructor, but log for debugging
            try:
                logger.debug(f"Error during dataloader cleanup: {e}")
            except:
                pass  # Even logging might fail during shutdown

    # Add iterator support with memory profiling
    def __iter__(self):
        """Return an iterator with optional memory profiling."""
        if self.memory_profiler:
            return self._memory_profiled_iterator()
        else:
            return iter(self.loader)

    def _memory_profiled_iterator(self):
        """Iterator with memory profiling enabled."""
        if not self.memory_profiler:
            # Fallback to regular iterator if profiler not available
            for batch in self.loader:
                yield batch
            return

        self.memory_profiler.record_snapshot("epoch_start")

        try:
            for batch_idx, batch in enumerate(self.loader):
                self.memory_profiler.record_snapshot(f"batch_{batch_idx}_processed")
                yield batch

                # Periodic memory cleanup for long epochs
                if batch_idx > 0 and batch_idx % 100 == 0:
                    current_usage = self.memory_profiler._get_memory_info()
                    if current_usage.get("percent", 0) > 85:  # High memory usage
                        logger.warning(
                            f"High memory usage detected ({current_usage['percent']:.1f}%), performing cleanup"
                        )
                        self.cleanup_memory_resources()

        finally:
            if self.memory_profiler:
                self.memory_profiler.record_snapshot("epoch_end")

                # Log epoch memory summary if in debug mode
                if logger.isEnabledFor(logging.DEBUG):
                    try:
                        delta = self.memory_profiler.get_memory_delta("epoch_start")
                        delta_mb = (
                            delta.get("delta_rss_mb", 0)
                            if isinstance(delta, dict)
                            else 0
                        )
                        logger.debug(
                            f"Epoch completed with memory delta: {delta_mb:+.1f} MB"
                        )
                    except Exception as e:
                        logger.debug(f"Could not calculate memory delta: {e}")

    def load_batches_concurrently(self, batch_indices: list, timeout: float = 30.0):
        """Load multiple batches concurrently using enhanced thread safety.

        Args:
            batch_indices: List of batch indices to load
            timeout: Maximum time to wait for all batches

        Returns:
            Dictionary mapping batch indices to loaded batch data

        Raises:
            TimeoutError: If loading takes longer than timeout
            RuntimeError: If concurrent loading fails
        """
        if not hasattr(self, "_enhanced_thread_safety"):
            raise RuntimeError("Enhanced thread safety not initialized")

        try:
            with enhanced_thread_safety_execution(
                context_name="concurrent_batch_loading"
            ) as context:
                # Use the concurrent batch loading manager
                with context["concurrent_batch_loading"](
                    self, batch_indices
                ) as batch_generator:
                    results = {}

                    start_time = time.time()
                    for batch_idx, batch_data in batch_generator():
                        if time.time() - start_time > timeout:
                            raise TimeoutError(
                                f"Concurrent batch loading exceeded {timeout}s timeout"
                            )
                        results[batch_idx] = batch_data

                    logger.debug(
                        f"Successfully loaded {len(results)} batches concurrently"
                    )
                    return results

        except Exception as e:
            logger.error(f"Concurrent batch loading failed: {e}")
            raise

    def get_thread_safety_performance_report(self) -> dict:
        """Get comprehensive thread safety performance report.

        Returns:
            Dictionary containing performance metrics and recommendations
        """
        if not hasattr(self, "_enhanced_thread_safety"):
            return {"error": "Enhanced thread safety not initialized"}

        try:
            return self._enhanced_thread_safety.get_comprehensive_performance_report()
        except Exception as e:
            logger.error(f"Failed to generate performance report: {e}")
            return {"error": str(e)}

    def optimize_thread_safety_settings(self) -> dict:
        """Optimize thread safety settings based on current performance.

        Returns:
            Dictionary containing optimization recommendations and changes
        """
        if not hasattr(self, "_enhanced_thread_safety"):
            return {"error": "Enhanced thread safety not initialized"}

        try:
            optimizations = self._enhanced_thread_safety.optimize_all_components()
            logger.info(f"Thread safety optimization completed: {optimizations}")
            return optimizations
        except Exception as e:
            logger.error(f"Failed to optimize thread safety settings: {e}")
            return {"error": str(e)}

    def cleanup_thread_safety_resources(self):
        """Clean up thread safety resources when dataloader is no longer needed."""
        try:
            # Unregister from enhanced thread safety manager
            if hasattr(self, "_enhanced_thread_safety"):
                self._enhanced_thread_safety.concurrent_loader_manager.unregister_dataloader(
                    self
                )

            # Clean up original thread safety resources
            if hasattr(self, "_thread_safety_manager"):
                self._thread_safety_manager.cleanup_orphaned_resources()

            logger.debug(
                f"Thread safety resources cleaned up for dataloader {id(self)}"
            )

        except Exception as e:
            logger.warning(f"Error during thread safety cleanup: {e}")
