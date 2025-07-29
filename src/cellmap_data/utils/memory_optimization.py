"""
Advanced memory optimization enhancements for Week 5 Day 3-4 objectives.

This module extends the existing memory management infrastructure with advanced
optimizations for memory-efficient data loading, streaming capabilities, and
enhanced resource cleanup patterns.
"""

import gc
import os
import time
import threading
import psutil
import torch
import numpy as np
from contextlib import contextmanager
from typing import Optional, Dict, Any, List, Tuple, Iterator, Generator
from dataclasses import dataclass

from .logging_config import get_logger
from .memory_manager import MemoryProfiler, MemoryOptimizedResourceManager
from .gpu_memory_optimizer import GPUMemoryOptimizer, gpu_memory_optimized_context

logger = get_logger("memory_optimization")


@dataclass
class MemoryOptimizationConfig:
    """Configuration for memory optimization strategies."""

    # Memory limits and thresholds
    max_memory_mb: Optional[float] = None
    warning_threshold_percent: float = 80.0
    cleanup_threshold_percent: float = 85.0

    # Streaming configuration
    enable_streaming: bool = True
    stream_buffer_size_mb: float = 512.0
    prefetch_factor: int = 2

    # GPU optimization
    enable_gpu_memory_optimization: bool = True
    gpu_memory_fraction: float = 0.9
    enable_memory_pool: bool = True

    # Garbage collection
    gc_frequency: int = 100  # Every N batches
    aggressive_cleanup: bool = False

    # Memory monitoring
    enable_detailed_profiling: bool = False
    snapshot_frequency: int = 50  # Every N operations


class StreamingDataBuffer:
    """Memory-efficient streaming buffer for large dataset processing."""

    def __init__(self, buffer_size_mb: float = 512.0, prefetch_factor: int = 2):
        """Initialize streaming buffer.

        Args:
            buffer_size_mb: Maximum buffer size in MB
            prefetch_factor: Number of items to prefetch ahead
        """
        self.buffer_size_mb = buffer_size_mb
        self.prefetch_factor = prefetch_factor
        self._buffer: List[Any] = []
        self._buffer_size_current = 0.0
        self._lock = threading.Lock()
        self._prefetch_thread = None
        self._stop_prefetch = threading.Event()

        logger.debug(
            f"StreamingDataBuffer initialized with {buffer_size_mb:.1f}MB capacity"
        )

    def _estimate_item_size_mb(self, item: Any) -> float:
        """Estimate memory size of an item in MB."""
        if isinstance(item, torch.Tensor):
            return (item.numel() * item.element_size()) / (1024 * 1024)
        elif isinstance(item, np.ndarray):
            return item.nbytes / (1024 * 1024)
        elif isinstance(item, dict):
            total_size = 0
            for value in item.values():
                total_size += self._estimate_item_size_mb(value)
            return total_size
        else:
            # Rough estimate for other objects
            return 0.1  # 100KB default

    def add_item(self, item: Any) -> bool:
        """Add item to buffer if there's space.

        Args:
            item: Item to add to buffer

        Returns:
            True if item was added, False if buffer is full
        """
        item_size = self._estimate_item_size_mb(item)

        with self._lock:
            if self._buffer_size_current + item_size <= self.buffer_size_mb:
                self._buffer.append(item)
                self._buffer_size_current += item_size
                return True
            else:
                return False

    def get_item(self) -> Optional[Any]:
        """Get next item from buffer."""
        with self._lock:
            if self._buffer:
                item = self._buffer.pop(0)
                item_size = self._estimate_item_size_mb(item)
                self._buffer_size_current -= item_size
                return item
            return None

    def clear_buffer(self) -> int:
        """Clear all items from buffer.

        Returns:
            Number of items cleared
        """
        with self._lock:
            count = len(self._buffer)
            self._buffer.clear()
            self._buffer_size_current = 0.0
            return count

    def get_buffer_status(self) -> Dict[str, Any]:
        """Get current buffer status."""
        with self._lock:
            return {
                "items_count": len(self._buffer),
                "current_size_mb": self._buffer_size_current,
                "max_size_mb": self.buffer_size_mb,
                "utilization_percent": (self._buffer_size_current / self.buffer_size_mb)
                * 100,
            }


class AdvancedMemoryManager:
    """Advanced memory manager with streaming and optimization capabilities."""

    def __init__(self, config: Optional[MemoryOptimizationConfig] = None):
        """Initialize advanced memory manager.

        Args:
            config: Memory optimization configuration
        """
        self.config = config or MemoryOptimizationConfig()
        self.profiler = MemoryProfiler()
        self.resource_manager = MemoryOptimizedResourceManager(
            self.config.max_memory_mb
        )
        self.streaming_buffer = StreamingDataBuffer(
            self.config.stream_buffer_size_mb, self.config.prefetch_factor
        )

        # Initialize GPU memory optimizer
        self.gpu_optimizer = GPUMemoryOptimizer() if torch.cuda.is_available() else None

        # Memory monitoring
        self._operation_count = 0
        self._last_gc_operation = 0
        self._memory_alerts: List[Dict[str, Any]] = []

        # Apply GPU optimizations
        if self.config.enable_gpu_memory_optimization:
            self._apply_gpu_optimizations()

        logger.info(f"AdvancedMemoryManager initialized with config: {self.config}")

    def _apply_gpu_optimizations(self):
        """Apply advanced GPU memory optimizations."""
        if not torch.cuda.is_available():
            return

        try:
            # Set memory fraction
            torch.cuda.set_per_process_memory_fraction(self.config.gpu_memory_fraction)

            # Use GPU optimizer for advanced optimizations
            if self.gpu_optimizer:
                self.gpu_optimizer.set_memory_fraction(self.config.gpu_memory_fraction)
                self.gpu_optimizer.optimize_memory_fragmentation(aggressive=False)

            # Enable memory pool optimization
            if self.config.enable_memory_pool:
                os.environ.setdefault(
                    "PYTORCH_CUDA_ALLOC_CONF",
                    "expandable_segments:True,roundup_power2_divisions:1",
                )

            # Clear any existing cache
            torch.cuda.empty_cache()

            logger.info(
                f"GPU memory optimizations applied: fraction={self.config.gpu_memory_fraction}"
            )

        except Exception as e:
            logger.warning(f"Failed to apply GPU optimizations: {e}")

    def allocate_optimized_tensor(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        name: str = "tensor",
    ) -> torch.Tensor:
        """Allocate tensor with advanced memory optimization strategies.

        Args:
            shape: Tensor shape
            dtype: Tensor data type
            device: Target device
            name: Resource name for tracking

        Returns:
            Optimized tensor allocation
        """
        self._operation_count += 1

        # Check memory before allocation
        current_memory = self.profiler._get_memory_info()
        if current_memory["percent"] > self.config.warning_threshold_percent:
            self._handle_high_memory_usage(current_memory)

        # Use resource manager for allocation
        tensor = self.resource_manager.allocate_tensor(shape, dtype, device, name)

        # Periodic profiling snapshots
        if (
            self.config.enable_detailed_profiling
            and self._operation_count % self.config.snapshot_frequency == 0
        ):
            self.profiler.record_snapshot(f"tensor_allocation_{self._operation_count}")

        return tensor

    def _handle_high_memory_usage(self, current_memory: Dict[str, Any]):
        """Handle high memory usage situations."""
        memory_percent = current_memory["percent"]

        # Log memory alert
        alert = {
            "timestamp": time.time(),
            "memory_percent": memory_percent,
            "rss_mb": current_memory["rss_mb"],
            "action_taken": [],
        }

        if memory_percent > self.config.cleanup_threshold_percent:
            # Aggressive cleanup
            logger.warning(
                f"High memory usage detected ({memory_percent:.1f}%), performing cleanup"
            )

            # Clear streaming buffer
            cleared_items = self.streaming_buffer.clear_buffer()
            if cleared_items > 0:
                alert["action_taken"].append(f"cleared_{cleared_items}_buffer_items")

            # Resource cleanup
            cleaned_resources = self.resource_manager.cleanup_resources(force_gc=True)
            if cleaned_resources > 0:
                alert["action_taken"].append(f"cleaned_{cleaned_resources}_resources")

            # GPU cache cleanup
            if torch.cuda.is_available():
                if self.gpu_optimizer:
                    # Use advanced GPU optimization
                    self.gpu_optimizer.optimize_memory_fragmentation(aggressive=True)
                else:
                    # Fallback to basic cache clear
                    torch.cuda.empty_cache()
                alert["action_taken"].append("optimized_gpu_memory")

            # Force garbage collection
            gc.collect()
            alert["action_taken"].append("forced_gc")

        else:
            # Standard cleanup
            logger.info(
                f"Memory usage warning ({memory_percent:.1f}%), performing light cleanup"
            )
            self._perform_periodic_gc()
            alert["action_taken"].append("periodic_gc")

        self._memory_alerts.append(alert)

        # Keep only recent alerts
        if len(self._memory_alerts) > 100:
            self._memory_alerts = self._memory_alerts[-50:]

    def _perform_periodic_gc(self):
        """Perform periodic garbage collection based on operation count."""
        if (
            self._operation_count - self._last_gc_operation
        ) >= self.config.gc_frequency:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._last_gc_operation = self._operation_count
            logger.debug(f"Periodic GC performed at operation {self._operation_count}")

    @contextmanager
    def memory_efficient_processing(self, operation_name: str = "processing"):
        """Context manager for memory-efficient data processing.

        Args:
            operation_name: Name of the processing operation
        """
        # Record initial state
        start_snapshot = self.profiler.record_snapshot(f"{operation_name}_start")
        initial_memory = start_snapshot["rss_mb"]

        logger.debug(
            f"Memory-efficient processing started '{operation_name}': {initial_memory:.1f} MB"
        )

        try:
            yield self

        finally:
            # Final cleanup and reporting
            end_snapshot = self.profiler.record_snapshot(f"{operation_name}_end")
            final_memory = end_snapshot["rss_mb"]
            delta = final_memory - initial_memory

            # Cleanup if memory increased significantly
            if delta > 100:  # More than 100MB increase
                logger.info(
                    f"Significant memory increase detected ({delta:+.1f} MB), performing cleanup"
                )
                self.resource_manager.cleanup_resources(force_gc=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            logger.info(
                f"Memory-efficient processing completed '{operation_name}': "
                f"{final_memory:.1f} MB ({delta:+.1f} MB change)"
            )

    def get_streaming_data_loader(
        self, data_source: Iterator[Any]
    ) -> Generator[Any, None, None]:
        """Create a memory-efficient streaming data loader.

        Args:
            data_source: Source data iterator

        Yields:
            Data items with memory-efficient streaming
        """
        buffer_status_logged = False

        for item in data_source:
            # Try to add to streaming buffer
            if self.streaming_buffer.add_item(item):
                # Buffer item for later processing
                continue
            else:
                # Buffer is full, yield items and add current item

                # Log buffer status once
                if not buffer_status_logged:
                    status = self.streaming_buffer.get_buffer_status()
                    logger.debug(f"Streaming buffer status: {status}")
                    buffer_status_logged = True

                # Yield items from buffer
                while True:
                    buffered_item = self.streaming_buffer.get_item()
                    if buffered_item is None:
                        break
                    yield buffered_item

                # Add current item to empty buffer
                self.streaming_buffer.add_item(item)

                # Check memory and perform cleanup if needed
                current_memory = self.profiler._get_memory_info()
                if current_memory["percent"] > self.config.warning_threshold_percent:
                    self._handle_high_memory_usage(current_memory)

        # Yield remaining items in buffer
        while True:
            remaining_item = self.streaming_buffer.get_item()
            if remaining_item is None:
                break
            yield remaining_item

    def optimize_for_large_dataset(self, dataset_size_gb: float) -> Dict[str, Any]:
        """Optimize memory settings for large dataset processing.

        Args:
            dataset_size_gb: Estimated dataset size in GB

        Returns:
            Applied optimization settings
        """
        optimizations = {}

        if dataset_size_gb > 10:  # > 10GB
            # Enable aggressive memory management
            self.config.cleanup_threshold_percent = 75.0
            self.config.gc_frequency = 50
            self.config.aggressive_cleanup = True
            optimizations["aggressive_cleanup"] = True

            # Reduce buffer sizes
            self.config.stream_buffer_size_mb = min(
                256.0, self.config.stream_buffer_size_mb
            )
            self.streaming_buffer.buffer_size_mb = self.config.stream_buffer_size_mb
            optimizations["reduced_buffer_size"] = self.config.stream_buffer_size_mb

            # Enable detailed profiling for large datasets
            self.config.enable_detailed_profiling = True
            optimizations["detailed_profiling"] = True

        if dataset_size_gb > 50:  # > 50GB
            # Extra aggressive settings for very large datasets
            self.config.cleanup_threshold_percent = 70.0
            self.config.gc_frequency = 25
            optimizations["extra_aggressive"] = True

        logger.info(
            f"Memory optimizations applied for {dataset_size_gb}GB dataset: {optimizations}"
        )
        return optimizations

    def get_comprehensive_memory_report(self) -> str:
        """Generate comprehensive memory usage and optimization report."""
        report_lines = [
            "=== Advanced Memory Management Report ===",
            "",
            "Configuration:",
            f"  Max Memory: {self.config.max_memory_mb or 'Unlimited'} MB",
            f"  Warning Threshold: {self.config.warning_threshold_percent}%",
            f"  Cleanup Threshold: {self.config.cleanup_threshold_percent}%",
            f"  GC Frequency: {self.config.gc_frequency} operations",
            "",
            "Operations:",
            f"  Total Operations: {self._operation_count}",
            f"  Last GC: Operation {self._last_gc_operation}",
            f"  Memory Alerts: {len(self._memory_alerts)}",
            "",
        ]

        # Add profiler report
        report_lines.append(self.profiler.generate_report())
        report_lines.append("")

        # Add resource manager summary
        resource_summary = self.resource_manager.get_memory_usage_summary()
        report_lines.extend(
            [
                "Resource Manager:",
                f"  Tracked Resources: {resource_summary['total_tracked_resources']}",
                f"  Allocated Memory: {resource_summary['total_allocated_mb']:.1f} MB",
                "",
            ]
        )

        # Add streaming buffer status
        buffer_status = self.streaming_buffer.get_buffer_status()
        report_lines.extend(
            [
                "Streaming Buffer:",
                f"  Items: {buffer_status['items_count']}",
                f"  Size: {buffer_status['current_size_mb']:.1f} / {buffer_status['max_size_mb']:.1f} MB",
                f"  Utilization: {buffer_status['utilization_percent']:.1f}%",
                "",
            ]
        )

        # Add GPU memory status if available
        if self.gpu_optimizer:
            gpu_report = self.gpu_optimizer.generate_memory_report()
            report_lines.extend(["GPU Memory Status:", gpu_report, ""])

        # Recent memory alerts
        if self._memory_alerts:
            report_lines.append("Recent Memory Alerts:")
            for alert in self._memory_alerts[-5:]:  # Last 5 alerts
                timestamp = time.strftime(
                    "%H:%M:%S", time.localtime(alert["timestamp"])
                )
                actions = ", ".join(alert["action_taken"])
                report_lines.append(
                    f"  {timestamp}: {alert['memory_percent']:.1f}% usage, "
                    f"actions: {actions}"
                )

        return "\n".join(report_lines)


# Convenience functions for common memory optimization patterns


def create_memory_efficient_dataloader(dataset, batch_size: int, **kwargs):
    """Create a memory-efficient data loader with optimized settings.

    Args:
        dataset: Dataset to load from
        batch_size: Batch size
        **kwargs: Additional DataLoader arguments

    Returns:
        Memory-optimized CellMapDataLoader instance
    """
    from ..dataloader import CellMapDataLoader

    # Estimate dataset memory requirements
    try:
        if hasattr(dataset, "input_arrays") and hasattr(dataset, "target_arrays"):
            input_arrays = getattr(dataset, "input_arrays", {})
            target_arrays = getattr(dataset, "target_arrays", {})

            # Calculate approximate memory per sample
            total_elements = 0
            for array_info in input_arrays.values():
                if "shape" in array_info:
                    total_elements += np.prod(array_info["shape"])
            for array_info in target_arrays.values():
                if "shape" in array_info:
                    total_elements += np.prod(array_info["shape"])

            # Estimate memory per batch (float32 = 4 bytes)
            memory_per_batch_mb = (total_elements * batch_size * 4) / (1024 * 1024)

            # Set memory limit to reasonable multiple of batch size
            suggested_memory_limit = memory_per_batch_mb * 50  # 50 batches worth

            kwargs.setdefault("memory_limit_mb", suggested_memory_limit)
            kwargs.setdefault("enable_memory_profiling", True)

            logger.info(
                f"Memory-efficient dataloader configured: "
                f"{memory_per_batch_mb:.1f}MB/batch, "
                f"{suggested_memory_limit:.1f}MB limit"
            )

    except Exception as e:
        logger.warning(f"Could not estimate memory requirements: {e}")

    return CellMapDataLoader(dataset, batch_size=batch_size, **kwargs)


@contextmanager
def memory_optimized_training_context(memory_limit_mb: Optional[float] = None):
    """Context manager for memory-optimized training sessions.

    Args:
        memory_limit_mb: Optional memory limit for training
    """
    config = MemoryOptimizationConfig(
        max_memory_mb=memory_limit_mb,
        cleanup_threshold_percent=80.0,
        gc_frequency=50,
        enable_gpu_memory_optimization=True,
        enable_detailed_profiling=True,
    )

    manager = AdvancedMemoryManager(config)

    with manager.memory_efficient_processing("training_session"):
        logger.info("Memory-optimized training context started")
        yield manager
        logger.info("Memory-optimized training context completed")
