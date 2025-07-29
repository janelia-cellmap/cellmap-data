"""
Enhanced memory management utilities for CellMap datasets.

This module provides memory optimization infrastructure to address Week 5 Day 3-4
memory usage optimization objectives, including memory profiling, resource cleanup,
and efficient memory allocation strategies.
"""

import gc
import psutil
import torch
import warnings
from contextlib import contextmanager
from typing import Optional, Dict, Any, List, Tuple
from threading import Lock
import time
import os

from .logging_config import CellMapLogger

logger = CellMapLogger.get_logger("memory_manager")


class MemoryProfiler:
    """System memory profiling and monitoring for performance optimization."""

    def __init__(self):
        """Initialize memory profiler with baseline measurements."""
        self._initial_memory = self._get_memory_info()
        self._peak_memory = self._initial_memory.copy()
        self._memory_history: List[Dict[str, Any]] = []
        self._lock = Lock()

    def _get_memory_info(self) -> Dict[str, Any]:
        """Get current memory usage information.

        Returns:
            Dictionary containing memory usage metrics in MB
        """
        process = psutil.Process()
        memory_info = process.memory_info()

        result = {
            "rss_mb": memory_info.rss / (1024 * 1024),  # Resident Set Size
            "vms_mb": memory_info.vms / (1024 * 1024),  # Virtual Memory Size
            "percent": process.memory_percent(),
            "timestamp": time.time(),
        }

        # Add GPU memory if available
        if torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)
                gpu_reserved = torch.cuda.memory_reserved() / (1024 * 1024)
                gpu_cached = torch.cuda.memory_cached() / (1024 * 1024)

                result.update(
                    {
                        "gpu_allocated_mb": gpu_memory,
                        "gpu_reserved_mb": gpu_reserved,
                        "gpu_cached_mb": gpu_cached,
                    }
                )
            except RuntimeError:
                # GPU memory tracking failed, continue without GPU metrics
                pass

        return result

    def record_snapshot(self, label: str = "") -> Dict[str, Any]:
        """Record a memory usage snapshot.

        Args:
            label: Optional label for this snapshot

        Returns:
            Current memory usage information
        """
        with self._lock:
            current_memory = self._get_memory_info()
            current_memory["label"] = label

            # Update peak memory tracking
            for key in ["rss_mb", "vms_mb"]:
                if key in current_memory and current_memory[
                    key
                ] > self._peak_memory.get(key, 0):
                    self._peak_memory[key] = current_memory[key]

            if "gpu_allocated_mb" in current_memory:
                if current_memory["gpu_allocated_mb"] > self._peak_memory.get(
                    "gpu_allocated_mb", 0
                ):
                    self._peak_memory["gpu_allocated_mb"] = current_memory[
                        "gpu_allocated_mb"
                    ]

            self._memory_history.append(current_memory)
            return current_memory.copy()

    def get_peak_usage(self) -> Dict[str, Any]:
        """Get peak memory usage since profiler initialization.

        Returns:
            Peak memory usage metrics
        """
        with self._lock:
            return self._peak_memory.copy()

    def get_memory_delta(self, baseline_label: str = "") -> Dict[str, Any]:
        """Calculate memory usage change from baseline.

        Args:
            baseline_label: Label of baseline snapshot, empty string for initial

        Returns:
            Memory usage delta from baseline
        """
        with self._lock:
            if baseline_label == "":
                baseline = self._initial_memory
            else:
                # Find snapshot with matching label
                baseline = None
                for snapshot in self._memory_history:
                    if snapshot.get("label") == baseline_label:
                        baseline = snapshot
                        break

                if baseline is None:
                    logger.warning(f"Baseline snapshot '{baseline_label}' not found")
                    baseline = self._initial_memory

            current = self._get_memory_info()
            delta = {}

            for key in baseline:
                if key in current and isinstance(baseline[key], (int, float)):
                    delta[f"delta_{key}"] = current[key] - baseline[key]

            return delta

    def generate_report(self) -> str:
        """Generate a comprehensive memory usage report.

        Returns:
            Formatted memory usage report
        """
        with self._lock:
            if not self._memory_history:
                current = self._get_memory_info()
                self._memory_history.append(current)

            report_lines = [
                "=== Memory Usage Report ===",
                f"Initial Memory: {self._initial_memory['rss_mb']:.1f} MB RSS, "
                f"{self._initial_memory['vms_mb']:.1f} MB VMS",
                f"Peak Memory: {self._peak_memory['rss_mb']:.1f} MB RSS, "
                f"{self._peak_memory['vms_mb']:.1f} MB VMS",
            ]

            if "gpu_allocated_mb" in self._peak_memory:
                report_lines.append(
                    f"Peak GPU Memory: {self._peak_memory['gpu_allocated_mb']:.1f} MB allocated, "
                    f"{self._peak_memory.get('gpu_reserved_mb', 0):.1f} MB reserved"
                )

            # Memory growth analysis
            if len(self._memory_history) > 1:
                first = self._memory_history[0]
                last = self._memory_history[-1]
                growth_rss = last["rss_mb"] - first["rss_mb"]
                growth_percent = (
                    (growth_rss / first["rss_mb"]) * 100 if first["rss_mb"] > 0 else 0
                )

                report_lines.append(
                    f"Memory Growth: {growth_rss:+.1f} MB ({growth_percent:+.1f}%)"
                )

            # Recent snapshots
            if len(self._memory_history) > 0:
                report_lines.append("\nRecent Snapshots:")
                for snapshot in self._memory_history[-5:]:  # Last 5 snapshots
                    label = snapshot.get("label", "unlabeled")
                    timestamp = snapshot.get("timestamp", 0)
                    rss = snapshot.get("rss_mb", 0)
                    gpu = snapshot.get("gpu_allocated_mb", 0)

                    if gpu > 0:
                        report_lines.append(
                            f"  {label}: {rss:.1f} MB RSS, {gpu:.1f} MB GPU"
                        )
                    else:
                        report_lines.append(f"  {label}: {rss:.1f} MB RSS")

            return "\n".join(report_lines)


class MemoryOptimizedResourceManager:
    """Resource manager with memory-aware allocation and cleanup strategies."""

    def __init__(self, memory_limit_mb: Optional[float] = None):
        """Initialize memory-optimized resource manager.

        Args:
            memory_limit_mb: Optional memory limit in MB for resource allocation
        """
        self.memory_limit_mb = memory_limit_mb
        self._allocated_resources: Dict[str, Any] = {}
        self._resource_sizes: Dict[str, float] = {}
        self._lock = Lock()
        self.profiler = MemoryProfiler()

        logger.debug(
            f"MemoryOptimizedResourceManager initialized with limit: {memory_limit_mb} MB"
        )

    def allocate_tensor(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        name: str = "tensor",
    ) -> torch.Tensor:
        """Allocate tensor with memory monitoring and limits.

        Args:
            shape: Tensor shape
            dtype: Tensor data type
            device: Target device for tensor
            name: Resource name for tracking

        Returns:
            Allocated tensor

        Raises:
            MemoryError: If allocation would exceed memory limit
        """
        # Calculate expected memory usage
        element_size = torch.tensor(0, dtype=dtype).element_size()
        expected_size_mb = (torch.tensor(shape).prod().item() * element_size) / (
            1024 * 1024
        )

        # Check memory limit before allocation
        if self.memory_limit_mb is not None:
            current_usage = self.profiler._get_memory_info()
            projected_usage = current_usage["rss_mb"] + expected_size_mb

            if projected_usage > self.memory_limit_mb:
                # Attempt garbage collection to free memory
                self.cleanup_resources()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

                # Re-check after cleanup
                current_usage = self.profiler._get_memory_info()
                projected_usage = current_usage["rss_mb"] + expected_size_mb

                if projected_usage > self.memory_limit_mb:
                    raise MemoryError(
                        f"Tensor allocation would exceed memory limit: "
                        f"{projected_usage:.1f}MB > {self.memory_limit_mb:.1f}MB"
                    )

        with self._lock:
            # Allocate tensor
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            tensor = torch.empty(shape, dtype=dtype, device=device)

            # Track allocation
            self._allocated_resources[name] = tensor
            self._resource_sizes[name] = expected_size_mb

            logger.debug(
                f"Allocated tensor '{name}': {shape} ({expected_size_mb:.1f} MB)"
            )

            return tensor

    def deallocate_resource(self, name: str) -> bool:
        """Deallocate a specific resource by name.

        Args:
            name: Name of resource to deallocate

        Returns:
            True if resource was found and deallocated
        """
        with self._lock:
            if name in self._allocated_resources:
                resource = self._allocated_resources[name]
                size_mb = self._resource_sizes.get(name, 0)

                # Clear tensor reference
                if isinstance(resource, torch.Tensor):
                    del resource

                # Remove from tracking
                del self._allocated_resources[name]
                del self._resource_sizes[name]

                logger.debug(f"Deallocated resource '{name}' ({size_mb:.1f} MB)")
                return True

            return False

    def cleanup_resources(self, force_gc: bool = True) -> int:
        """Clean up all allocated resources and perform garbage collection.

        Args:
            force_gc: Whether to force garbage collection

        Returns:
            Number of resources cleaned up
        """
        with self._lock:
            cleanup_count = len(self._allocated_resources)

            # Clear all tracked resources
            for name in list(self._allocated_resources.keys()):
                resource = self._allocated_resources[name]
                if isinstance(resource, torch.Tensor):
                    del resource

            self._allocated_resources.clear()
            self._resource_sizes.clear()

            logger.debug(f"Cleaned up {cleanup_count} resources")

        if force_gc:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return cleanup_count

    def get_memory_usage_summary(self) -> Dict[str, Any]:
        """Get summary of current memory usage and allocations.

        Returns:
            Memory usage summary
        """
        with self._lock:
            total_allocated_mb = sum(self._resource_sizes.values())

            return {
                "total_tracked_resources": len(self._allocated_resources),
                "total_allocated_mb": total_allocated_mb,
                "memory_limit_mb": self.memory_limit_mb,
                "resource_breakdown": self._resource_sizes.copy(),
                "system_memory": self.profiler._get_memory_info(),
            }


@contextmanager
def memory_profiled_execution(
    label: str = "execution", profiler: Optional[MemoryProfiler] = None
):
    """Context manager for memory-profiled code execution.

    Args:
        label: Label for this execution block
        profiler: Optional existing profiler, creates new one if None
    """
    if profiler is None:
        profiler = MemoryProfiler()

    # Record start
    start_snapshot = profiler.record_snapshot(f"{label}_start")
    logger.debug(
        f"Memory profiling started for '{label}': {start_snapshot['rss_mb']:.1f} MB RSS"
    )

    try:
        yield profiler
    finally:
        # Record end and generate summary
        end_snapshot = profiler.record_snapshot(f"{label}_end")
        delta = profiler.get_memory_delta(f"{label}_start")

        logger.info(
            f"Memory profiling completed for '{label}': "
            f"{end_snapshot['rss_mb']:.1f} MB RSS "
            f"({delta.get('delta_rss_mb', 0):+.1f} MB change)"
        )


def optimize_cuda_memory_allocation():
    """Optimize CUDA memory allocation strategies for better performance.

    This function applies CUDA memory optimization settings that can improve
    memory usage patterns and reduce fragmentation.
    """
    if not torch.cuda.is_available():
        logger.debug("CUDA not available, skipping CUDA memory optimization")
        return

    try:
        # Enable memory pool optimization
        if hasattr(torch.cuda, "memory"):
            # Use expandable segments to reduce memory fragmentation
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        # Set memory fraction if not already set
        if not os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
            # Conservative memory allocation to prevent OOM
            torch.cuda.set_per_process_memory_fraction(0.9)

        logger.info("CUDA memory optimization settings applied")

    except Exception as e:
        logger.warning(f"Failed to apply CUDA memory optimization: {e}")


def get_memory_optimization_recommendations(current_usage: Dict[str, Any]) -> List[str]:
    """Generate memory optimization recommendations based on current usage.

    Args:
        current_usage: Current memory usage metrics

    Returns:
        List of optimization recommendations
    """
    recommendations = []

    rss_mb = current_usage.get("rss_mb", 0)
    percent = current_usage.get("percent", 0)
    gpu_allocated = current_usage.get("gpu_allocated_mb", 0)
    gpu_reserved = current_usage.get("gpu_reserved_mb", 0)

    # High memory usage warnings
    if percent > 80:
        recommendations.append(
            f"High memory usage detected ({percent:.1f}%). Consider reducing batch size or enabling gradient checkpointing."
        )

    # GPU memory fragmentation detection
    if gpu_reserved > 0 and gpu_allocated > 0:
        fragmentation_ratio = (gpu_reserved - gpu_allocated) / gpu_reserved
        if fragmentation_ratio > 0.3:
            recommendations.append(
                f"GPU memory fragmentation detected ({fragmentation_ratio:.1%}). "
                "Consider calling torch.cuda.empty_cache() periodically."
            )

    # Memory growth detection
    if rss_mb > 8000:  # > 8GB
        recommendations.append(
            "Large memory footprint detected. Consider using memory-mapped datasets or reducing data preprocessing in memory."
        )

    # Performance recommendations
    if gpu_allocated < 500 and torch.cuda.is_available():  # < 500MB GPU usage
        recommendations.append(
            "Low GPU memory utilization. Consider increasing batch size for better GPU utilization."
        )

    return recommendations
