"""
Advanced GPU memory optimization for Week 5 Day 3-4 memory management.

This module provides specialized GPU memory management utilities including
CUDA memory pool optimization, memory fragmentation detection, and
GPU-specific resource cleanup strategies.
"""

import gc
import os
import torch
import psutil
from contextlib import contextmanager
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from .logging_config import get_logger

logger = get_logger("gpu_memory_optimization")


@dataclass
class GPUMemoryStats:
    """GPU memory statistics and utilization information."""

    device_id: int
    allocated_mb: float
    reserved_mb: float
    free_mb: float
    total_mb: float
    utilization_percent: float
    fragmentation_percent: float
    peak_allocated_mb: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "device_id": self.device_id,
            "allocated_mb": self.allocated_mb,
            "reserved_mb": self.reserved_mb,
            "free_mb": self.free_mb,
            "total_mb": self.total_mb,
            "utilization_percent": self.utilization_percent,
            "fragmentation_percent": self.fragmentation_percent,
            "peak_allocated_mb": self.peak_allocated_mb,
        }


class GPUMemoryOptimizer:
    """Advanced GPU memory optimization with fragmentation detection and cleanup."""

    def __init__(self):
        """Initialize GPU memory optimizer."""
        self.device_count = (
            torch.cuda.device_count() if torch.cuda.is_available() else 0
        )
        self._baseline_stats: Dict[int, GPUMemoryStats] = {}
        self._optimization_history: List[Dict[str, Any]] = []

        # Apply initial optimizations
        if self.device_count > 0:
            self._apply_initial_optimizations()

        logger.info(
            f"GPUMemoryOptimizer initialized for {self.device_count} GPU devices"
        )

    def _apply_initial_optimizations(self):
        """Apply initial GPU memory optimizations."""
        try:
            # Set memory allocation strategy
            os.environ.setdefault(
                "PYTORCH_CUDA_ALLOC_CONF",
                "expandable_segments:True,roundup_power2_divisions:1,garbage_collection_threshold:0.8",
            )

            # Enable memory pool snapshot for debugging
            if hasattr(torch.cuda, "memory_snapshot"):
                # Only available in newer PyTorch versions
                pass

            # Clear initial cache
            for device_id in range(self.device_count):
                with torch.cuda.device(device_id):
                    torch.cuda.empty_cache()

            logger.debug("Initial GPU memory optimizations applied")

        except Exception as e:
            logger.warning(f"Failed to apply initial GPU optimizations: {e}")

    def get_memory_stats(
        self, device_id: Optional[int] = None
    ) -> Dict[int, GPUMemoryStats]:
        """Get comprehensive memory statistics for GPU devices.

        Args:
            device_id: Specific device ID, or None for all devices

        Returns:
            Dictionary mapping device ID to memory statistics
        """
        if not torch.cuda.is_available():
            return {}

        devices = (
            [device_id] if device_id is not None else list(range(self.device_count))
        )
        stats = {}

        for dev_id in devices:
            try:
                with torch.cuda.device(dev_id):
                    # Get basic memory info
                    allocated = torch.cuda.memory_allocated(dev_id)
                    reserved = torch.cuda.memory_reserved(dev_id)

                    # Get device properties
                    props = torch.cuda.get_device_properties(dev_id)
                    total_memory = props.total_memory

                    # Calculate metrics
                    allocated_mb = allocated / (1024**2)
                    reserved_mb = reserved / (1024**2)
                    total_mb = total_memory / (1024**2)
                    free_mb = total_mb - reserved_mb

                    utilization_percent = (
                        (reserved / total_memory) * 100 if total_memory > 0 else 0
                    )

                    # Calculate fragmentation (reserved but not allocated)
                    fragmentation = reserved - allocated
                    fragmentation_percent = (
                        (fragmentation / reserved) * 100 if reserved > 0 else 0
                    )

                    # Get peak memory usage
                    peak_allocated = torch.cuda.max_memory_allocated(dev_id)
                    peak_allocated_mb = peak_allocated / (1024**2)

                    stats[dev_id] = GPUMemoryStats(
                        device_id=dev_id,
                        allocated_mb=allocated_mb,
                        reserved_mb=reserved_mb,
                        free_mb=free_mb,
                        total_mb=total_mb,
                        utilization_percent=utilization_percent,
                        fragmentation_percent=fragmentation_percent,
                        peak_allocated_mb=peak_allocated_mb,
                    )

            except Exception as e:
                logger.warning(f"Failed to get memory stats for device {dev_id}: {e}")

        return stats

    def detect_memory_fragmentation(self, threshold_percent: float = 20.0) -> List[int]:
        """Detect GPU devices with high memory fragmentation.

        Args:
            threshold_percent: Fragmentation threshold percentage

        Returns:
            List of device IDs with high fragmentation
        """
        fragmented_devices = []
        stats = self.get_memory_stats()

        for device_id, device_stats in stats.items():
            if device_stats.fragmentation_percent > threshold_percent:
                fragmented_devices.append(device_id)
                logger.warning(
                    f"High memory fragmentation detected on GPU {device_id}: "
                    f"{device_stats.fragmentation_percent:.1f}% "
                    f"({device_stats.reserved_mb - device_stats.allocated_mb:.1f}MB fragmented)"
                )

        return fragmented_devices

    def optimize_memory_fragmentation(
        self, device_id: Optional[int] = None, aggressive: bool = False
    ) -> Dict[str, Any]:
        """Optimize memory fragmentation on GPU devices.

        Args:
            device_id: Specific device to optimize, or None for all devices
            aggressive: Whether to use aggressive optimization strategies

        Returns:
            Optimization results summary
        """
        if not torch.cuda.is_available():
            return {"status": "no_gpu_available"}

        devices = (
            [device_id] if device_id is not None else list(range(self.device_count))
        )
        results = {
            "optimized_devices": [],
            "fragmentation_before": {},
            "fragmentation_after": {},
            "memory_freed_mb": {},
            "strategies_applied": [],
        }

        for dev_id in devices:
            try:
                # Get before stats
                before_stats = self.get_memory_stats(dev_id)[dev_id]
                results["fragmentation_before"][
                    dev_id
                ] = before_stats.fragmentation_percent

                with torch.cuda.device(dev_id):
                    # Strategy 1: Clear cache
                    torch.cuda.empty_cache()
                    results["strategies_applied"].append("cache_clear")

                    if aggressive:
                        # Strategy 2: Force garbage collection
                        gc.collect()
                        torch.cuda.empty_cache()  # Clear again after GC
                        results["strategies_applied"].append("aggressive_gc")

                        # Strategy 3: Reset peak memory stats
                        torch.cuda.reset_peak_memory_stats(dev_id)
                        results["strategies_applied"].append("reset_peak_stats")

                    # Strategy 4: Synchronize to ensure all operations complete
                    torch.cuda.synchronize(dev_id)
                    results["strategies_applied"].append("synchronize")

                # Get after stats
                after_stats = self.get_memory_stats(dev_id)[dev_id]
                results["fragmentation_after"][
                    dev_id
                ] = after_stats.fragmentation_percent

                # Calculate memory freed
                memory_freed = before_stats.reserved_mb - after_stats.reserved_mb
                results["memory_freed_mb"][dev_id] = memory_freed

                results["optimized_devices"].append(dev_id)

                logger.info(
                    f"GPU {dev_id} memory optimization: "
                    f"fragmentation {before_stats.fragmentation_percent:.1f}% → "
                    f"{after_stats.fragmentation_percent:.1f}%, "
                    f"freed {memory_freed:.1f}MB"
                )

            except Exception as e:
                logger.error(f"Failed to optimize memory for GPU {dev_id}: {e}")

        # Record optimization in history
        self._optimization_history.append(
            {"timestamp": torch.cuda.Event(enable_timing=True), "results": results}
        )

        # Keep only recent history
        if len(self._optimization_history) > 100:
            self._optimization_history = self._optimization_history[-50:]

        return results

    def set_memory_fraction(self, fraction: float, device_id: Optional[int] = None):
        """Set memory fraction for GPU devices.

        Args:
            fraction: Memory fraction (0.0 to 1.0)
            device_id: Specific device ID, or None for all devices
        """
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, cannot set memory fraction")
            return

        if not (0.0 <= fraction <= 1.0):
            raise ValueError(
                f"Memory fraction must be between 0.0 and 1.0, got {fraction}"
            )

        devices = (
            [device_id] if device_id is not None else list(range(self.device_count))
        )

        for dev_id in devices:
            try:
                torch.cuda.set_per_process_memory_fraction(fraction, dev_id)
                logger.info(f"Set memory fraction to {fraction:.2f} for GPU {dev_id}")
            except Exception as e:
                logger.error(f"Failed to set memory fraction for GPU {dev_id}: {e}")

    def monitor_memory_usage(
        self, operation_name: str = "operation"
    ) -> "GPUMemoryMonitor":
        """Create a context manager for monitoring GPU memory usage during operations.

        Args:
            operation_name: Name of the operation being monitored

        Returns:
            GPU memory monitor context manager
        """
        return GPUMemoryMonitor(self, operation_name)

    def get_memory_recommendations(self) -> List[str]:
        """Get memory optimization recommendations based on current state.

        Returns:
            List of optimization recommendations
        """
        recommendations = []

        if not torch.cuda.is_available():
            return ["CUDA not available"]

        stats = self.get_memory_stats()

        for device_id, device_stats in stats.items():
            device_recs = []

            # High utilization check
            if device_stats.utilization_percent > 90:
                device_recs.append(
                    f"GPU {device_id}: Very high memory usage ({device_stats.utilization_percent:.1f}%), consider reducing batch size"
                )
            elif device_stats.utilization_percent > 80:
                device_recs.append(
                    f"GPU {device_id}: High memory usage ({device_stats.utilization_percent:.1f}%), monitor for OOM errors"
                )

            # Fragmentation check
            if device_stats.fragmentation_percent > 30:
                device_recs.append(
                    f"GPU {device_id}: High fragmentation ({device_stats.fragmentation_percent:.1f}%), run memory optimization"
                )
            elif device_stats.fragmentation_percent > 15:
                device_recs.append(
                    f"GPU {device_id}: Medium fragmentation ({device_stats.fragmentation_percent:.1f}%), consider clearing cache"
                )

            # Peak vs current check
            if device_stats.peak_allocated_mb > device_stats.allocated_mb * 1.5:
                device_recs.append(
                    f"GPU {device_id}: Peak usage much higher than current, possible memory spikes"
                )

            recommendations.extend(device_recs)

        # General recommendations
        if not recommendations:
            recommendations.append("GPU memory usage appears optimal")
        else:
            recommendations.append(
                "Consider using mixed precision training to reduce memory usage"
            )
            recommendations.append("Enable gradient checkpointing for large models")

        return recommendations

    def generate_memory_report(self) -> str:
        """Generate comprehensive GPU memory report.

        Returns:
            Formatted memory report string
        """
        if not torch.cuda.is_available():
            return "GPU Memory Report: CUDA not available"

        report_lines = [
            "=== GPU Memory Optimization Report ===",
            f"Devices Available: {self.device_count}",
            "",
        ]

        stats = self.get_memory_stats()

        for device_id, device_stats in stats.items():
            props = torch.cuda.get_device_properties(device_id)
            report_lines.extend(
                [
                    f"GPU {device_id}: {props.name}",
                    f"  Total Memory: {device_stats.total_mb:.1f} MB",
                    f"  Allocated: {device_stats.allocated_mb:.1f} MB ({device_stats.allocated_mb/device_stats.total_mb*100:.1f}%)",
                    f"  Reserved: {device_stats.reserved_mb:.1f} MB ({device_stats.reserved_mb/device_stats.total_mb*100:.1f}%)",
                    f"  Free: {device_stats.free_mb:.1f} MB ({device_stats.free_mb/device_stats.total_mb*100:.1f}%)",
                    f"  Fragmentation: {device_stats.fragmentation_percent:.1f}%",
                    f"  Peak Allocated: {device_stats.peak_allocated_mb:.1f} MB",
                    "",
                ]
            )

        # Add fragmentation analysis
        fragmented_devices = self.detect_memory_fragmentation(threshold_percent=15.0)
        if fragmented_devices:
            report_lines.extend(
                [
                    "Fragmentation Analysis:",
                    f"  Devices with high fragmentation: {fragmented_devices}",
                    "",
                ]
            )

        # Add recommendations
        recommendations = self.get_memory_recommendations()
        report_lines.extend(
            ["Recommendations:", *[f"  - {rec}" for rec in recommendations], ""]
        )

        # Add optimization history summary
        if self._optimization_history:
            report_lines.extend(
                [
                    f"Recent Optimizations: {len(self._optimization_history)} performed",
                    "",
                ]
            )

        return "\n".join(report_lines)


class GPUMemoryMonitor:
    """Context manager for monitoring GPU memory usage during operations."""

    def __init__(self, optimizer: GPUMemoryOptimizer, operation_name: str):
        """Initialize GPU memory monitor.

        Args:
            optimizer: GPU memory optimizer instance
            operation_name: Name of the operation being monitored
        """
        self.optimizer = optimizer
        self.operation_name = operation_name
        self.start_stats = {}
        self.end_stats = {}

    def __enter__(self):
        """Start monitoring GPU memory usage."""
        self.start_stats = self.optimizer.get_memory_stats()
        logger.debug(f"GPU memory monitoring started for '{self.operation_name}'")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop monitoring and report GPU memory usage changes."""
        self.end_stats = self.optimizer.get_memory_stats()

        # Calculate and report changes
        for device_id in self.start_stats:
            if device_id in self.end_stats:
                start = self.start_stats[device_id]
                end = self.end_stats[device_id]

                allocated_change = end.allocated_mb - start.allocated_mb
                reserved_change = end.reserved_mb - start.reserved_mb

                logger.info(
                    f"GPU {device_id} memory change for '{self.operation_name}': "
                    f"allocated {allocated_change:+.1f}MB, reserved {reserved_change:+.1f}MB"
                )

                # Warn about significant memory increases
                if allocated_change > 100:  # More than 100MB increase
                    logger.warning(
                        f"Significant memory increase detected for '{self.operation_name}': "
                        f"{allocated_change:.1f}MB allocated"
                    )


# Convenience functions for common GPU memory optimization patterns


@contextmanager
def gpu_memory_optimized_context(
    memory_fraction: float = 0.9, optimize_fragmentation: bool = True
):
    """Context manager for GPU memory optimized operations.

    Args:
        memory_fraction: GPU memory fraction to use
        optimize_fragmentation: Whether to optimize memory fragmentation
    """
    optimizer = GPUMemoryOptimizer()

    # Set memory fraction
    optimizer.set_memory_fraction(memory_fraction)

    # Optimize fragmentation if requested
    if optimize_fragmentation:
        optimizer.optimize_memory_fragmentation(aggressive=False)

    try:
        yield optimizer
    finally:
        # Final cleanup
        if optimize_fragmentation:
            optimizer.optimize_memory_fragmentation(aggressive=True)


def optimize_gpu_memory_for_training(
    model_size_mb: Optional[float] = None,
) -> GPUMemoryOptimizer:
    """Optimize GPU memory specifically for training workflows.

    Args:
        model_size_mb: Estimated model size in MB for optimization

    Returns:
        Configured GPU memory optimizer
    """
    optimizer = GPUMemoryOptimizer()

    if not torch.cuda.is_available():
        logger.warning("CUDA not available, GPU optimization skipped")
        return optimizer

    # Apply training-specific optimizations
    if model_size_mb is not None:
        # Reserve more memory for large models
        if model_size_mb > 1000:  # > 1GB model
            memory_fraction = 0.95
        elif model_size_mb > 500:  # > 500MB model
            memory_fraction = 0.9
        else:
            memory_fraction = 0.8

        optimizer.set_memory_fraction(memory_fraction)
        logger.info(
            f"GPU memory optimized for {model_size_mb:.1f}MB model with {memory_fraction:.1f} fraction"
        )

    # Clear initial fragmentation
    optimizer.optimize_memory_fragmentation(aggressive=True)

    return optimizer


def get_gpu_memory_summary() -> Dict[str, Any]:
    """Get a quick summary of GPU memory status.

    Returns:
        Dictionary with GPU memory summary
    """
    optimizer = GPUMemoryOptimizer()

    if not torch.cuda.is_available():
        return {"status": "cuda_not_available"}

    stats = optimizer.get_memory_stats()
    fragmented_devices = optimizer.detect_memory_fragmentation()
    recommendations = optimizer.get_memory_recommendations()

    return {
        "device_count": optimizer.device_count,
        "memory_stats": {dev_id: stats.to_dict() for dev_id, stats in stats.items()},
        "fragmented_devices": fragmented_devices,
        "recommendations": recommendations,
        "status": "available",
    }
