"""
Week 5 Day 5: Enhanced Thread Safety Integration

This module enhances the existing thread safety framework with advanced integration
features for memory optimization, concurrent data loading patterns, and comprehensive
performance monitoring in multi-threaded scenarios.
"""

import time
import threading
import weakref
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Callable, Set, Tuple
from dataclasses import dataclass, field
import warnings

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

import numpy as np

from .thread_safety import (
    ThreadSafetyConfig,
    ThreadSafeExecutorManager,
    ThreadSafeCUDAStreamManager,
    ThreadSafeResource,
    get_global_executor_manager,
    get_global_cuda_manager,
)
from .memory_optimization import AdvancedMemoryManager, MemoryOptimizationConfig
from .logging_config import get_logger

logger = get_logger("enhanced_thread_safety")


@dataclass
class EnhancedThreadSafetyConfig(ThreadSafetyConfig):
    """Enhanced configuration for advanced thread safety features."""

    # Memory-aware threading
    enable_memory_aware_threading: bool = True
    memory_threshold_for_threading: float = 1024.0  # MB
    thread_memory_limit_mb: float = 512.0

    # Advanced concurrency patterns
    enable_priority_queuing: bool = True
    max_priority_levels: int = 5
    enable_load_balancing: bool = True

    # Performance monitoring
    enable_detailed_profiling: bool = True
    profiling_sample_rate: float = 0.1  # 10% of operations
    enable_bottleneck_detection: bool = True

    # Integration features
    integrate_with_memory_manager: bool = True
    enable_cuda_memory_threading: bool = True

    # Advanced error handling
    enable_thread_error_recovery: bool = True
    max_retry_attempts: int = 3
    retry_backoff_factor: float = 1.5


class MemoryAwareThreadManager:
    """Thread manager that integrates with memory optimization system."""

    def __init__(self, config: Optional[EnhancedThreadSafetyConfig] = None):
        """Initialize memory-aware thread manager.

        Args:
            config: Enhanced thread safety configuration
        """
        self.config = config or EnhancedThreadSafetyConfig()
        self.executor_manager = get_global_executor_manager()
        self.memory_manager = None

        # Error handling callback (can be set by parent manager)
        self.error_callback: Optional[Callable[[str, Exception], None]] = None

        # Memory monitoring
        self._memory_usage_by_thread: Dict[int, float] = {}
        self._memory_lock = threading.Lock()

        # Performance tracking
        self._performance_metrics: Dict[str, List[float]] = {
            "execution_times": [],
            "memory_usage": [],
            "thread_utilization": [],
            "bottleneck_events": [],
        }
        self._metrics_lock = threading.Lock()

        if self.config.integrate_with_memory_manager:
            self._initialize_memory_integration()

        logger.info(f"MemoryAwareThreadManager initialized with config: {self.config}")

    def set_error_callback(self, callback: Callable[[str, Exception], None]):
        """Set error callback for handling task failures.

        Args:
            callback: Function to call when task errors occur
        """
        self.error_callback = callback

    def _initialize_memory_integration(self):
        """Initialize integration with memory management system."""
        try:
            memory_config = MemoryOptimizationConfig(
                max_memory_mb=self.config.thread_memory_limit_mb,
                enable_streaming=True,
                enable_gpu_memory_optimization=self.config.enable_cuda_memory_threading,
            )
            self.memory_manager = AdvancedMemoryManager(memory_config)
            logger.debug("Memory management integration initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize memory integration: {e}")
            self.memory_manager = None

    @contextmanager
    def memory_aware_execution(self, task_name: str = "task"):
        """Context manager for memory-aware task execution.

        Args:
            task_name: Name of the task for monitoring
        """
        thread_id = threading.get_ident()
        start_time = time.time()
        initial_memory = self._get_thread_memory_usage(thread_id)

        try:
            # Check memory threshold before execution
            if self._should_throttle_execution():
                logger.warning(f"Throttling execution due to memory pressure")
                time.sleep(0.1)  # Brief pause to allow memory cleanup

            # Apply memory management context if available
            if self.memory_manager:
                with self.memory_manager.memory_efficient_processing("thread_task"):
                    yield
            else:
                yield

        finally:
            # Record performance metrics
            execution_time = time.time() - start_time
            final_memory = self._get_thread_memory_usage(thread_id)
            memory_delta = final_memory - initial_memory

            self._record_performance_metrics(task_name, execution_time, memory_delta)

    def submit_memory_aware_task(
        self,
        executor_name: str,
        task_func: Callable,
        *args,
        priority: int = 0,
        memory_limit_mb: Optional[float] = None,
        **kwargs,
    ):
        """Submit a task with memory awareness and priority.

        Args:
            executor_name: Name of executor to use
            task_func: Function to execute
            *args: Positional arguments for task
            priority: Task priority (0 = highest)
            memory_limit_mb: Memory limit for this task
            **kwargs: Keyword arguments for task

        Returns:
            Future object for the task
        """

        def wrapped_task(*args, **kwargs):
            task_name = getattr(task_func, "__name__", "anonymous_task")

            try:
                with self.memory_aware_execution(task_name):
                    # Apply per-task memory limit if specified
                    if memory_limit_mb and self.memory_manager:
                        # Use the general context with memory monitoring
                        with self.memory_manager.memory_efficient_processing(
                            f"limited_task_{memory_limit_mb}mb"
                        ):
                            return task_func(*args, **kwargs)
                    else:
                        return task_func(*args, **kwargs)
            except Exception as e:
                # Call error callback if set
                if self.error_callback:
                    self.error_callback(executor_name, e)
                raise  # Re-raise the exception

        # Submit with priority handling
        if self.config.enable_priority_queuing:
            return self._submit_with_priority(
                executor_name, wrapped_task, priority, *args, **kwargs
            )
        else:
            return self.executor_manager.submit_task(
                executor_name, wrapped_task, *args, **kwargs
            )

    def _submit_with_priority(
        self, executor_name: str, task_func: Callable, priority: int, *args, **kwargs
    ):
        """Submit task with priority queuing."""
        # For now, implement simple priority by delaying lower priority tasks
        if priority > 0:
            delay = priority * 0.01  # 10ms delay per priority level

            def delayed_task(*args, **kwargs):
                time.sleep(delay)
                return task_func(*args, **kwargs)

            return self.executor_manager.submit_task(
                executor_name, delayed_task, *args, **kwargs
            )
        else:
            return self.executor_manager.submit_task(
                executor_name, task_func, *args, **kwargs
            )

    def _should_throttle_execution(self) -> bool:
        """Check if execution should be throttled due to memory pressure."""
        if not self.config.enable_memory_aware_threading:
            return False

        try:
            import psutil

            memory_percent = psutil.virtual_memory().percent
            return memory_percent > 85.0  # Throttle if memory usage > 85%
        except ImportError:
            return False

    def _get_thread_memory_usage(self, thread_id: int) -> float:
        """Get memory usage for a specific thread (approximation)."""
        try:
            import psutil
            import os

            process = psutil.Process(os.getpid())
            # Rough approximation of per-thread memory usage
            total_memory = process.memory_info().rss / 1024 / 1024  # MB
            thread_count = process.num_threads()
            return total_memory / max(thread_count, 1)
        except ImportError:
            return 0.0

    def _record_performance_metrics(
        self, task_name: str, execution_time: float, memory_delta: float
    ):
        """Record performance metrics for analysis."""
        if not self.config.enable_detailed_profiling:
            return

        # Sample based on configured rate
        if np.random.random() > self.config.profiling_sample_rate:
            return

        with self._metrics_lock:
            self._performance_metrics["execution_times"].append(execution_time)
            self._performance_metrics["memory_usage"].append(memory_delta)

            # Detect potential bottlenecks
            if execution_time > 1.0:  # Tasks taking more than 1 second
                self._performance_metrics["bottleneck_events"].append(time.time())
                logger.warning(
                    f"Potential bottleneck detected in task '{task_name}': {execution_time:.2f}s"
                )

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        with self._metrics_lock:
            metrics = dict(self._performance_metrics)

        report = {
            "total_tasks_sampled": len(metrics["execution_times"]),
            "average_execution_time": (
                np.mean(metrics["execution_times"]) if metrics["execution_times"] else 0
            ),
            "max_execution_time": (
                np.max(metrics["execution_times"]) if metrics["execution_times"] else 0
            ),
            "average_memory_delta": (
                np.mean(metrics["memory_usage"]) if metrics["memory_usage"] else 0
            ),
            "max_memory_delta": (
                np.max(metrics["memory_usage"]) if metrics["memory_usage"] else 0
            ),
            "bottleneck_events_count": len(metrics["bottleneck_events"]),
            "memory_manager_available": self.memory_manager is not None,
        }

        return report

    def optimize_thread_allocation(self) -> Dict[str, int]:
        """Dynamically optimize thread allocation based on performance metrics."""
        if not self.config.enable_load_balancing:
            return {}

        performance_report = self.get_performance_report()

        # Simple load balancing logic
        recommendations = {}

        if performance_report["bottleneck_events_count"] > 10:
            recommendations["reduce_max_workers"] = max(2, self.config.max_workers - 1)
            logger.info("Recommending thread reduction due to bottlenecks")
        elif performance_report["average_execution_time"] < 0.1:
            recommendations["increase_max_workers"] = min(
                8, self.config.max_workers + 1
            )
            logger.info("Recommending thread increase due to fast execution")

        return recommendations


class ConcurrentDataLoadingManager:
    """Enhanced manager for concurrent data loading with thread safety."""

    def __init__(self, config: Optional[EnhancedThreadSafetyConfig] = None):
        """Initialize concurrent data loading manager.

        Args:
            config: Enhanced thread safety configuration
        """
        self.config = config or EnhancedThreadSafetyConfig()
        self.thread_manager = MemoryAwareThreadManager(config)
        self.cuda_manager = get_global_cuda_manager()

        # Data loading coordination
        self._active_loaders: Set[weakref.ref] = set()
        self._loader_lock = threading.Lock()

        # Batch coordination
        self._batch_queues: Dict[str, List[Any]] = {}
        self._queue_locks: Dict[str, threading.Lock] = {}

        logger.info("ConcurrentDataLoadingManager initialized")

    def register_dataloader(self, dataloader):
        """Register a dataloader for concurrent management.

        Args:
            dataloader: The dataloader instance to register
        """
        with self._loader_lock:
            loader_ref = weakref.ref(dataloader)
            self._active_loaders.add(loader_ref)

            # Initialize batch queue for this loader
            loader_id = str(id(dataloader))
            self._batch_queues[loader_id] = []
            self._queue_locks[loader_id] = threading.Lock()

            logger.debug(f"Registered dataloader: {loader_id}")

    def unregister_dataloader(self, dataloader):
        """Unregister a dataloader.

        Args:
            dataloader: The dataloader instance to unregister
        """
        with self._loader_lock:
            loader_id = str(id(dataloader))

            # Remove from active loaders
            self._active_loaders = {
                ref for ref in self._active_loaders if ref() is not dataloader
            }

            # Cleanup batch queue
            if loader_id in self._batch_queues:
                del self._batch_queues[loader_id]
            if loader_id in self._queue_locks:
                del self._queue_locks[loader_id]

            logger.debug(f"Unregistered dataloader: {loader_id}")

    @contextmanager
    def concurrent_batch_loading(self, dataloader, batch_indices: List[int]):
        """Context manager for concurrent batch loading.

        Args:
            dataloader: The dataloader instance
            batch_indices: List of batch indices to load concurrently
        """
        loader_id = str(id(dataloader))
        futures = []  # Initialize futures list

        try:
            # Submit concurrent loading tasks
            for idx in batch_indices:
                future = self.thread_manager.submit_memory_aware_task(
                    f"loader_{loader_id}",
                    self._load_single_batch,
                    dataloader,
                    idx,
                    priority=0,  # High priority for data loading
                )
                futures.append((idx, future))

            # Yield a generator that retrieves results in order
            def batch_generator():
                # Sort futures by index to maintain order
                sorted_futures = sorted(futures, key=lambda x: x[0])

                for idx, future in sorted_futures:
                    try:
                        batch_data = future.result(timeout=30.0)
                        yield idx, batch_data
                    except Exception as e:
                        logger.error(f"Failed to load batch {idx}: {e}")
                        yield idx, None

            yield batch_generator()

        except Exception as e:
            logger.error(f"Error in concurrent batch loading: {e}")
            # Cancel remaining futures
            for _, future in futures:
                future.cancel()
            raise

    def _load_single_batch(self, dataloader, batch_idx: int):
        """Load a single batch with CUDA stream awareness.

        Args:
            dataloader: The dataloader instance
            batch_idx: Batch index to load

        Returns:
            Loaded batch data
        """
        try:
            # Use CUDA stream context for GPU operations
            with self.cuda_manager.cuda_stream_context() as stream:
                try:
                    return dataloader[batch_idx]
                except (TypeError, AttributeError, KeyError, IndexError) as e:
                    raise ValueError(f"Dataloader does not support indexing: {e}")

        except Exception as e:
            logger.error(f"Error loading batch {batch_idx}: {e}")
            raise

    def get_loader_statistics(self) -> Dict[str, Any]:
        """Return statistics for all registered dataloaders."""
        with self._loader_lock:
            # Use batch queues as the primary source of truth for active loaders
            # This is more reliable than weak references which may be garbage collected
            active_loaders = len(self._batch_queues)

            # Also clean up any dead weak references while we're here
            self._active_loaders = {
                ref for ref in self._active_loaders if ref() is not None
            }

            return {
                "active_loaders": active_loaders,
                "total_batch_queues": len(self._batch_queues),
                "performance_report": self.thread_manager.get_performance_report(),
            }


class ThreadSafetyEnhancementManager:
    """Main manager for all thread safety enhancements."""

    def __init__(self, config: Optional[EnhancedThreadSafetyConfig] = None):
        """Initialize thread safety enhancement manager.

        Args:
            config: Enhanced thread safety configuration
        """
        self.config = config or EnhancedThreadSafetyConfig()

        # Component managers
        self.memory_aware_manager = MemoryAwareThreadManager(config)
        self.concurrent_loader_manager = ConcurrentDataLoadingManager(config)

        # Set up error callback for memory aware manager
        self.memory_aware_manager.set_error_callback(self._handle_context_error)

        # Error recovery
        self._error_counts: Dict[str, int] = {}
        self._error_lock = threading.Lock()

        logger.info("ThreadSafetyEnhancementManager initialized")

    @contextmanager
    def enhanced_thread_safety_context(self, context_name: str = "default"):
        """Comprehensive thread safety context with all enhancements.

        Args:
            context_name: Name for this context (for monitoring)
        """
        start_time = time.time()

        try:
            logger.info(f"Enhanced thread safety context '{context_name}' started")

            # Provide access to all enhanced features
            yield {
                "memory_aware_manager": self.memory_aware_manager,
                "concurrent_loader_manager": self.concurrent_loader_manager,
                "submit_memory_aware_task": self.memory_aware_manager.submit_memory_aware_task,
                "concurrent_batch_loading": self.concurrent_loader_manager.concurrent_batch_loading,
                "register_dataloader": self.concurrent_loader_manager.register_dataloader,
                "performance_report": self.get_comprehensive_performance_report,
            }

        except Exception as e:
            self._handle_context_error(context_name, e)
            raise

        finally:
            execution_time = time.time() - start_time
            logger.info(
                f"Enhanced thread safety context '{context_name}' completed in {execution_time:.2f}s"
            )

    def _handle_context_error(self, context_name: str, error: Exception):
        """Handle errors in thread safety contexts with recovery.

        Args:
            context_name: Name of the context where error occurred
            error: The exception that occurred
        """
        if not self.config.enable_thread_error_recovery:
            return

        with self._error_lock:
            error_key = f"{context_name}:{type(error).__name__}"
            self._error_counts[error_key] = self._error_counts.get(error_key, 0) + 1

            if self._error_counts[error_key] > self.config.max_retry_attempts:
                logger.error(f"Max retry attempts exceeded for {error_key}: {error}")
            else:
                logger.warning(
                    f"Recoverable error in {context_name}: {error} (attempt {self._error_counts[error_key]})"
                )

    def get_comprehensive_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report across all components."""
        return {
            "memory_aware_performance": self.memory_aware_manager.get_performance_report(),
            "dataloader_statistics": self.concurrent_loader_manager.get_loader_statistics(),
            "error_recovery_stats": dict(self._error_counts),
            "configuration": {
                "memory_aware_threading": self.config.enable_memory_aware_threading,
                "priority_queuing": self.config.enable_priority_queuing,
                "detailed_profiling": self.config.enable_detailed_profiling,
                "memory_integration": self.config.integrate_with_memory_manager,
            },
        }

    def optimize_all_components(self) -> Dict[str, Any]:
        """Optimize all thread safety components based on current performance."""
        optimizations = {}

        # Optimize thread allocation
        thread_optimizations = self.memory_aware_manager.optimize_thread_allocation()
        if thread_optimizations:
            optimizations["thread_allocation"] = thread_optimizations

        # Get comprehensive performance report
        performance_report = self.get_comprehensive_performance_report()
        optimizations["current_performance"] = performance_report

        logger.info(f"Thread safety optimization completed: {optimizations}")
        return optimizations


# Global enhanced thread safety manager
_global_enhancement_manager: Optional[ThreadSafetyEnhancementManager] = None
_enhancement_lock = threading.Lock()


def get_global_thread_safety_enhancements(
    config: Optional[EnhancedThreadSafetyConfig] = None,
) -> ThreadSafetyEnhancementManager:
    """Get or create the global thread safety enhancement manager.

    Args:
        config: Configuration (only used for first creation)

    Returns:
        Global ThreadSafetyEnhancementManager instance
    """
    global _global_enhancement_manager

    if _global_enhancement_manager is None:
        with _enhancement_lock:
            if _global_enhancement_manager is None:
                _global_enhancement_manager = ThreadSafetyEnhancementManager(config)

    return _global_enhancement_manager


@contextmanager
def enhanced_thread_safety_execution(
    config: Optional[EnhancedThreadSafetyConfig] = None, context_name: str = "default"
):
    """Context manager for enhanced thread safety execution.

    Args:
        config: Enhanced thread safety configuration
        context_name: Name for this execution context
    """
    manager = get_global_thread_safety_enhancements(config)

    with manager.enhanced_thread_safety_context(context_name) as context:
        yield context


def optimize_thread_safety_globally():
    """Optimize thread safety settings globally based on current performance."""
    manager = get_global_thread_safety_enhancements()
    return manager.optimize_all_components()


def shutdown_enhanced_thread_safety():
    """Shutdown enhanced thread safety manager."""
    global _global_enhancement_manager

    with _enhancement_lock:
        if _global_enhancement_manager:
            # Perform any necessary cleanup
            logger.info("Shutting down enhanced thread safety manager")
            _global_enhancement_manager = None
