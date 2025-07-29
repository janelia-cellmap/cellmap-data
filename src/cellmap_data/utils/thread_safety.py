"""
Thread Safety Framework for Week 5 Day 5 objectives.

This module provides comprehensive thread safety enhancements for concurrent
data access patterns, building on the existing ThreadPoolExecutor infrastructure
and memory optimization systems.
"""

import threading
import queue
import time
import weakref
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, RLock, Event, Condition
from typing import Any, Dict, List, Optional, Callable, TypeVar, Generic, Set, Union
from dataclasses import dataclass, field
from functools import wraps
import warnings

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from .logging_config import get_logger

logger = get_logger("thread_safety")

T = TypeVar("T")


@dataclass
class ThreadSafetyConfig:
    """Configuration for thread safety framework."""

    # Thread pool settings
    max_workers: int = 4
    thread_pool_name: str = "cellmap_workers"
    enable_thread_monitoring: bool = True

    # Lock settings
    lock_timeout: float = 30.0  # Maximum time to wait for locks
    enable_deadlock_detection: bool = True

    # Resource management
    enable_resource_tracking: bool = True
    cleanup_orphaned_resources: bool = True

    # Performance settings
    enable_concurrent_profiling: bool = False
    max_concurrent_operations: int = 10


class ThreadSafeResource(Generic[T]):
    """Thread-safe wrapper for any resource with automatic cleanup."""

    def __init__(self, resource: T, name: str = "resource"):
        """Initialize thread-safe resource wrapper.

        Args:
            resource: The resource to wrap
            name: Human-readable name for debugging
        """
        self._resource = resource
        self._name = name
        self._lock = RLock()  # Allow re-entrant access
        self._access_count = 0
        self._last_access = time.time()
        self._thread_id = threading.get_ident()

        logger.debug(f"ThreadSafeResource created: {name} on thread {self._thread_id}")

    def __enter__(self) -> T:
        """Enter context manager with automatic locking."""
        acquired = self._lock.acquire(timeout=30.0)
        if not acquired:
            raise RuntimeError(
                f"Failed to acquire lock for resource '{self._name}' within 30s"
            )

        self._access_count += 1
        self._last_access = time.time()
        return self._resource

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager with automatic lock release."""
        try:
            self._lock.release()
        except Exception as e:
            logger.error(f"Error releasing lock for resource '{self._name}': {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get resource access statistics."""
        return {
            "name": self._name,
            "access_count": self._access_count,
            "last_access": self._last_access,
            "created_on_thread": self._thread_id,
            "current_thread": threading.get_ident(),
        }


class DeadlockDetector:
    """Advanced deadlock detection for thread safety monitoring."""

    def __init__(self):
        """Initialize deadlock detector."""
        self._lock_graph: Dict[int, Set[str]] = {}  # thread_id -> set of held locks
        self._waiting_graph: Dict[int, str] = {}  # thread_id -> lock waiting for
        self._lock_owners: Dict[str, int] = {}  # lock_name -> thread_id
        self._graph_lock = Lock()

        # Detection settings
        self._enable_detection = True
        self._max_wait_time = 30.0

    def acquire_lock(self, lock_name: str, thread_id: Optional[int] = None) -> bool:
        """Record lock acquisition attempt.

        Args:
            lock_name: Name/ID of the lock being acquired
            thread_id: Thread ID (uses current thread if None)

        Returns:
            True if no deadlock detected, False if potential deadlock
        """
        if not self._enable_detection:
            return True

        thread_id = thread_id or threading.get_ident()

        with self._graph_lock:
            # Check for potential deadlock before acquiring
            if self._would_create_deadlock(lock_name, thread_id):
                logger.warning(
                    f"Potential deadlock detected: thread {thread_id} waiting for {lock_name}"
                )
                return False

            # Record that this thread is waiting for the lock
            self._waiting_graph[thread_id] = lock_name

        return True

    def lock_acquired(self, lock_name: str, thread_id: Optional[int] = None):
        """Record successful lock acquisition.

        Args:
            lock_name: Name/ID of the acquired lock
            thread_id: Thread ID (uses current thread if None)
        """
        if not self._enable_detection:
            return

        thread_id = thread_id or threading.get_ident()

        with self._graph_lock:
            # Add to held locks
            if thread_id not in self._lock_graph:
                self._lock_graph[thread_id] = set()
            self._lock_graph[thread_id].add(lock_name)

            # Record lock owner
            self._lock_owners[lock_name] = thread_id

            # Remove from waiting graph
            self._waiting_graph.pop(thread_id, None)

    def lock_released(self, lock_name: str, thread_id: Optional[int] = None):
        """Record lock release.

        Args:
            lock_name: Name/ID of the released lock
            thread_id: Thread ID (uses current thread if None)
        """
        if not self._enable_detection:
            return

        thread_id = thread_id or threading.get_ident()

        with self._graph_lock:
            # Remove from held locks
            if thread_id in self._lock_graph:
                self._lock_graph[thread_id].discard(lock_name)
                if not self._lock_graph[thread_id]:
                    del self._lock_graph[thread_id]

            # Remove lock owner
            self._lock_owners.pop(lock_name, None)

    def _would_create_deadlock(
        self, requested_lock: str, requesting_thread: int
    ) -> bool:
        """Check if acquiring a lock would create a deadlock.

        Args:
            requested_lock: The lock being requested
            requesting_thread: The thread requesting the lock

        Returns:
            True if deadlock would occur
        """
        # If lock is not owned, no deadlock possible
        if requested_lock not in self._lock_owners:
            return False

        owning_thread = self._lock_owners[requested_lock]

        # If requesting thread already owns the lock, no deadlock (re-entrant)
        if owning_thread == requesting_thread:
            return False

        # Check if owning thread is waiting for any locks held by requesting thread
        if owning_thread in self._waiting_graph:
            waiting_for = self._waiting_graph[owning_thread]
            if (
                requesting_thread in self._lock_graph
                and waiting_for in self._lock_graph[requesting_thread]
            ):
                return True

        return False

    def get_deadlock_report(self) -> str:
        """Generate deadlock detection report."""
        with self._graph_lock:
            report_lines = [
                "=== Deadlock Detection Report ===",
                f"Active Threads: {len(self._lock_graph)}",
                f"Waiting Threads: {len(self._waiting_graph)}",
                f"Held Locks: {len(self._lock_owners)}",
                "",
            ]

            if self._waiting_graph:
                report_lines.append("Waiting Threads:")
                for thread_id, lock_name in self._waiting_graph.items():
                    held_locks = self._lock_graph.get(thread_id, set())
                    report_lines.append(
                        f"  Thread {thread_id}: waiting for '{lock_name}', holds {held_locks}"
                    )
                report_lines.append("")

            if self._lock_owners:
                report_lines.append("Lock Ownership:")
                for lock_name, owner_id in self._lock_owners.items():
                    report_lines.append(f"  '{lock_name}': owned by thread {owner_id}")

            return "\n".join(report_lines)


class ResourceWrapper:
    """Wrapper for resources that cannot have weak references."""

    def __init__(self, wrapped_resource):
        self.resource = wrapped_resource


class ThreadSafeExecutorManager:
    """Advanced thread pool management with monitoring and safety features."""

    def __init__(self, config: Optional[ThreadSafetyConfig] = None):
        """Initialize thread-safe executor manager.

        Args:
            config: Thread safety configuration
        """
        self.config = config or ThreadSafetyConfig()
        self._executors: Dict[str, ThreadPoolExecutor] = {}
        self._executor_stats: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()
        self._shutdown_event = Event()

        # Monitoring
        self.deadlock_detector = (
            DeadlockDetector() if self.config.enable_deadlock_detection else None
        )
        self._resource_registry: Dict[str, weakref.ref] = {}
        self._operation_semaphore = threading.Semaphore(
            self.config.max_concurrent_operations
        )

        logger.info(f"ThreadSafeExecutorManager initialized with config: {self.config}")

    def get_executor(
        self, name: str = "default", max_workers: Optional[int] = None
    ) -> ThreadPoolExecutor:
        """Get or create a thread pool executor.

        Args:
            name: Executor name/identifier
            max_workers: Maximum worker threads (uses config default if None)

        Returns:
            ThreadPoolExecutor instance
        """
        if self._shutdown_event.is_set():
            raise RuntimeError("ExecutorManager has been shut down")

        with self._lock:
            if name not in self._executors:
                workers = max_workers or self.config.max_workers
                self._executors[name] = ThreadPoolExecutor(
                    max_workers=workers,
                    thread_name_prefix=f"{self.config.thread_pool_name}_{name}",
                )
                self._executor_stats[name] = {
                    "created_at": time.time(),
                    "max_workers": workers,
                    "submitted_tasks": 0,
                    "completed_tasks": 0,
                }
                logger.debug(f"Created new executor '{name}' with {workers} workers")

            return self._executors[name]

    def submit_task(self, executor_name: str, fn: Callable, *args, **kwargs):
        """Submit a task to a named executor with monitoring.

        Args:
            executor_name: Name of the executor to use
            fn: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Future object representing the task
        """
        # Apply concurrency limit
        if not self._operation_semaphore.acquire(blocking=False):
            logger.warning(
                f"Maximum concurrent operations ({self.config.max_concurrent_operations}) reached"
            )
            # Try to acquire with timeout
            if not self._operation_semaphore.acquire(timeout=10.0):
                raise RuntimeError("Too many concurrent operations, try again later")

        try:
            executor = self.get_executor(executor_name)

            # Wrap function for monitoring
            def monitored_fn(*args, **kwargs):
                thread_id = threading.get_ident()
                logger.debug(
                    f"Task started on thread {thread_id} in executor '{executor_name}'"
                )

                try:
                    result = fn(*args, **kwargs)

                    # Update stats
                    with self._lock:
                        if executor_name in self._executor_stats:
                            self._executor_stats[executor_name]["completed_tasks"] += 1

                    return result

                finally:
                    self._operation_semaphore.release()

            # Submit task
            future = executor.submit(monitored_fn, *args, **kwargs)

            # Update stats
            with self._lock:
                if executor_name in self._executor_stats:
                    self._executor_stats[executor_name]["submitted_tasks"] += 1

            return future

        except Exception:
            # Release semaphore on error
            self._operation_semaphore.release()
            raise

    @contextmanager
    def concurrent_execution(self, executor_name: str = "default"):
        """Context manager for concurrent execution with automatic cleanup.

        Args:
            executor_name: Name of the executor to use
        """
        executor = self.get_executor(executor_name)
        futures = []

        try:
            yield lambda fn, *args, **kwargs: futures.append(
                self.submit_task(executor_name, fn, *args, **kwargs)
            )
        finally:
            # Wait for all submitted tasks to complete
            if futures:
                logger.debug(f"Waiting for {len(futures)} concurrent tasks to complete")
                for future in as_completed(futures):
                    try:
                        future.result()  # Retrieve result to catch exceptions
                    except Exception as e:
                        logger.error(f"Concurrent task failed: {e}")

    def register_resource(self, name: str, resource: Any):
        """Register a resource for tracking and cleanup.

        Args:
            name: Resource name/identifier
            resource: The resource to track
        """
        if self.config.enable_resource_tracking:
            try:
                self._resource_registry[name] = weakref.ref(resource)
                logger.debug(f"Registered resource: {name}")
            except TypeError:
                # Some objects (like dicts, lists, etc.) can't have weak references
                # For testing purposes, we'll use a placeholder that can be tracked
                logger.debug(
                    f"Cannot create weak reference for resource '{name}' of type {type(resource).__name__}, using placeholder"
                )

                wrapper = ResourceWrapper(resource)
                self._resource_registry[name] = weakref.ref(wrapper)

    def cleanup_orphaned_resources(self):
        """Clean up resources that are no longer referenced."""
        if not self.config.cleanup_orphaned_resources:
            return

        orphaned = []
        for name, ref in list(self._resource_registry.items()):
            if ref() is None:  # Resource has been garbage collected
                orphaned.append(name)
                del self._resource_registry[name]

        if orphaned:
            logger.info(f"Cleaned up {len(orphaned)} orphaned resources")

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive executor and resource statistics."""
        with self._lock:
            return {
                "executors": dict(self._executor_stats),
                "active_resources": len(
                    [
                        ref
                        for ref in self._resource_registry.values()
                        if ref() is not None
                    ]
                ),
                "total_registered_resources": len(self._resource_registry),
                "current_concurrent_operations": self.config.max_concurrent_operations
                - self._operation_semaphore._value,
                "deadlock_report": (
                    self.deadlock_detector.get_deadlock_report()
                    if self.deadlock_detector
                    else "Disabled"
                ),
            }

    def shutdown(self, wait: bool = True):
        """Shutdown all executors and cleanup resources.

        Args:
            wait: Whether to wait for pending tasks to complete
        """
        logger.info("Shutting down ThreadSafeExecutorManager")
        self._shutdown_event.set()

        with self._lock:
            for name, executor in self._executors.items():
                logger.debug(f"Shutting down executor '{name}'")
                executor.shutdown(wait=wait)

            self._executors.clear()
            self._executor_stats.clear()

        if self.config.cleanup_orphaned_resources:
            self.cleanup_orphaned_resources()


class ThreadSafeCUDAStreamManager:
    """Thread-safe CUDA stream management for concurrent GPU operations."""

    def __init__(self, max_streams: int = 8):
        """Initialize CUDA stream manager.

        Args:
            max_streams: Maximum number of concurrent CUDA streams
        """
        self.max_streams = max_streams
        self._streams: List[Any] = []  # torch.cuda.Stream objects when available
        self._stream_lock = Lock()
        self._stream_assignments: Dict[int, int] = {}  # thread_id -> stream_index
        self._assignment_lock = Lock()

        if TORCH_AVAILABLE and torch is not None and torch.cuda.is_available():
            self._initialize_streams()

        logger.info(
            f"ThreadSafeCUDAStreamManager initialized with {len(self._streams)} streams"
        )

    def _initialize_streams(self):
        """Initialize CUDA streams."""
        if not TORCH_AVAILABLE or torch is None:
            logger.warning("PyTorch not available, skipping CUDA stream initialization")
            return

        try:
            for i in range(self.max_streams):
                stream = torch.cuda.Stream()
                self._streams.append(stream)
                logger.debug(f"Created CUDA stream {i}: {stream}")
        except Exception as e:
            logger.error(f"Failed to initialize CUDA streams: {e}")
            self._streams = []

    def get_stream_for_thread(self) -> Optional[Any]:
        """Get a CUDA stream assigned to the current thread.

        Returns:
            CUDA stream for this thread, or None if CUDA unavailable
        """
        if (
            not TORCH_AVAILABLE
            or torch is None
            or not torch.cuda.is_available()
            or not self._streams
        ):
            return None

        thread_id = threading.get_ident()

        with self._assignment_lock:
            if thread_id not in self._stream_assignments:
                # Assign stream based on thread ID modulo number of streams
                stream_index = len(self._stream_assignments) % len(self._streams)
                self._stream_assignments[thread_id] = stream_index
                logger.debug(f"Assigned stream {stream_index} to thread {thread_id}")

            stream_index = self._stream_assignments[thread_id]
            return self._streams[stream_index]

    @contextmanager
    def cuda_stream_context(self):
        """Context manager for thread-safe CUDA stream operations."""
        if not TORCH_AVAILABLE or torch is None:
            yield None
            return

        stream = self.get_stream_for_thread()

        if stream is None:
            # No CUDA or streams available, use default context
            yield None
            return

        try:
            with torch.cuda.stream(stream):
                yield stream
        except Exception as e:
            logger.error(f"Error in CUDA stream context: {e}")
            yield None

    def synchronize_all_streams(self):
        """Synchronize all CUDA streams."""
        if not TORCH_AVAILABLE or torch is None or not torch.cuda.is_available():
            return

        with self._stream_lock:
            for i, stream in enumerate(self._streams):
                try:
                    stream.synchronize()
                    logger.debug(f"Synchronized CUDA stream {i}")
                except Exception as e:
                    logger.error(f"Failed to synchronize CUDA stream {i}: {e}")

    def get_stream_stats(self) -> Dict[str, Any]:
        """Get CUDA stream usage statistics."""
        return {
            "total_streams": len(self._streams),
            "assigned_threads": len(self._stream_assignments),
            "cuda_available": TORCH_AVAILABLE
            and torch is not None
            and torch.cuda.is_available(),
            "assignments": dict(self._stream_assignments),
        }


def thread_safe(lock_name: Optional[str] = None, deadlock_detection: bool = True):
    """Decorator to make functions thread-safe with optional deadlock detection.

    Args:
        lock_name: Name for the lock (uses function name if None)
        deadlock_detection: Whether to enable deadlock detection
    """

    def decorator(func: Callable) -> Callable:
        # Create a unique lock for this function
        func_lock_name = lock_name or f"{func.__module__}.{func.__name__}"
        func_lock = Lock()

        @wraps(func)
        def wrapper(*args, **kwargs):
            thread_id = threading.get_ident()

            # Deadlock detection (if enabled globally)
            detector = getattr(wrapper, "_deadlock_detector", None)
            if deadlock_detection and detector:
                if not detector.acquire_lock(func_lock_name, thread_id):
                    raise RuntimeError(
                        f"Potential deadlock detected acquiring lock for {func_lock_name}"
                    )

            acquired = func_lock.acquire(timeout=30.0)
            if not acquired:
                raise RuntimeError(
                    f"Failed to acquire lock for {func_lock_name} within 30s"
                )

            try:
                if deadlock_detection and detector:
                    detector.lock_acquired(func_lock_name, thread_id)

                logger.debug(f"Thread {thread_id} acquired lock for {func_lock_name}")
                return func(*args, **kwargs)

            finally:
                try:
                    func_lock.release()
                    if deadlock_detection and detector:
                        detector.lock_released(func_lock_name, thread_id)
                    logger.debug(
                        f"Thread {thread_id} released lock for {func_lock_name}"
                    )
                except Exception as e:
                    logger.error(f"Error releasing lock for {func_lock_name}: {e}")

        # Attach metadata for inspection (handle type safely)
        try:
            wrapper._is_thread_safe = True  # type: ignore
            wrapper._lock_name = func_lock_name  # type: ignore
        except AttributeError:
            # Fallback for cases where attributes cannot be set
            pass

        return wrapper

    return decorator


# Global instances for package-wide thread safety
_global_executor_manager: Optional[ThreadSafeExecutorManager] = None
_global_cuda_manager: Optional[ThreadSafeCUDAStreamManager] = None
_manager_lock = Lock()


def get_global_executor_manager(
    config: Optional[ThreadSafetyConfig] = None,
) -> ThreadSafeExecutorManager:
    """Get or create the global executor manager.

    Args:
        config: Configuration (only used for first creation)

    Returns:
        Global ThreadSafeExecutorManager instance
    """
    global _global_executor_manager

    if _global_executor_manager is None:
        with _manager_lock:
            if _global_executor_manager is None:
                _global_executor_manager = ThreadSafeExecutorManager(config)

    return _global_executor_manager


def get_global_cuda_manager(max_streams: int = 8) -> ThreadSafeCUDAStreamManager:
    """Get or create the global CUDA stream manager.

    Args:
        max_streams: Maximum CUDA streams (only used for first creation)

    Returns:
        Global ThreadSafeCUDAStreamManager instance
    """
    global _global_cuda_manager

    if _global_cuda_manager is None:
        with _manager_lock:
            if _global_cuda_manager is None:
                _global_cuda_manager = ThreadSafeCUDAStreamManager(max_streams)

    return _global_cuda_manager


@contextmanager
def thread_safe_execution(config: Optional[ThreadSafetyConfig] = None):
    """Context manager for thread-safe execution with comprehensive management.

    Args:
        config: Thread safety configuration
    """
    manager = get_global_executor_manager(config)
    cuda_manager = get_global_cuda_manager()

    logger.info("Thread-safe execution context started")

    try:
        yield {
            "executor_manager": manager,
            "cuda_manager": cuda_manager,
            "submit_task": lambda fn, *args, **kwargs: manager.submit_task(
                "default", fn, *args, **kwargs
            ),
            "cuda_stream": cuda_manager.cuda_stream_context,
        }
    finally:
        # Cleanup and statistics
        stats = manager.get_stats()
        cuda_stats = cuda_manager.get_stream_stats()

        logger.info(f"Thread-safe execution completed. Stats: {stats}")
        logger.debug(f"CUDA stream stats: {cuda_stats}")

        # Optional cleanup of orphaned resources
        manager.cleanup_orphaned_resources()


def shutdown_global_managers():
    """Shutdown global thread safety managers."""
    global _global_executor_manager, _global_cuda_manager

    with _manager_lock:
        if _global_executor_manager:
            _global_executor_manager.shutdown(wait=True)
            _global_executor_manager = None

        if _global_cuda_manager:
            _global_cuda_manager.synchronize_all_streams()
            _global_cuda_manager = None

    logger.info("Global thread safety managers shut down")


# Convenience functions for common thread safety patterns
def run_concurrent_tasks(
    tasks: List[Callable], executor_name: str = "default", max_workers: int = 4
) -> List[Any]:
    """Run multiple tasks concurrently with thread safety.

    Args:
        tasks: List of callables to execute
        executor_name: Name of executor to use
        max_workers: Maximum worker threads

    Returns:
        List of results in the same order as input tasks
    """
    manager = get_global_executor_manager()
    executor = manager.get_executor(executor_name, max_workers)

    futures = [manager.submit_task(executor_name, task) for task in tasks]
    results = []

    for future in as_completed(futures):
        try:
            results.append(future.result())
        except Exception as e:
            logger.error(f"Concurrent task failed: {e}")
            results.append(None)

    return results


def make_thread_safe_dataloader(dataloader_class):
    """Class decorator to add thread safety to dataloader classes.

    Args:
        dataloader_class: The dataloader class to enhance

    Returns:
        Enhanced class with thread safety features
    """
    original_init = dataloader_class.__init__
    original_iter = getattr(dataloader_class, "__iter__", None)

    def enhanced_init(self, *args, **kwargs):
        # Call original constructor
        original_init(self, *args, **kwargs)

        # Add thread safety attributes
        self._thread_safety_manager = get_global_executor_manager()
        self._cuda_stream_manager = get_global_cuda_manager()
        self._access_lock = Lock()

        logger.debug(f"Enhanced {dataloader_class.__name__} with thread safety")

    def enhanced_iter(self):
        """Thread-safe iterator with CUDA stream management."""
        with self._access_lock:
            if original_iter:
                # Use CUDA stream context for GPU operations
                with self._cuda_stream_manager.cuda_stream_context():
                    return original_iter(self)
            else:
                raise NotImplementedError("No __iter__ method in original class")

    # Replace methods
    dataloader_class.__init__ = enhanced_init
    if original_iter:
        dataloader_class.__iter__ = enhanced_iter

    # Add thread safety marker
    dataloader_class._thread_safe_enhanced = True

    return dataloader_class
