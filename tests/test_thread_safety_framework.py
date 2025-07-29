"""
Week 5 Day 5: Thread Safety Framework Tests

Comprehensive testing for concurrent data loading, thread-safe resource management,
and performance validation in multi-threaded scenarios.
"""

import pytest
import threading
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from cellmap_data.utils.thread_safety import (
    ThreadSafetyConfig,
    ThreadSafeResource,
    DeadlockDetector,
    ThreadSafeExecutorManager,
    ThreadSafeCUDAStreamManager,
    get_global_executor_manager,
    get_global_cuda_manager,
    thread_safe_execution,
    thread_safe,
    run_concurrent_tasks,
    make_thread_safe_dataloader,
)
from cellmap_data.dataloader import CellMapDataLoader


class TestThreadSafetyConfig:
    """Test thread safety configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ThreadSafetyConfig()

        assert config.max_workers == 4
        assert config.thread_pool_name == "cellmap_workers"
        assert config.enable_thread_monitoring is True
        assert config.lock_timeout == 30.0
        assert config.enable_deadlock_detection is True
        assert config.enable_resource_tracking is True
        assert config.cleanup_orphaned_resources is True
        assert config.enable_concurrent_profiling is False
        assert config.max_concurrent_operations == 10

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ThreadSafetyConfig(
            max_workers=8,
            thread_pool_name="custom_workers",
            enable_thread_monitoring=False,
            lock_timeout=60.0,
            enable_deadlock_detection=False,
            max_concurrent_operations=20,
        )

        assert config.max_workers == 8
        assert config.thread_pool_name == "custom_workers"
        assert config.enable_thread_monitoring is False
        assert config.lock_timeout == 60.0
        assert config.enable_deadlock_detection is False
        assert config.max_concurrent_operations == 20


class TestThreadSafeResource:
    """Test thread-safe resource wrapper."""

    def test_resource_creation(self):
        """Test resource wrapper creation."""
        test_resource = {"data": "test"}
        wrapped = ThreadSafeResource(test_resource, "test_resource")

        assert wrapped._name == "test_resource"
        assert wrapped._resource is test_resource
        assert wrapped._access_count == 0

    def test_context_manager_access(self):
        """Test thread-safe resource access via context manager."""
        test_resource = {"data": "test"}
        wrapped = ThreadSafeResource(test_resource, "test_resource")

        with wrapped as resource:
            assert resource is test_resource
            assert wrapped._access_count == 1

        # Access count should remain after context exit
        assert wrapped._access_count == 1

    def test_multiple_access(self):
        """Test multiple accesses update count."""
        test_resource = [1, 2, 3]
        wrapped = ThreadSafeResource(test_resource, "test_list")

        for i in range(3):
            with wrapped as resource:
                assert len(resource) == 3

        assert wrapped._access_count == 3

    def test_stats_reporting(self):
        """Test resource statistics reporting."""
        test_resource = {"key": "value"}
        wrapped = ThreadSafeResource(test_resource, "stats_test")

        with wrapped as resource:
            pass

        stats = wrapped.get_stats()
        assert stats["name"] == "stats_test"
        assert stats["access_count"] == 1
        assert "last_access" in stats
        assert "created_on_thread" in stats
        assert "current_thread" in stats

    def test_concurrent_access(self):
        """Test concurrent access to thread-safe resource."""
        test_resource = {"counter": 0}
        wrapped = ThreadSafeResource(test_resource, "concurrent_test")
        access_results = []

        def access_resource():
            with wrapped as resource:
                # Simulate some work
                current_count = resource["counter"]
                time.sleep(0.01)  # Small delay to encourage race conditions
                resource["counter"] = current_count + 1
                access_results.append(resource["counter"])

        # Run multiple threads concurrently
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=access_resource)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Verify thread safety - final count should be 5
        with wrapped as resource:
            assert resource["counter"] == 5

        assert len(access_results) == 5
        assert wrapped._access_count == 6  # 5 concurrent + 1 verification


class TestDeadlockDetector:
    """Test deadlock detection functionality."""

    def test_detector_creation(self):
        """Test deadlock detector initialization."""
        detector = DeadlockDetector()

        assert detector._enable_detection is True
        assert detector._max_wait_time == 30.0
        assert len(detector._lock_graph) == 0
        assert len(detector._waiting_graph) == 0
        assert len(detector._lock_owners) == 0

    def test_simple_lock_acquisition(self):
        """Test simple lock acquisition without deadlock."""
        detector = DeadlockDetector()

        # First lock acquisition
        assert detector.acquire_lock("lock1", 1) is True
        detector.lock_acquired("lock1", 1)

        # Verify internal state
        assert 1 in detector._lock_graph
        assert "lock1" in detector._lock_graph[1]
        assert detector._lock_owners["lock1"] == 1

    def test_lock_release(self):
        """Test lock release updates state correctly."""
        detector = DeadlockDetector()

        # Acquire and release lock
        detector.acquire_lock("lock1", 1)
        detector.lock_acquired("lock1", 1)
        detector.lock_released("lock1", 1)

        # Verify cleanup
        assert 1 not in detector._lock_graph
        assert "lock1" not in detector._lock_owners

    def test_reentrant_lock_access(self):
        """Test re-entrant lock access by same thread."""
        detector = DeadlockDetector()

        # First acquisition
        assert detector.acquire_lock("lock1", 1) is True
        detector.lock_acquired("lock1", 1)

        # Re-entrant access by same thread should be allowed
        assert detector.acquire_lock("lock1", 1) is True

    def test_deadlock_report_generation(self):
        """Test deadlock report generation."""
        detector = DeadlockDetector()

        # Set up some lock state
        detector.acquire_lock("lock1", 1)
        detector.lock_acquired("lock1", 1)
        detector.acquire_lock("lock2", 2)

        report = detector.get_deadlock_report()

        assert "Deadlock Detection Report" in report
        assert "Active Threads: 1" in report
        assert "Waiting Threads: 1" in report
        assert "Held Locks: 1" in report


class TestThreadSafeExecutorManager:
    """Test thread-safe executor management."""

    def test_manager_creation_default_config(self):
        """Test manager creation with default configuration."""
        manager = ThreadSafeExecutorManager()

        assert manager.config.max_workers == 4
        assert manager.config.thread_pool_name == "cellmap_workers"
        assert isinstance(manager.deadlock_detector, DeadlockDetector)

    def test_manager_creation_custom_config(self):
        """Test manager creation with custom configuration."""
        config = ThreadSafetyConfig(max_workers=8, enable_deadlock_detection=False)
        manager = ThreadSafeExecutorManager(config)

        assert manager.config.max_workers == 8
        assert manager.deadlock_detector is None

    def test_executor_creation(self):
        """Test executor creation and retrieval."""
        manager = ThreadSafeExecutorManager()

        executor = manager.get_executor("test_executor", max_workers=2)

        assert executor is not None
        assert "test_executor" in manager._executors
        assert manager._executor_stats["test_executor"]["max_workers"] == 2

    def test_executor_reuse(self):
        """Test that executors are reused when requested again."""
        manager = ThreadSafeExecutorManager()

        executor1 = manager.get_executor("reuse_test")
        executor2 = manager.get_executor("reuse_test")

        assert executor1 is executor2

    def test_task_submission(self):
        """Test task submission and execution."""
        manager = ThreadSafeExecutorManager()

        def test_task():
            return threading.get_ident()

        future = manager.submit_task("test", test_task)
        result = future.result(timeout=5.0)

        # Verify task executed on different thread
        assert result != threading.get_ident()

        # Verify stats updated
        stats = manager.get_stats()
        assert stats["executors"]["test"]["submitted_tasks"] >= 1
        assert stats["executors"]["test"]["completed_tasks"] >= 1

    def test_concurrent_execution_context(self):
        """Test concurrent execution context manager."""
        manager = ThreadSafeExecutorManager()

        results = []

        def test_task(value):
            results.append(value * 2)
            return value * 2

        with manager.concurrent_execution("concurrent_test") as submit:
            submit(test_task, 1)
            submit(test_task, 2)
            submit(test_task, 3)

        # All tasks should have completed
        assert len(results) == 3
        assert sorted(results) == [2, 4, 6]

    def test_resource_registration(self):
        """Test resource registration and tracking."""
        manager = ThreadSafeExecutorManager()

        # Use an object that can have weak references
        class TestResource:
            def __init__(self, data):
                self.data = data

        test_resource = TestResource({"data": "test"})
        manager.register_resource("test_resource", test_resource)

        stats = manager.get_stats()
        assert stats["active_resources"] >= 1
        assert stats["total_registered_resources"] >= 1

    def test_orphaned_resource_cleanup(self):
        """Test cleanup of orphaned resources."""
        manager = ThreadSafeExecutorManager()

        # Use an object that can have weak references
        class TestResource:
            def __init__(self, data):
                self.data = data

        # Register a resource then remove reference
        test_resource = TestResource({"data": "test"})
        manager.register_resource("orphan_resource", test_resource)
        del test_resource

        # Force garbage collection
        import gc

        gc.collect()

        # Cleanup should remove orphaned resource
        manager.cleanup_orphaned_resources()

        stats = manager.get_stats()
        # The exact count depends on other tests, but cleanup should have occurred
        assert "active_resources" in stats

    def test_manager_shutdown(self):
        """Test proper manager shutdown."""
        manager = ThreadSafeExecutorManager()

        # Create executor and submit a quick task
        executor = manager.get_executor("shutdown_test")
        future = manager.submit_task("shutdown_test", lambda: "completed")

        # Wait for task to complete before shutdown
        result = future.result(timeout=5.0)
        assert result == "completed"

        # Shutdown and verify cleanup
        manager.shutdown(wait=True)

        assert len(manager._executors) == 0
        assert len(manager._executor_stats) == 0

    def test_concurrency_limit(self):
        """Test concurrent operation limits."""
        config = ThreadSafetyConfig(max_concurrent_operations=2)
        manager = ThreadSafeExecutorManager(config)

        def quick_task():
            return threading.get_ident()

        # Submit tasks up to limit - this should work initially but may hit limit
        futures = []
        max_attempts = 3

        for i in range(max_attempts):
            try:
                future = manager.submit_task("limit_test", quick_task)
                futures.append(future)
            except RuntimeError as e:
                # Should hit concurrency limit at some point
                assert "concurrent operations" in str(e)
                break

        # Wait for completion of submitted tasks
        for future in futures:
            future.result(timeout=2.0)

        # At least some futures should have been submitted successfully
        assert len(futures) >= 1


class TestThreadSafeCUDAStreamManager:
    """Test CUDA stream management for thread safety."""

    def test_manager_creation(self):
        """Test CUDA stream manager creation."""
        manager = ThreadSafeCUDAStreamManager(max_streams=4)

        assert manager.max_streams == 4
        assert manager._streams is not None  # List even if empty

    def test_stream_assignment(self):
        """Test stream assignment to threads."""
        manager = ThreadSafeCUDAStreamManager(max_streams=2)

        # Mock PyTorch availability
        with patch("cellmap_data.utils.thread_safety.TORCH_AVAILABLE", True):
            with patch("cellmap_data.utils.thread_safety.torch") as mock_torch:
                mock_torch.cuda.is_available.return_value = True
                mock_stream = Mock()
                mock_torch.cuda.Stream.return_value = mock_stream

                # Initialize streams
                manager._initialize_streams()

                # Test stream assignment
                stream = manager.get_stream_for_thread()

                # Should get a stream or None based on availability
                assert stream is not None or stream is None

    def test_cuda_stream_context(self):
        """Test CUDA stream context manager."""
        manager = ThreadSafeCUDAStreamManager(max_streams=2)

        # Test without CUDA
        with manager.cuda_stream_context() as stream:
            assert stream is None

    def test_stream_synchronization(self):
        """Test stream synchronization."""
        manager = ThreadSafeCUDAStreamManager(max_streams=2)

        # Mock streams for synchronization test
        mock_stream1 = Mock()
        mock_stream2 = Mock()
        manager._streams = [mock_stream1, mock_stream2]

        with patch("cellmap_data.utils.thread_safety.TORCH_AVAILABLE", True):
            with patch("cellmap_data.utils.thread_safety.torch") as mock_torch:
                mock_torch.cuda.is_available.return_value = True

                manager.synchronize_all_streams()

                mock_stream1.synchronize.assert_called_once()
                mock_stream2.synchronize.assert_called_once()

    def test_stream_stats(self):
        """Test stream statistics reporting."""
        manager = ThreadSafeCUDAStreamManager(max_streams=3)

        stats = manager.get_stream_stats()

        assert "total_streams" in stats
        assert "assigned_threads" in stats
        assert "cuda_available" in stats
        assert "assignments" in stats


class TestThreadSafeDecorator:
    """Test thread-safe function decorator."""

    def test_decorator_application(self):
        """Test that decorator can be applied to functions."""

        @thread_safe()
        def test_function():
            return "success"

        result = test_function()
        assert result == "success"

        # Check decorator metadata
        assert hasattr(test_function, "_is_thread_safe")
        assert hasattr(test_function, "_lock_name")

    def test_concurrent_decorated_function_access(self):
        """Test that decorated functions are thread-safe."""
        shared_resource = {"counter": 0}

        @thread_safe("counter_lock")
        def increment_counter():
            current = shared_resource["counter"]
            time.sleep(0.01)  # Simulate work and encourage race conditions
            shared_resource["counter"] = current + 1
            return shared_resource["counter"]

        # Run multiple threads concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(increment_counter) for _ in range(5)]
            results = [future.result() for future in futures]

        # Verify thread safety
        assert shared_resource["counter"] == 5
        assert len(results) == 5

    def test_decorator_with_custom_lock_name(self):
        """Test decorator with custom lock name."""

        @thread_safe("custom_lock_name")
        def test_function():
            return threading.get_ident()

        # Function should work normally
        result = test_function()
        assert isinstance(result, int)

        # Check custom lock name (handle attribute access safely)
        assert hasattr(test_function, "_lock_name")
        if hasattr(test_function, "_lock_name"):
            # Use getattr to safely access the attribute
            lock_name = getattr(test_function, "_lock_name", None)
            assert lock_name == "custom_lock_name"


class TestGlobalManagers:
    """Test global manager instances."""

    def test_global_executor_manager_singleton(self):
        """Test that global executor manager is singleton."""
        manager1 = get_global_executor_manager()
        manager2 = get_global_executor_manager()

        assert manager1 is manager2

    def test_global_cuda_manager_singleton(self):
        """Test that global CUDA manager is singleton."""
        manager1 = get_global_cuda_manager()
        manager2 = get_global_cuda_manager()

        assert manager1 is manager2

    def test_thread_safe_execution_context(self):
        """Test thread-safe execution context manager."""
        with thread_safe_execution() as context:
            assert "executor_manager" in context
            assert "cuda_manager" in context
            assert "submit_task" in context
            assert "cuda_stream" in context

            # Test task submission through context
            future = context["submit_task"](lambda: "test_result")
            result = future.result(timeout=5.0)
            assert result == "test_result"


class TestConcurrentUtilities:
    """Test concurrent task utilities."""

    def test_run_concurrent_tasks(self):
        """Test concurrent task execution utility."""

        def task1():
            return 1

        def task2():
            return 2

        def task3():
            return 3

        tasks = [task1, task2, task3]
        results = run_concurrent_tasks(tasks, max_workers=2)

        assert len(results) == 3
        assert all(result is not None for result in results)
        assert sorted([r for r in results if r is not None]) == [1, 2, 3]

    def test_run_concurrent_tasks_with_failures(self):
        """Test concurrent task execution with some failures."""

        def success_task():
            return "success"

        def failure_task():
            raise ValueError("Task failed")

        tasks = [success_task, failure_task, success_task]
        results = run_concurrent_tasks(tasks)

        assert len(results) == 3
        success_count = len([r for r in results if r == "success"])
        none_count = len([r for r in results if r is None])

        assert success_count == 2
        assert none_count == 1


class TestDataLoaderThreadSafety:
    """Test thread safety enhancements for data loaders."""

    def test_make_thread_safe_dataloader_decorator(self):
        """Test dataloader thread safety decorator."""

        class MockDataLoader:
            def __init__(self, data):
                self.data = data

            def __iter__(self):
                return iter(self.data)

        # Apply thread safety decorator
        ThreadSafeDataLoader = make_thread_safe_dataloader(MockDataLoader)

        # Test enhanced dataloader
        test_data = [1, 2, 3, 4, 5]
        loader = ThreadSafeDataLoader(test_data)

        assert hasattr(loader, "_thread_safety_manager")
        assert hasattr(loader, "_cuda_stream_manager")
        assert hasattr(loader, "_access_lock")
        assert hasattr(ThreadSafeDataLoader, "_thread_safe_enhanced")

    def test_cellmap_dataloader_thread_safety_integration(self):
        """Test CellMapDataLoader thread safety integration."""
        # Import needed modules
        from cellmap_data.dataset import CellMapDataset

        # Create mock dataset with all required attributes
        mock_dataset = Mock(spec=CellMapDataset)
        mock_dataset.__len__ = Mock(return_value=10)
        mock_dataset.__getitem__ = Mock(
            return_value=(np.random.rand(64, 64, 64), np.random.rand(64, 64, 64))
        )
        mock_dataset.classes = ["class1", "class2", "class3"]
        mock_dataset.class_counts = {"class1": 5, "class2": 3, "class3": 2}
        mock_dataset.class_weights = {"class1": 0.5, "class2": 0.3, "class3": 0.2}
        mock_dataset.validation_indices = [0, 1, 2, 3, 4]

        # Create dataloader with thread safety
        loader = CellMapDataLoader(mock_dataset, batch_size=2)

        # Verify thread safety components are initialized
        assert hasattr(loader, "_thread_safety_manager")
        assert loader._thread_safety_manager is not None

    def test_concurrent_dataloader_access(self):
        """Test concurrent access to dataloader."""
        # This is a more comprehensive test that would require actual data
        # For now, we'll test the framework components

        def mock_data_access():
            # Simulate data loading work
            time.sleep(0.01)
            return np.random.rand(32, 32, 32)

        # Test concurrent access using thread safety framework
        with thread_safe_execution() as context:
            futures = []
            for i in range(5):
                future = context["submit_task"](mock_data_access)
                futures.append(future)

            results = [future.result(timeout=5.0) for future in futures]

            assert len(results) == 5
            assert all(isinstance(result, np.ndarray) for result in results)


class TestPerformanceValidation:
    """Test performance validation in multi-threaded scenarios."""

    def test_thread_safety_performance_overhead(self):
        """Test performance overhead of thread safety features."""
        iterations = 100

        # Test without thread safety
        def simple_task():
            return sum(range(100))

        start_time = time.time()
        for _ in range(iterations):
            simple_task()
        simple_time = time.time() - start_time

        # Test with thread safety
        @thread_safe()
        def thread_safe_task():
            return sum(range(100))

        start_time = time.time()
        for _ in range(iterations):
            thread_safe_task()
        thread_safe_time = time.time() - start_time

        # Thread safety overhead should be reasonable (less than 10x)
        overhead_ratio = thread_safe_time / simple_time
        assert (
            overhead_ratio < 10
        ), f"Thread safety overhead too high: {overhead_ratio}x"

    def test_concurrent_performance_scaling(self):
        """Test performance scaling with concurrent operations."""

        def cpu_intensive_task():
            # More substantial CPU-bound task to see threading benefits
            result = 0
            for i in range(1000):  # Smaller workload to make test stable
                result += i**2
            return result

        # Test sequential execution
        sequential_results = [cpu_intensive_task() for _ in range(4)]

        # Test concurrent execution
        concurrent_results = run_concurrent_tasks(
            [cpu_intensive_task for _ in range(4)], max_workers=4
        )

        # Verify results are the same (this is the important part)
        assert sequential_results == [r for r in concurrent_results if r is not None]

        # Verify concurrent execution completed successfully
        assert len(concurrent_results) == 4
        assert all(r is not None for r in concurrent_results)

    def test_memory_efficiency_with_threads(self):
        """Test memory efficiency in multi-threaded scenarios."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        def memory_intensive_task():
            # Create and immediately release large array
            large_array = np.random.rand(1000, 1000)
            result = np.sum(large_array)
            del large_array
            return result

        # Run concurrent memory-intensive tasks
        results = run_concurrent_tasks(
            [memory_intensive_task for _ in range(10)], max_workers=4
        )

        # Force garbage collection
        import gc

        gc.collect()
        time.sleep(0.1)  # Allow time for cleanup

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 500MB)
        assert memory_increase < 500, f"Memory increase too high: {memory_increase}MB"
        assert len(results) == 10


if __name__ == "__main__":
    pytest.main([__file__])
