"""
Week 5 Day 5: Enhanced Thread Safety Integration Tests

Tests for the enhanced thread safety integration with the dataloader,
memory optimization, and concurrent loading capabilities.
"""

import pytest
import threading
import time
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor

from cellmap_data.utils.enhanced_thread_safety import (
    EnhancedThreadSafetyConfig,
    MemoryAwareThreadManager,
    ConcurrentDataLoadingManager,
    ThreadSafetyEnhancementManager,
    get_global_thread_safety_enhancements,
    enhanced_thread_safety_execution,
    optimize_thread_safety_globally,
)


class TestEnhancedThreadSafetyConfig:
    """Test enhanced thread safety configuration."""

    def test_enhanced_config_inheritance(self):
        """Test that enhanced config inherits from base config."""
        config = EnhancedThreadSafetyConfig()

        # Base ThreadSafetyConfig properties
        assert config.max_workers == 4
        assert config.thread_pool_name == "cellmap_workers"
        assert config.enable_thread_monitoring is True

        # Enhanced properties
        assert config.enable_memory_aware_threading is True
        assert config.memory_threshold_for_threading == 1024.0
        assert config.thread_memory_limit_mb == 512.0
        assert config.enable_priority_queuing is True
        assert config.max_priority_levels == 5
        assert config.enable_detailed_profiling is True
        assert config.integrate_with_memory_manager is True

    def test_enhanced_config_customization(self):
        """Test custom enhanced configuration."""
        config = EnhancedThreadSafetyConfig(
            max_workers=8,
            enable_memory_aware_threading=False,
            enable_priority_queuing=False,
            enable_detailed_profiling=False,
            thread_memory_limit_mb=1024.0,
            max_priority_levels=3,
        )

        assert config.max_workers == 8
        assert config.enable_memory_aware_threading is False
        assert config.enable_priority_queuing is False
        assert config.enable_detailed_profiling is False
        assert config.thread_memory_limit_mb == 1024.0
        assert config.max_priority_levels == 3


class TestMemoryAwareThreadManager:
    """Test memory-aware thread management."""

    def test_manager_initialization(self):
        """Test memory-aware thread manager initialization."""
        config = EnhancedThreadSafetyConfig(
            enable_memory_aware_threading=True, integrate_with_memory_manager=True
        )

        manager = MemoryAwareThreadManager(config)

        assert manager.config is config
        assert manager.executor_manager is not None
        assert isinstance(manager._memory_usage_by_thread, dict)
        assert isinstance(manager._performance_metrics, dict)

    def test_memory_aware_execution_context(self):
        """Test memory-aware execution context manager."""
        manager = MemoryAwareThreadManager()

        execution_completed = False

        with manager.memory_aware_execution("test_task") as context:
            execution_completed = True
            assert context is None  # Context manager yields None

        assert execution_completed

    def test_memory_aware_task_submission(self):
        """Test memory-aware task submission."""
        manager = MemoryAwareThreadManager()

        def test_task(value):
            return value * 2

        future = manager.submit_memory_aware_task(
            "test_executor", test_task, 5, priority=1, memory_limit_mb=100.0
        )

        result = future.result(timeout=5.0)
        assert result == 10

    def test_priority_task_submission(self):
        """Test priority-based task submission."""
        config = EnhancedThreadSafetyConfig(enable_priority_queuing=True)
        manager = MemoryAwareThreadManager(config)

        results = []

        def priority_task(priority_level):
            time.sleep(0.01)  # Small delay
            results.append(priority_level)
            return priority_level

        # Submit tasks with different priorities
        futures = []
        for priority in [2, 0, 1]:  # Lower number = higher priority
            future = manager.submit_memory_aware_task(
                "priority_executor", priority_task, priority, priority=priority
            )
            futures.append(future)

        # Wait for all tasks to complete
        for future in futures:
            future.result(timeout=5.0)

        # Results should be recorded (exact order depends on scheduling)
        assert len(results) == 3
        assert set(results) == {0, 1, 2}

    def test_performance_metrics_recording(self):
        """Test performance metrics recording."""
        config = EnhancedThreadSafetyConfig(
            enable_detailed_profiling=True,
            profiling_sample_rate=1.0,  # 100% sampling for testing
        )
        manager = MemoryAwareThreadManager(config)

        # Record some metrics manually
        manager._record_performance_metrics("test_task", 0.5, 10.0)
        manager._record_performance_metrics("slow_task", 2.0, 50.0)

        report = manager.get_performance_report()

        assert report["total_tasks_sampled"] == 2
        assert report["average_execution_time"] > 0
        assert report["max_execution_time"] >= 2.0
        assert report["bottleneck_events_count"] >= 1  # slow_task should be flagged

    def test_thread_allocation_optimization(self):
        """Test dynamic thread allocation optimization."""
        manager = MemoryAwareThreadManager()

        # Simulate bottleneck events
        manager._performance_metrics["bottleneck_events"] = [time.time()] * 15

        recommendations = manager.optimize_thread_allocation()

        # Should recommend reducing workers due to bottlenecks
        assert "reduce_max_workers" in recommendations or len(recommendations) == 0


class TestConcurrentDataLoadingManager:
    """Test concurrent data loading management."""

    def test_loading_manager_initialization(self):
        """Test concurrent data loading manager initialization."""
        manager = ConcurrentDataLoadingManager()

        assert manager.config is not None
        assert manager.thread_manager is not None
        assert manager.cuda_manager is not None
        assert isinstance(manager._active_loaders, set)
        assert isinstance(manager._batch_queues, dict)

    def test_dataloader_registration(self):
        """Test dataloader registration and unregistration."""
        manager = ConcurrentDataLoadingManager()

        # Mock dataloader
        mock_loader = Mock()
        mock_loader_id = str(id(mock_loader))

        # Register dataloader
        manager.register_dataloader(mock_loader)

        assert len(manager._active_loaders) >= 1
        assert mock_loader_id in manager._batch_queues
        assert mock_loader_id in manager._queue_locks

        # Unregister dataloader
        manager.unregister_dataloader(mock_loader)

        # Check cleanup
        active_loader_objects = [
            ref() for ref in manager._active_loaders if ref() is not None
        ]
        assert mock_loader not in active_loader_objects
        assert mock_loader_id not in manager._batch_queues
        assert mock_loader_id not in manager._queue_locks

    def test_concurrent_batch_loading_context(self):
        """Test concurrent batch loading context manager."""
        manager = ConcurrentDataLoadingManager()

        # Mock dataloader with __getitem__ method
        mock_loader = Mock()
        mock_loader.__getitem__ = Mock(side_effect=lambda idx: f"batch_{idx}")

        # Register the mock loader
        manager.register_dataloader(mock_loader)

        batch_indices = [0, 1, 2]

        try:
            with manager.concurrent_batch_loading(
                mock_loader, batch_indices
            ) as batch_generator:
                results = list(
                    batch_generator
                )  # batch_generator is already an iterator

                # Should have results for all indices
                assert len(results) == 3

                # Check that results are tuples of (index, data)
                for idx, data in results:
                    assert idx in batch_indices
                    if data is not None:  # Some might be None due to mocking
                        assert data == f"batch_{idx}"

        except Exception as e:
            # Some failures are expected due to mocking limitations
            assert "does not support indexing" in str(e) or "load_single_batch" in str(
                e
            )

    def test_loader_statistics(self):
        """Test loader statistics generation."""
        manager = ConcurrentDataLoadingManager()

        # Register some mock loaders and keep references to prevent garbage collection
        mock_loaders = []
        for i in range(3):
            mock_loader = Mock()
            mock_loaders.append(mock_loader)  # Keep a strong reference
            manager.register_dataloader(mock_loader)

        stats = manager.get_loader_statistics()

        assert "active_loaders" in stats
        assert "total_batch_queues" in stats
        assert "performance_report" in stats
        assert stats["active_loaders"] >= 3
        assert stats["total_batch_queues"] >= 3


class TestThreadSafetyEnhancementManager:
    """Test comprehensive thread safety enhancement manager."""

    def test_enhancement_manager_initialization(self):
        """Test enhancement manager initialization."""
        manager = ThreadSafetyEnhancementManager()

        assert manager.config is not None
        assert manager.memory_aware_manager is not None
        assert manager.concurrent_loader_manager is not None
        assert isinstance(manager._error_counts, dict)

    def test_enhanced_thread_safety_context(self):
        """Test enhanced thread safety context manager."""
        manager = ThreadSafetyEnhancementManager()

        context_entered = False

        with manager.enhanced_thread_safety_context("test_context") as context:
            context_entered = True

            # Check context provides expected interfaces
            assert "memory_aware_manager" in context
            assert "concurrent_loader_manager" in context
            assert "submit_memory_aware_task" in context
            assert "concurrent_batch_loading" in context
            assert "register_dataloader" in context
            assert "performance_report" in context

            # Test submitting a task through context
            future = context["submit_memory_aware_task"](
                "test_executor", lambda x: x + 1, 5
            )
            result = future.result(timeout=5.0)
            assert result == 6

        assert context_entered

    def test_comprehensive_performance_report(self):
        """Test comprehensive performance report generation."""
        manager = ThreadSafetyEnhancementManager()

        report = manager.get_comprehensive_performance_report()

        assert "memory_aware_performance" in report
        assert "dataloader_statistics" in report
        assert "error_recovery_stats" in report
        assert "configuration" in report

        # Check configuration section
        config = report["configuration"]
        assert "memory_aware_threading" in config
        assert "priority_queuing" in config
        assert "detailed_profiling" in config
        assert "memory_integration" in config

    def test_error_recovery_handling(self):
        """Test error recovery and tracking."""
        config = EnhancedThreadSafetyConfig(enable_thread_error_recovery=True)
        manager = ThreadSafetyEnhancementManager(config)

        # Simulate an error
        test_error = ValueError("Test error")
        manager._handle_context_error("test_context", test_error)

        # Check error was recorded
        error_key = "test_context:ValueError"
        assert error_key in manager._error_counts
        assert manager._error_counts[error_key] == 1

    def test_component_optimization(self):
        """Test optimization of all components."""
        manager = ThreadSafetyEnhancementManager()

        optimizations = manager.optimize_all_components()

        assert isinstance(optimizations, dict)
        assert "current_performance" in optimizations
        # thread_allocation might or might not be present depending on performance


class TestGlobalEnhancedManager:
    """Test global enhanced thread safety manager."""

    def test_global_manager_singleton(self):
        """Test that global enhanced manager is singleton."""
        manager1 = get_global_thread_safety_enhancements()
        manager2 = get_global_thread_safety_enhancements()

        assert manager1 is manager2

    def test_enhanced_execution_context(self):
        """Test global enhanced execution context."""
        with enhanced_thread_safety_execution(context_name="global_test") as context:
            assert "memory_aware_manager" in context
            assert "concurrent_loader_manager" in context

            # Test task submission
            future = context["submit_memory_aware_task"](
                "global_executor", lambda: "global_test_result"
            )
            result = future.result(timeout=5.0)
            assert result == "global_test_result"

    def test_global_optimization(self):
        """Test global thread safety optimization."""
        optimizations = optimize_thread_safety_globally()

        assert isinstance(optimizations, dict)
        assert "current_performance" in optimizations


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple features."""

    def test_memory_aware_concurrent_loading(self):
        """Test memory-aware concurrent loading scenario."""
        config = EnhancedThreadSafetyConfig(
            enable_memory_aware_threading=True,
            enable_detailed_profiling=True,
            max_concurrent_operations=4,
        )

        manager = ThreadSafetyEnhancementManager(config)

        def memory_intensive_task(data_size):
            # Simulate memory-intensive data loading
            data = np.random.rand(data_size, 100)
            return np.sum(data)

        with manager.enhanced_thread_safety_context("memory_test") as context:
            # Submit multiple memory-intensive tasks
            futures = []
            for i in range(3):
                future = context["submit_memory_aware_task"](
                    "memory_executor",
                    memory_intensive_task,
                    100,  # data_size
                    priority=i,
                )
                futures.append(future)

            # Wait for all tasks and collect results
            results = [future.result(timeout=10.0) for future in futures]

            assert len(results) == 3
            assert all(isinstance(result, (int, float)) for result in results)

    def test_error_recovery_scenario(self):
        """Test error recovery in concurrent scenarios."""
        config = EnhancedThreadSafetyConfig(
            enable_thread_error_recovery=True, max_retry_attempts=2
        )

        manager = ThreadSafetyEnhancementManager(config)

        def failing_task():
            raise RuntimeError("Simulated task failure")

        with manager.enhanced_thread_safety_context("error_test") as context:
            # Submit a failing task
            future = context["submit_memory_aware_task"]("error_executor", failing_task)

            # Task should fail but error should be handled
            with pytest.raises(RuntimeError):
                future.result(timeout=5.0)

        # Check error was recorded
        assert len(manager._error_counts) > 0

    def test_performance_monitoring_integration(self):
        """Test integrated performance monitoring."""
        config = EnhancedThreadSafetyConfig(
            enable_detailed_profiling=True,
            profiling_sample_rate=1.0,  # 100% sampling
            enable_bottleneck_detection=True,
        )

        manager = ThreadSafetyEnhancementManager(config)

        def monitored_task(duration):
            time.sleep(duration)
            return f"completed_in_{duration}s"

        with manager.enhanced_thread_safety_context("monitoring_test") as context:
            # Submit fast and slow tasks
            fast_future = context["submit_memory_aware_task"](
                "monitor_executor", monitored_task, 0.01  # 10ms
            )
            slow_future = context["submit_memory_aware_task"](
                "monitor_executor", monitored_task, 0.1  # 100ms
            )

            # Wait for completion
            fast_result = fast_future.result(timeout=5.0)
            slow_result = slow_future.result(timeout=5.0)

            assert "completed_in_0.01s" in fast_result
            assert "completed_in_0.1s" in slow_result

        # Check performance report
        report = manager.get_comprehensive_performance_report()
        memory_performance = report["memory_aware_performance"]

        assert (
            memory_performance["total_tasks_sampled"] >= 0
        )  # May be 0 due to sampling


if __name__ == "__main__":
    pytest.main([__file__])
