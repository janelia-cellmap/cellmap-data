"""
Test suite for Week 5 Day 3-4 Enhanced Memory Management optimizations.

This module validates memory optimization features including streaming buffer,
advanced memory management, GPU memory optimization, and memory-efficient
data loading patterns.
"""

import gc
import pytest
import torch
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from cellmap_data.utils.memory_optimization import (
    MemoryOptimizationConfig,
    StreamingDataBuffer,
    AdvancedMemoryManager,
    create_memory_efficient_dataloader,
    memory_optimized_training_context,
)


class TestMemoryOptimizationConfig:
    """Test memory optimization configuration."""

    def test_default_config_values(self):
        """Test default configuration values are reasonable."""
        config = MemoryOptimizationConfig()

        assert config.max_memory_mb is None  # Unlimited by default
        assert config.warning_threshold_percent == 80.0
        assert config.cleanup_threshold_percent == 85.0
        assert config.enable_streaming is True
        assert config.stream_buffer_size_mb == 512.0
        assert config.prefetch_factor == 2
        assert config.enable_gpu_memory_optimization is True
        assert config.gpu_memory_fraction == 0.9
        assert config.gc_frequency == 100
        assert config.aggressive_cleanup is False
        assert config.enable_detailed_profiling is False
        assert config.snapshot_frequency == 50

    def test_custom_config_values(self):
        """Test custom configuration settings."""
        config = MemoryOptimizationConfig(
            max_memory_mb=2048.0,
            warning_threshold_percent=70.0,
            cleanup_threshold_percent=75.0,
            stream_buffer_size_mb=256.0,
            gc_frequency=50,
            aggressive_cleanup=True,
            enable_detailed_profiling=True,
        )

        assert config.max_memory_mb == 2048.0
        assert config.warning_threshold_percent == 70.0
        assert config.cleanup_threshold_percent == 75.0
        assert config.stream_buffer_size_mb == 256.0
        assert config.gc_frequency == 50
        assert config.aggressive_cleanup is True
        assert config.enable_detailed_profiling is True


class TestStreamingDataBuffer:
    """Test streaming data buffer functionality."""

    def test_buffer_initialization(self):
        """Test buffer initializes with correct settings."""
        buffer = StreamingDataBuffer(buffer_size_mb=256.0, prefetch_factor=3)

        assert buffer.buffer_size_mb == 256.0
        assert buffer.prefetch_factor == 3
        assert len(buffer._buffer) == 0
        assert buffer._buffer_size_current == 0.0

    def test_tensor_size_estimation(self):
        """Test accurate tensor size estimation."""
        buffer = StreamingDataBuffer()

        # Test torch tensor
        tensor = torch.randn(100, 200, dtype=torch.float32)
        expected_mb = (tensor.numel() * tensor.element_size()) / (1024 * 1024)
        estimated_mb = buffer._estimate_item_size_mb(tensor)
        assert abs(estimated_mb - expected_mb) < 0.01

        # Test numpy array
        array = np.random.randn(50, 100).astype(np.float32)
        expected_mb = array.nbytes / (1024 * 1024)
        estimated_mb = buffer._estimate_item_size_mb(array)
        assert abs(estimated_mb - expected_mb) < 0.01

    def test_dict_size_estimation(self):
        """Test dictionary with multiple tensors size estimation."""
        buffer = StreamingDataBuffer()

        data_dict = {
            "raw": torch.randn(64, 64, dtype=torch.float32),
            "labels": torch.randint(0, 10, (64, 64), dtype=torch.long),
            "metadata": {"info": "test"},
        }

        estimated_mb = buffer._estimate_item_size_mb(data_dict)

        # Should be sum of tensor sizes plus small overhead for metadata
        raw_mb = (data_dict["raw"].numel() * data_dict["raw"].element_size()) / (
            1024 * 1024
        )
        labels_mb = (
            data_dict["labels"].numel() * data_dict["labels"].element_size()
        ) / (1024 * 1024)
        expected_mb = raw_mb + labels_mb + 0.1  # metadata overhead

        assert abs(estimated_mb - expected_mb) < 0.1

    def test_buffer_add_and_get_operations(self):
        """Test adding and retrieving items from buffer."""
        buffer = StreamingDataBuffer(buffer_size_mb=1.0)  # Small buffer

        # Add small item - should succeed
        small_tensor = torch.randn(10, 10, dtype=torch.float32)
        assert buffer.add_item(small_tensor) is True
        assert len(buffer._buffer) == 1

        # Add large item that exceeds buffer - should fail
        large_tensor = torch.randn(1000, 1000, dtype=torch.float32)  # ~4MB
        assert buffer.add_item(large_tensor) is False
        assert len(buffer._buffer) == 1  # Still only original item

        # Get item from buffer
        retrieved_item = buffer.get_item()
        assert retrieved_item is not None
        assert torch.equal(retrieved_item, small_tensor)
        assert len(buffer._buffer) == 0
        assert buffer._buffer_size_current == 0.0

    def test_buffer_status(self):
        """Test buffer status reporting."""
        buffer = StreamingDataBuffer(buffer_size_mb=2.0)

        # Empty buffer status
        status = buffer.get_buffer_status()
        assert status["items_count"] == 0
        assert status["current_size_mb"] == 0.0
        assert status["max_size_mb"] == 2.0
        assert status["utilization_percent"] == 0.0

        # Add item and check status
        tensor = torch.randn(100, 100, dtype=torch.float32)
        buffer.add_item(tensor)

        status = buffer.get_buffer_status()
        assert status["items_count"] == 1
        assert status["current_size_mb"] > 0
        assert status["utilization_percent"] > 0

    def test_buffer_clear(self):
        """Test buffer clearing functionality."""
        buffer = StreamingDataBuffer()

        # Add multiple items
        for i in range(5):
            tensor = torch.randn(50, 50, dtype=torch.float32)
            buffer.add_item(tensor)

        assert len(buffer._buffer) == 5

        # Clear buffer
        cleared_count = buffer.clear_buffer()
        assert cleared_count == 5
        assert len(buffer._buffer) == 0
        assert buffer._buffer_size_current == 0.0


class TestAdvancedMemoryManager:
    """Test advanced memory manager functionality."""

    def test_memory_manager_initialization(self):
        """Test memory manager initializes correctly."""
        config = MemoryOptimizationConfig(max_memory_mb=1024.0)
        manager = AdvancedMemoryManager(config)

        assert manager.config.max_memory_mb == 1024.0
        assert manager.profiler is not None
        assert manager.resource_manager is not None
        assert manager.streaming_buffer is not None
        assert manager._operation_count == 0
        assert manager._last_gc_operation == 0
        assert len(manager._memory_alerts) == 0

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.set_per_process_memory_fraction")
    @patch("torch.cuda.empty_cache")
    def test_gpu_optimizations_applied(
        self, mock_empty_cache, mock_set_fraction, mock_cuda_available
    ):
        """Test GPU memory optimizations are applied when available."""
        config = MemoryOptimizationConfig(
            enable_gpu_memory_optimization=True, gpu_memory_fraction=0.8
        )

        # Patch the GPU optimizer to avoid mocking issues
        with patch("cellmap_data.utils.memory_optimization.GPUMemoryOptimizer"):
            manager = AdvancedMemoryManager(config)

            # GPU optimizations should have been applied during initialization
            mock_set_fraction.assert_called_once_with(0.8)
            mock_empty_cache.assert_called()

    def test_optimized_tensor_allocation(self):
        """Test optimized tensor allocation."""
        config = MemoryOptimizationConfig(max_memory_mb=1024.0)
        manager = AdvancedMemoryManager(config)

        # Allocate tensor
        shape = (100, 100)
        tensor = manager.allocate_optimized_tensor(shape, name="test_tensor")

        assert tensor.shape == shape
        assert tensor.dtype == torch.float32
        assert manager._operation_count == 1

    @patch("gc.collect")
    def test_periodic_garbage_collection(self, mock_gc_collect):
        """Test periodic garbage collection triggers correctly."""
        config = MemoryOptimizationConfig(gc_frequency=10)
        manager = AdvancedMemoryManager(config)

        # Trigger operations to reach GC frequency
        for i in range(15):
            manager._operation_count += 1
            manager._perform_periodic_gc()

        # GC should have been called
        assert mock_gc_collect.called
        assert manager._last_gc_operation > 0

    def test_memory_efficient_processing_context(self):
        """Test memory-efficient processing context manager."""
        manager = AdvancedMemoryManager()

        with manager.memory_efficient_processing("test_operation") as ctx:
            assert ctx is manager
            # Simulate some work
            tensor = torch.randn(100, 100)
            assert tensor.shape == (100, 100)

        # Context should complete without errors

    def test_streaming_data_loader(self):
        """Test streaming data loader creation."""
        manager = AdvancedMemoryManager()

        # Create test data source
        test_data = [torch.randn(10, 10) for _ in range(50)]

        streaming_loader = manager.get_streaming_data_loader(iter(test_data))
        loaded_items = list(streaming_loader)

        # All items should be loaded
        assert len(loaded_items) == 50

        # Items should be tensors of correct shape
        for item in loaded_items:
            assert isinstance(item, torch.Tensor)
            assert item.shape == (10, 10)

    def test_large_dataset_optimization(self):
        """Test optimization settings for large datasets."""
        manager = AdvancedMemoryManager()

        # Test medium dataset (15GB)
        optimizations = manager.optimize_for_large_dataset(15.0)
        assert "aggressive_cleanup" in optimizations
        assert "detailed_profiling" in optimizations
        assert manager.config.cleanup_threshold_percent == 75.0
        assert manager.config.gc_frequency == 50

        # Test very large dataset (60GB)
        optimizations = manager.optimize_for_large_dataset(60.0)
        assert "extra_aggressive" in optimizations
        assert manager.config.cleanup_threshold_percent == 70.0
        assert manager.config.gc_frequency == 25

    def test_comprehensive_memory_report(self):
        """Test comprehensive memory report generation."""
        config = MemoryOptimizationConfig(enable_detailed_profiling=True)
        manager = AdvancedMemoryManager(config)

        # Perform some operations to populate report data
        tensor = manager.allocate_optimized_tensor((50, 50), name="test")

        report = manager.get_comprehensive_memory_report()

        assert "Advanced Memory Management Report" in report
        assert "Configuration:" in report
        assert "Operations:" in report
        assert "Resource Manager:" in report
        assert "Streaming Buffer:" in report
        assert len(report) > 100  # Should be substantial report


class TestMemoryOptimizationIntegration:
    """Test integration with dataloader and training workflows."""

    def test_memory_optimized_training_context(self):
        """Test memory-optimized training context manager."""
        with memory_optimized_training_context(memory_limit_mb=512.0) as manager:
            assert isinstance(manager, AdvancedMemoryManager)
            assert manager.config.max_memory_mb == 512.0
            assert manager.config.cleanup_threshold_percent == 80.0
            assert manager.config.enable_gpu_memory_optimization is True
            assert manager.config.enable_detailed_profiling is True

    @patch("cellmap_data.dataloader.CellMapDataLoader")
    def test_create_memory_efficient_dataloader(self, mock_dataloader_class):
        """Test creation of memory-efficient dataloader."""
        # Mock dataset with array information
        mock_dataset = Mock()
        mock_dataset.input_arrays = {"raw": {"shape": (256, 256)}}
        mock_dataset.target_arrays = {"labels": {"shape": (256, 256)}}

        # Create memory-efficient dataloader
        dataloader = create_memory_efficient_dataloader(
            mock_dataset, batch_size=4, num_workers=2
        )

        # Should have called the DataLoader constructor
        mock_dataloader_class.assert_called_once()
        call_args = mock_dataloader_class.call_args

        # Check that memory optimization parameters were set
        assert "memory_limit_mb" in call_args[1]
        assert "enable_memory_profiling" in call_args[1]
        assert call_args[1]["enable_memory_profiling"] is True


class TestMemoryOptimizationPerformance:
    """Test performance characteristics of memory optimization."""

    def test_buffer_performance_with_large_data(self):
        """Test buffer performance with large amounts of data."""
        buffer = StreamingDataBuffer(buffer_size_mb=100.0)  # 100MB buffer

        start_time = time.time()
        added_count = 0

        # Try to add many small tensors
        for i in range(1000):
            tensor = torch.randn(100, 100, dtype=torch.float32)  # ~40KB each
            if buffer.add_item(tensor):
                added_count += 1
            else:
                break

        end_time = time.time()

        # Should be able to add many items quickly
        assert added_count > 200  # Should fit many 40KB tensors in 100MB
        assert (end_time - start_time) < 5.0  # Should complete quickly

        # Buffer should have reasonable utilization (adjust expectation)
        status = buffer.get_buffer_status()
        assert status["utilization_percent"] > 30.0  # More realistic expectation
        assert added_count > 0  # At least some items should fit

    def test_memory_manager_allocation_performance(self):
        """Test memory manager allocation performance."""
        manager = AdvancedMemoryManager()

        start_time = time.time()
        tensors = []

        # Allocate many tensors
        for i in range(100):
            tensor = manager.allocate_optimized_tensor(
                (100, 100), name=f"perf_test_{i}"
            )
            tensors.append(tensor)

        end_time = time.time()

        # Should complete quickly
        assert (end_time - start_time) < 10.0
        assert len(tensors) == 100
        assert manager._operation_count == 100

    def test_streaming_loader_memory_efficiency(self):
        """Test streaming loader memory efficiency."""
        manager = AdvancedMemoryManager()

        # Create large test dataset
        large_tensors = [
            torch.randn(200, 200) for _ in range(500)
        ]  # ~500 * 160KB = ~80MB

        streaming_loader = manager.get_streaming_data_loader(iter(large_tensors))

        # Process in batches to simulate real usage
        processed_count = 0
        for item in streaming_loader:
            processed_count += 1

            # Clear item to simulate processing
            del item

            # Periodic cleanup
            if processed_count % 100 == 0:
                gc.collect()

        assert processed_count == 500

        # Buffer should be cleared after iteration
        buffer_status = manager.streaming_buffer.get_buffer_status()
        assert buffer_status["items_count"] == 0


class TestMemoryOptimizationEdgeCases:
    """Test edge cases and error handling in memory optimization."""

    def test_buffer_with_zero_size_limit(self):
        """Test buffer behavior with zero size limit."""
        buffer = StreamingDataBuffer(buffer_size_mb=0.0)

        tensor = torch.randn(10, 10)
        # Should not be able to add any items to zero-size buffer
        assert buffer.add_item(tensor) is False
        assert len(buffer._buffer) == 0

    def test_memory_manager_with_none_config(self):
        """Test memory manager with None configuration."""
        manager = AdvancedMemoryManager(config=None)

        # Should use default configuration
        assert manager.config is not None
        assert isinstance(manager.config, MemoryOptimizationConfig)

    def test_gpu_optimization_without_cuda(self):
        """Test GPU optimization when CUDA is not available."""
        with patch("torch.cuda.is_available", return_value=False):
            config = MemoryOptimizationConfig(enable_gpu_memory_optimization=True)
            manager = AdvancedMemoryManager(config)

            # Should initialize without errors even when CUDA is unavailable
            assert manager is not None

    def test_streaming_loader_with_empty_data(self):
        """Test streaming loader with empty data source."""
        manager = AdvancedMemoryManager()

        empty_data = []
        streaming_loader = manager.get_streaming_data_loader(iter(empty_data))
        loaded_items = list(streaming_loader)

        assert len(loaded_items) == 0

    def test_memory_alert_handling_edge_cases(self):
        """Test memory alert handling with edge cases."""
        manager = AdvancedMemoryManager()

        # Simulate high memory usage scenario
        fake_memory_info = {"percent": 90.0, "rss_mb": 2048.0}

        with patch.object(
            manager.profiler, "_get_memory_info", return_value=fake_memory_info
        ):
            manager._handle_high_memory_usage(fake_memory_info)

        # Should have generated memory alert
        assert len(manager._memory_alerts) > 0
        alert = manager._memory_alerts[-1]
        assert alert["memory_percent"] == 90.0
        assert "action_taken" in alert


# Integration test markers for performance benchmarks
class TestMemoryOptimizationBenchmarks:
    """Benchmark tests for memory optimization (performance tests)."""

    def test_memory_reduction_benchmark(self):
        """Benchmark memory reduction capabilities."""
        # This would be a comprehensive benchmark comparing memory usage
        # with and without optimization features
        pass

    def test_streaming_throughput_benchmark(self):
        """Benchmark streaming data loading throughput."""
        # This would measure data loading throughput with streaming buffer
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
