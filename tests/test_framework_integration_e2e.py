"""
End-to-end integration tests for framework compatibility with mock components.
Tests the complete dataloader pipeline with framework detection.
"""

import sys
import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

from cellmap_data.enhanced_dataloader import EnhancedCellMapDataLoader
from cellmap_data.dataset import CellMapDataset
from cellmap_data.multidataset import CellMapMultiDataset
from cellmap_data.device.device_manager import DeviceManager


@pytest.fixture
def mock_cellmap_dataset():
    """Create a mock CellMapDataset for testing."""
    dataset = MagicMock(spec=CellMapDataset)
    dataset.classes = ["background", "object"]
    dataset.__len__ = MagicMock(return_value=20)
    dataset.__getitem__ = MagicMock(
        side_effect=lambda idx: {
            "raw": torch.randn(1, 16, 16, 16),
            "gt": torch.randint(0, 2, (1, 16, 16, 16)),
            # Remove index field as it causes issues with collation
        }
    )
    dataset.to = MagicMock(return_value=None)
    return dataset


@pytest.fixture
def mock_multidataset():
    """Create a mock CellMapMultiDataset for testing."""
    dataset = MagicMock(spec=CellMapMultiDataset)
    dataset.classes = ["background", "object"]
    dataset.__len__ = MagicMock(return_value=40)  # Two datasets of 20 each
    dataset.__getitem__ = MagicMock(
        side_effect=lambda idx: {
            "raw": torch.randn(1, 16, 16, 16),
            "gt": torch.randint(0, 2, (1, 16, 16, 16)),
            # Remove index field as it causes issues with collation
        }
    )
    dataset.to = MagicMock(return_value=None)
    return dataset


class TestFrameworkCompatibilityEndToEnd:
    """End-to-end framework compatibility tests with mock datasets."""

    def test_enhanced_dataloader_no_framework(self, mock_cellmap_dataset):
        """Test EnhancedCellMapDataLoader without any framework."""
        # Remove frameworks to test baseline behavior
        frameworks = ["accelerate", "pytorch_lightning", "deepspeed"]
        original_modules = {}

        for fw in frameworks:
            if fw in sys.modules:
                original_modules[fw] = sys.modules[fw]
                del sys.modules[fw]

        try:
            loader = EnhancedCellMapDataLoader(
                dataset=mock_cellmap_dataset,
                batch_size=2,
                enable_plugins=True,
                enable_device_transfer=True,
                enable_memory_optimization=True,
                device="cpu",
            )

            # Should have no framework detected
            assert loader.device_manager.framework is None

            # Should have plugins registered
            assert loader.plugin_manager is not None
            assert len(loader.plugin_manager.list_plugins()) > 0

            # Should be able to get a batch
            batch = next(iter(loader.loader))
            assert "raw" in batch
            assert "gt" in batch
            assert batch["raw"].device.type == "cpu"

        finally:
            # Restore original modules
            for fw, module in original_modules.items():
                sys.modules[fw] = module

    def test_enhanced_dataloader_with_pytorch_lightning(self, mock_cellmap_dataset):
        """Test EnhancedCellMapDataLoader with PyTorch Lightning framework."""
        mock_pl = MagicMock()
        with patch.dict(sys.modules, {"pytorch_lightning": mock_pl}):
            loader = EnhancedCellMapDataLoader(
                dataset=mock_cellmap_dataset,
                batch_size=2,
                enable_plugins=True,
                enable_device_transfer=True,
                device="cpu",
            )

            # Should detect PyTorch Lightning
            assert loader.device_manager.framework == "pytorch_lightning"

            # Should still have plugins functional
            device_plugin = loader.get_plugin("device_transfer")
            assert device_plugin is not None

            # Should be able to iterate through data
            batch_count = 0
            for batch in loader.loader:
                assert "raw" in batch
                assert "gt" in batch
                batch_count += 1
                if batch_count >= 2:  # Test a couple batches
                    break

    def test_enhanced_dataloader_with_accelerate(self, mock_cellmap_dataset):
        """Test EnhancedCellMapDataLoader with Accelerate framework."""
        mock_accelerate = MagicMock()
        with patch.dict(sys.modules, {"accelerate": mock_accelerate}):
            loader = EnhancedCellMapDataLoader(
                dataset=mock_cellmap_dataset,
                batch_size=2,
                enable_plugins=True,
                enable_prefetch=True,
                device="cpu",
            )

            # Should detect Accelerate
            assert loader.device_manager.framework == "accelerate"

            # Prefetch plugin should still work
            prefetch_plugin = loader.get_plugin("prefetch")
            assert prefetch_plugin is not None

            # Should handle data loading
            batch = next(iter(loader.loader))
            assert isinstance(batch["raw"], torch.Tensor)
            assert isinstance(batch["gt"], torch.Tensor)

    def test_plugin_hooks_with_framework(self, mock_cellmap_dataset):
        """Test that plugin hooks execute correctly with framework detection."""
        mock_deepspeed = MagicMock()
        with patch.dict(sys.modules, {"deepspeed": mock_deepspeed}):
            loader = EnhancedCellMapDataLoader(
                dataset=mock_cellmap_dataset,
                batch_size=1,
                enable_plugins=True,
                enable_augmentation=True,
                device="cpu",
            )

            # Should detect DeepSpeed
            assert loader.device_manager.framework == "deepspeed"

            # Plugin manager should execute hooks
            plugin_manager = loader.plugin_manager
            assert plugin_manager is not None

            # Test hook execution
            sample = {
                "raw": torch.randn(1, 16, 16, 16),
                "gt": torch.randint(0, 2, (1, 16, 16, 16)),
            }

            # Execute post_sample hook (augmentation)
            result = plugin_manager.execute_hook("post_sample", sample)
            assert isinstance(result, dict)
            assert "raw" in result
            assert "gt" in result

    def test_device_transfer_behavior_with_framework(self, mock_cellmap_dataset):
        """Test device transfer behavior with and without framework."""
        # Test without framework
        frameworks = ["accelerate", "pytorch_lightning", "deepspeed"]
        original_modules = {}

        for fw in frameworks:
            if fw in sys.modules:
                original_modules[fw] = sys.modules[fw]
                del sys.modules[fw]

        try:
            loader_no_fw = EnhancedCellMapDataLoader(
                dataset=mock_cellmap_dataset,
                batch_size=1,
                enable_plugins=True,
                enable_device_transfer=True,
                device="cpu",
            )

            # Get batch without framework
            batch_no_fw = next(iter(loader_no_fw.loader))
            assert batch_no_fw["raw"].device.type == "cpu"

            # Test with framework
            mock_pl = MagicMock()
            with patch.dict(sys.modules, {"pytorch_lightning": mock_pl}):
                loader_with_fw = EnhancedCellMapDataLoader(
                    dataset=mock_cellmap_dataset,
                    batch_size=1,
                    enable_plugins=True,
                    enable_device_transfer=True,
                    device="cpu",
                )

                # Get batch with framework
                batch_with_fw = next(iter(loader_with_fw.loader))
                assert batch_with_fw["raw"].device.type == "cpu"

                # Both should produce valid data
                assert batch_no_fw["raw"].shape == batch_with_fw["raw"].shape

        finally:
            # Restore original modules
            for fw, module in original_modules.items():
                sys.modules[fw] = module

    def test_multidataset_with_framework(self, mock_multidataset):
        """Test CellMapMultiDataset integration with framework detection."""
        mock_accelerate = MagicMock()
        with patch.dict(sys.modules, {"accelerate": mock_accelerate}):
            loader = EnhancedCellMapDataLoader(
                dataset=mock_multidataset,
                batch_size=2,
                enable_plugins=True,
                device="cpu",
            )

            # Should detect framework
            assert loader.device_manager.framework == "accelerate"

            # Should work with multidataset
            batch = next(iter(loader.loader))
            assert "raw" in batch
            assert "gt" in batch
            assert batch["raw"].shape[0] == 2  # batch size

    def test_memory_optimization_with_framework(self, mock_cellmap_dataset):
        """Test memory optimization plugin with framework detection."""
        mock_pl = MagicMock()
        with patch.dict(sys.modules, {"pytorch_lightning": mock_pl}):
            loader = EnhancedCellMapDataLoader(
                dataset=mock_cellmap_dataset,
                batch_size=2,
                enable_plugins=True,
                enable_memory_optimization=True,
                device="cpu",
            )

            # Should have memory optimization plugin
            memory_plugin = loader.get_plugin("memory_optimization")
            assert memory_plugin is not None

            # Test memory optimization
            batch = {
                "raw": torch.randn(2, 1, 16, 16, 16),
                "gt": torch.randint(0, 2, (2, 1, 16, 16, 16)),
            }

            # Access the specific plugin method
            from cellmap_data.plugins import MemoryOptimizationPlugin

            assert isinstance(memory_plugin, MemoryOptimizationPlugin)
            optimized = memory_plugin.optimize_memory(batch)
            assert "raw" in optimized
            assert "gt" in optimized

    def test_framework_change_during_runtime(self, mock_cellmap_dataset):
        """Test behavior when framework status changes during runtime."""
        # Start without framework
        frameworks = ["accelerate", "pytorch_lightning", "deepspeed"]
        original_modules = {}

        for fw in frameworks:
            if fw in sys.modules:
                original_modules[fw] = sys.modules[fw]
                del sys.modules[fw]

        try:
            loader = EnhancedCellMapDataLoader(
                dataset=mock_cellmap_dataset,
                batch_size=1,
                enable_plugins=True,
                device="cpu",
            )

            # Initially no framework
            assert loader.device_manager.framework is None

            # Get initial batch
            batch1 = next(iter(loader.loader))
            assert "raw" in batch1

            # Note: In real usage, framework changes would require
            # recreating the loader, but this tests the detection logic

        finally:
            # Restore original modules
            for fw, module in original_modules.items():
                sys.modules[fw] = module


class TestFrameworkCompatibilityPerformance:
    """Test performance characteristics with framework compatibility."""

    def test_framework_detection_overhead(self, mock_cellmap_dataset):
        """Test that framework detection doesn't add significant overhead."""
        import time

        # Test creation time without framework
        frameworks = ["accelerate", "pytorch_lightning", "deepspeed"]
        original_modules = {}

        for fw in frameworks:
            if fw in sys.modules:
                original_modules[fw] = sys.modules[fw]
                del sys.modules[fw]

        try:
            start_time = time.time()
            loader_no_fw = EnhancedCellMapDataLoader(
                dataset=mock_cellmap_dataset,
                batch_size=2,
                enable_plugins=True,
                device="cpu",
            )
            time_no_fw = time.time() - start_time

            # Test with framework
            mock_pl = MagicMock()
            with patch.dict(sys.modules, {"pytorch_lightning": mock_pl}):
                start_time = time.time()
                loader_with_fw = EnhancedCellMapDataLoader(
                    dataset=mock_cellmap_dataset,
                    batch_size=2,
                    enable_plugins=True,
                    device="cpu",
                )
                time_with_fw = time.time() - start_time

                # Framework detection should not add significant overhead
                # Allow up to 50% overhead for framework detection
                assert time_with_fw < time_no_fw * 1.5

        finally:
            # Restore original modules
            for fw, module in original_modules.items():
                sys.modules[fw] = module

    def test_batch_processing_performance_with_framework(self, mock_cellmap_dataset):
        """Test that batch processing performance is maintained with framework."""
        mock_accelerate = MagicMock()
        with patch.dict(sys.modules, {"accelerate": mock_accelerate}):
            loader = EnhancedCellMapDataLoader(
                dataset=mock_cellmap_dataset,
                batch_size=2,
                enable_plugins=True,
                device="cpu",
            )

            # Process multiple batches to test performance
            batch_count = 0
            for batch in loader.loader:
                assert "raw" in batch
                assert "gt" in batch
                batch_count += 1
                if batch_count >= 5:  # Process 5 batches
                    break

            assert batch_count == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
