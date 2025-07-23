"""
Integration tests for framework compatibility.
Tests how cellmap-data works with different ML frameworks (PyTorch Lightning, Accelerate, etc.).
"""

import os
import sys
import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock
from typing import Dict, Any

from cellmap_data.enhanced_dataloader import EnhancedCellMapDataLoader
from cellmap_data.device.device_manager import DeviceManager
from cellmap_data.plugins import (
    PluginManager,
    DeviceTransferPlugin,
    MemoryOptimizationPlugin,
    PrefetchPlugin,
    AugmentationPlugin,
)
from cellmap_data.dataset import CellMapDataset
from cellmap_data.multidataset import CellMapMultiDataset


@pytest.fixture
def mock_dataset():
    """Create a minimal dataset for testing."""
    dataset = MagicMock(spec=CellMapDataset)
    dataset.classes = ["class1", "class2"]
    dataset.__len__ = MagicMock(return_value=10)
    dataset.__getitem__ = MagicMock(
        side_effect=lambda idx: {
            "raw": torch.randn(1, 64, 64),
            "gt": torch.randint(0, 2, (1, 64, 64)),
            "index": idx,
        }
    )
    dataset.to = MagicMock(return_value=None)
    return dataset


@pytest.fixture
def mock_multidataset():
    """Create a minimal multidataset for testing."""
    dataset = MagicMock(spec=CellMapMultiDataset)
    dataset.classes = ["class1", "class2"]
    dataset.__len__ = MagicMock(return_value=10)
    dataset.__getitem__ = MagicMock(
        side_effect=lambda idx: {
            "raw": torch.randn(1, 64, 64),
            "gt": torch.randint(0, 2, (1, 64, 64)),
            "index": idx,
        }
    )
    dataset.to = MagicMock(return_value=None)
    return dataset


class TestFrameworkDetection:
    """Test framework detection capabilities."""

    def test_no_framework_detected(self):
        """Test when no framework is present."""
        # Temporarily remove frameworks from sys.modules
        original_modules = {}
        frameworks = ["accelerate", "pytorch_lightning", "deepspeed"]

        for fw in frameworks:
            if fw in sys.modules:
                original_modules[fw] = sys.modules[fw]
                del sys.modules[fw]

        try:
            device_manager = DeviceManager()
            assert device_manager.framework is None
        finally:
            # Restore original modules
            for fw, module in original_modules.items():
                sys.modules[fw] = module

    def test_pytorch_lightning_detected(self):
        """Test PyTorch Lightning framework detection."""
        mock_pl = MagicMock()
        with patch.dict(sys.modules, {"pytorch_lightning": mock_pl}):
            device_manager = DeviceManager()
            assert device_manager.framework == "pytorch_lightning"

    def test_accelerate_detected(self):
        """Test Accelerate framework detection."""
        mock_accelerate = MagicMock()
        with patch.dict(sys.modules, {"accelerate": mock_accelerate}):
            device_manager = DeviceManager()
            assert device_manager.framework == "accelerate"

    def test_deepspeed_detected(self):
        """Test DeepSpeed framework detection."""
        mock_deepspeed = MagicMock()
        with patch.dict(sys.modules, {"deepspeed": mock_deepspeed}):
            device_manager = DeviceManager()
            assert device_manager.framework == "deepspeed"

    def test_framework_priority_order(self):
        """Test that first detected framework is returned."""
        mock_accelerate = MagicMock()
        mock_pl = MagicMock()

        with patch.dict(
            sys.modules, {"accelerate": mock_accelerate, "pytorch_lightning": mock_pl}
        ):
            device_manager = DeviceManager()
            # Should return accelerate as it's checked first
            assert device_manager.framework == "accelerate"


class TestDeviceTransferCompatibility:
    """Test device transfer behavior with different frameworks."""

    def test_device_transfer_without_framework(self, mock_dataset):
        """Test device transfer when no framework is managing devices."""
        # Save original modules to restore later
        original_modules = {}
        frameworks = ["accelerate", "pytorch_lightning", "deepspeed"]

        for fw in frameworks:
            if fw in sys.modules:
                original_modules[fw] = sys.modules[fw]
                del sys.modules[fw]

        try:
            device_manager = DeviceManager(device="cpu")
            assert device_manager.framework is None

            # Create tensor and test transfer
            tensor = torch.randn(2, 3, 4)
            result = device_manager.to_device(tensor, device="cpu")
            assert result.device.type == "cpu"
        finally:
            # Restore original modules
            for fw, module in original_modules.items():
                sys.modules[fw] = module

    def test_device_transfer_with_pytorch_lightning(self, mock_dataset):
        """Test that device transfer is skipped with PyTorch Lightning."""
        mock_pl = MagicMock()
        with patch.dict(sys.modules, {"pytorch_lightning": mock_pl}):
            device_manager = DeviceManager(device="cpu")
            assert device_manager.framework == "pytorch_lightning"

            # Create tensor on CPU
            original_tensor = torch.randn(2, 3, 4)
            result = device_manager.to_device(original_tensor, device="cuda")

            # Should return original tensor unchanged (framework manages devices)
            assert result is original_tensor
            assert result.device.type == "cpu"

    def test_device_transfer_with_accelerate(self, mock_dataset):
        """Test that device transfer is skipped with Accelerate."""
        mock_accelerate = MagicMock()
        with patch.dict(sys.modules, {"accelerate": mock_accelerate}):
            device_manager = DeviceManager(device="cpu")
            assert device_manager.framework == "accelerate"

            # Create tensor and test transfer is skipped
            original_tensor = torch.randn(2, 3, 4)
            result = device_manager.to_device(original_tensor, device="cuda")

            # Should return original tensor unchanged
            assert result is original_tensor

    def test_device_transfer_with_deepspeed(self, mock_dataset):
        """Test that device transfer is skipped with DeepSpeed."""
        mock_deepspeed = MagicMock()
        with patch.dict(sys.modules, {"deepspeed": mock_deepspeed}):
            device_manager = DeviceManager(device="cpu")
            assert device_manager.framework == "deepspeed"

            # Create tensor and test transfer is skipped
            original_tensor = torch.randn(2, 3, 4)
            result = device_manager.to_device(original_tensor, device="cuda")

            # Should return original tensor unchanged
            assert result is original_tensor


class TestPluginFrameworkIntegration:
    """Test plugin system integration with different frameworks."""

    def test_device_transfer_plugin_with_framework(self, mock_dataset):
        """Test DeviceTransferPlugin behavior with external framework."""
        mock_pl = MagicMock()
        with patch.dict(sys.modules, {"pytorch_lightning": mock_pl}):
            device_manager = DeviceManager(device="cpu")
            plugin = DeviceTransferPlugin(device_manager=device_manager)

            # Create batch
            batch = {
                "raw": torch.randn(2, 1, 64, 64),
                "gt": torch.randint(0, 2, (2, 1, 64, 64)),
            }

            # Plugin should return batch unchanged when framework is detected
            result = plugin.transfer_to_device(batch, device="cuda")

            # Tensors should remain unchanged (framework manages devices)
            assert torch.equal(result["raw"], batch["raw"])
            assert torch.equal(result["gt"], batch["gt"])

    def test_memory_optimization_plugin_with_framework(self, mock_dataset):
        """Test MemoryOptimizationPlugin behavior with external framework."""
        mock_accelerate = MagicMock()
        with patch.dict(sys.modules, {"accelerate": mock_accelerate}):
            device_manager = DeviceManager(device="cpu")
            plugin = MemoryOptimizationPlugin(device_manager=device_manager)

            # Create batch
            batch = {
                "raw": torch.randn(2, 1, 64, 64),
                "gt": torch.randint(0, 2, (2, 1, 64, 64)),
            }

            # Plugin should still apply memory optimization
            result = plugin.optimize_memory(batch)

            # Should process tensors through device manager
            assert "raw" in result
            assert "gt" in result

    def test_enhanced_dataloader_with_framework(self, mock_dataset):
        """Test EnhancedCellMapDataLoader with external framework."""
        mock_pl = MagicMock()
        with patch.dict(sys.modules, {"pytorch_lightning": mock_pl}):
            loader = EnhancedCellMapDataLoader(
                dataset=mock_dataset,
                batch_size=2,
                enable_plugins=True,
                enable_device_transfer=True,
                device="cpu",
            )

            # Should detect framework through device manager
            assert loader.device_manager.framework == "pytorch_lightning"

            # Should still have device transfer plugin registered
            device_plugin = loader.get_plugin("device_transfer")
            assert device_plugin is not None
            assert isinstance(device_plugin, DeviceTransferPlugin)


class TestFrameworkSpecificOptimizations:
    """Test framework-specific optimizations and behaviors."""

    def test_cuda_streams_with_framework(self, mock_dataset):
        """Test CUDA stream optimization with external framework."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        mock_pl = MagicMock()
        with patch.dict(sys.modules, {"pytorch_lightning": mock_pl}):
            loader = EnhancedCellMapDataLoader(
                dataset=mock_dataset, batch_size=4, device="cuda", enable_plugins=True
            )

            # Framework detection should not interfere with stream optimization
            assert loader.device_manager.framework == "pytorch_lightning"

            # Loader should still be functional
            assert loader.device == "cuda"

    def test_multiprocessing_with_framework(self, mock_dataset):
        """Test multiprocessing behavior with external framework."""
        mock_accelerate = MagicMock()
        with patch.dict(sys.modules, {"accelerate": mock_accelerate}):
            loader = EnhancedCellMapDataLoader(
                dataset=mock_dataset,
                batch_size=2,
                num_workers=2,  # Enable multiprocessing
                enable_plugins=True,
            )

            # Should work regardless of framework
            assert loader.num_workers == 2
            assert loader.device_manager.framework == "accelerate"

    def test_prefetch_plugin_with_framework(self, mock_dataset):
        """Test PrefetchPlugin works with external frameworks."""
        mock_deepspeed = MagicMock()
        with patch.dict(sys.modules, {"deepspeed": mock_deepspeed}):
            loader = EnhancedCellMapDataLoader(
                dataset=mock_dataset,
                batch_size=2,
                enable_plugins=True,
                enable_prefetch=True,
            )

            # Prefetch should work regardless of framework
            prefetch_plugin = loader.get_plugin("prefetch")
            assert prefetch_plugin is not None
            assert isinstance(prefetch_plugin, PrefetchPlugin)


class TestFrameworkInteroperability:
    """Test interoperability between cellmap-data and ML frameworks."""

    def test_dataloader_in_pytorch_lightning_context(self, mock_dataset):
        """Test using CellMapDataLoader within PyTorch Lightning."""
        mock_pl = MagicMock()
        mock_trainer = MagicMock()
        mock_trainer.device = torch.device("cpu")

        with patch.dict(sys.modules, {"pytorch_lightning": mock_pl}):
            loader = EnhancedCellMapDataLoader(
                dataset=mock_dataset,
                batch_size=2,
                enable_plugins=True,
                enable_device_transfer=True,
            )

            # Simulate PyTorch Lightning trainer using our loader
            assert loader.device_manager.framework == "pytorch_lightning"

            # Should not interfere with PL's device management
            batch = {
                "raw": torch.randn(2, 1, 64, 64),
                "gt": torch.randint(0, 2, (2, 1, 64, 64)),
            }

            device_plugin = loader.get_plugin("device_transfer")
            result = device_plugin.transfer_to_device(batch)

            # Should return tensors unchanged (PL manages devices)
            assert torch.equal(result["raw"], batch["raw"])

    def test_dataloader_in_accelerate_context(self, mock_dataset):
        """Test using CellMapDataLoader with Accelerate."""
        mock_accelerate = MagicMock()

        with patch.dict(sys.modules, {"accelerate": mock_accelerate}):
            loader = EnhancedCellMapDataLoader(
                dataset=mock_dataset, batch_size=2, enable_plugins=True
            )

            # Should detect Accelerate
            assert loader.device_manager.framework == "accelerate"

            # All plugins should still be functional but respect framework
            assert len(loader.plugin_manager.list_plugins()) > 0

    def test_mixed_framework_scenario(self, mock_dataset):
        """Test behavior when multiple frameworks are available."""
        mock_accelerate = MagicMock()
        mock_pl = MagicMock()
        mock_deepspeed = MagicMock()

        with patch.dict(
            sys.modules,
            {
                "accelerate": mock_accelerate,
                "pytorch_lightning": mock_pl,
                "deepspeed": mock_deepspeed,
            },
        ):
            loader = EnhancedCellMapDataLoader(
                dataset=mock_dataset, batch_size=2, enable_plugins=True
            )

            # Should detect first available framework (accelerate)
            assert loader.device_manager.framework == "accelerate"


class TestFrameworkConfigurationOptions:
    """Test configuration options for framework compatibility."""

    def test_force_framework_detection_off(self, mock_dataset):
        """Test forcing framework detection to be disabled."""
        mock_pl = MagicMock()

        with patch.dict(sys.modules, {"pytorch_lightning": mock_pl}):
            # Create device manager with forced framework=None
            device_manager = DeviceManager()
            device_manager.framework = None  # Force override

            loader = EnhancedCellMapDataLoader(
                dataset=mock_dataset,
                batch_size=2,
                enable_plugins=True,
                enable_device_transfer=True,
            )
            loader.device_manager = device_manager  # Override with our custom one

            # Should behave as if no framework is present
            batch = {
                "raw": torch.randn(2, 1, 64, 64),
                "gt": torch.randint(0, 2, (2, 1, 64, 64)),
            }

            device_plugin = loader.get_plugin("device_transfer")
            # Should now perform actual device transfer
            result = device_plugin.transfer_to_device(batch, device="cpu")
            assert "raw" in result
            assert "gt" in result

    def test_custom_device_manager_with_framework(self, mock_dataset):
        """Test using custom DeviceManager with framework awareness."""

        class CustomDeviceManager(DeviceManager):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.transfer_count = 0

            def to_device(self, tensor, device=None, non_blocking=True):
                self.transfer_count += 1
                return super().to_device(tensor, device, non_blocking)

        mock_accelerate = MagicMock()
        with patch.dict(sys.modules, {"accelerate": mock_accelerate}):
            custom_manager = CustomDeviceManager(device="cpu")

            loader = EnhancedCellMapDataLoader(
                dataset=mock_dataset, batch_size=2, enable_plugins=True
            )
            loader.device_manager = custom_manager

            # Should respect framework detection
            assert custom_manager.framework == "accelerate"

            # Test device transfer
            batch = {"raw": torch.randn(2, 1, 64, 64)}
            device_plugin = loader.get_plugin("device_transfer")
            device_plugin.transfer_to_device(batch)

            # Should not increment counter due to framework detection
            assert custom_manager.transfer_count == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
