"""
Simplified framework compatibility integration tests.
Focus on core framework detection and device transfer behavior.
"""

import os
import sys
import pytest
import torch
from unittest.mock import patch, MagicMock

from cellmap_data.device.device_manager import DeviceManager
from cellmap_data.plugins import DeviceTransferPlugin, MemoryOptimizationPlugin


class TestCoreFrameworkCompatibility:
    """Test core framework compatibility without complex setup."""

    def test_framework_detection_accelerate(self):
        """Test Accelerate framework detection."""
        mock_accelerate = MagicMock()
        with patch.dict(sys.modules, {"accelerate": mock_accelerate}):
            device_manager = DeviceManager()
            assert device_manager.framework == "accelerate"

    def test_framework_detection_pytorch_lightning(self):
        """Test PyTorch Lightning framework detection."""
        mock_pl = MagicMock()
        with patch.dict(sys.modules, {"pytorch_lightning": mock_pl}):
            device_manager = DeviceManager()
            assert device_manager.framework == "pytorch_lightning"

    def test_framework_detection_deepspeed(self):
        """Test DeepSpeed framework detection."""
        mock_deepspeed = MagicMock()
        with patch.dict(sys.modules, {"deepspeed": mock_deepspeed}):
            device_manager = DeviceManager()
            assert device_manager.framework == "deepspeed"

    def test_device_transfer_skipped_with_framework(self):
        """Test that device transfer is skipped when framework is detected."""
        mock_pl = MagicMock()
        with patch.dict(sys.modules, {"pytorch_lightning": mock_pl}):
            device_manager = DeviceManager(device="cpu")

            # Create a tensor
            original_tensor = torch.randn(2, 3, 4)

            # Transfer should be skipped due to framework detection
            result = device_manager.to_device(original_tensor, device="cuda")

            # Should return the same tensor (no transfer performed)
            assert result is original_tensor

    def test_device_transfer_performed_without_framework(self):
        """Test that device transfer is performed when no framework is detected."""
        # Temporarily remove frameworks from sys.modules
        frameworks = ["accelerate", "pytorch_lightning", "deepspeed"]
        original_modules = {}

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

            # Should complete the transfer
            assert result.device.type == "cpu"

        finally:
            # Restore original modules
            for fw, module in original_modules.items():
                sys.modules[fw] = module

    def test_device_transfer_plugin_with_framework(self):
        """Test DeviceTransferPlugin respects framework detection."""
        mock_accelerate = MagicMock()
        with patch.dict(sys.modules, {"accelerate": mock_accelerate}):
            device_manager = DeviceManager(device="cpu")
            plugin = DeviceTransferPlugin(device_manager=device_manager)

            # Create batch
            batch = {
                "raw": torch.randn(2, 1, 64, 64),
                "gt": torch.randint(0, 2, (2, 1, 64, 64)),
            }
            original_raw = batch["raw"].clone()
            original_gt = batch["gt"].clone()

            # Plugin should return batch unchanged due to framework detection
            result = plugin.transfer_to_device(batch, device="cuda")

            # Tensors should remain unchanged
            assert torch.equal(result["raw"], original_raw)
            assert torch.equal(result["gt"], original_gt)

    def test_memory_optimization_plugin_with_framework(self):
        """Test MemoryOptimizationPlugin works with framework detection."""
        mock_deepspeed = MagicMock()
        with patch.dict(sys.modules, {"deepspeed": mock_deepspeed}):
            device_manager = DeviceManager(device="cpu")
            plugin = MemoryOptimizationPlugin(device_manager=device_manager)

            # Create batch
            batch = {
                "raw": torch.randn(2, 1, 64, 64),
                "gt": torch.randint(0, 2, (2, 1, 64, 64)),
            }

            # Plugin should still process the batch
            result = plugin.optimize_memory(batch)

            # Should return a result with the same keys
            assert "raw" in result
            assert "gt" in result
            assert isinstance(result["raw"], torch.Tensor)
            assert isinstance(result["gt"], torch.Tensor)

    def test_multiple_frameworks_priority(self):
        """Test that framework detection follows priority order."""
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
            device_manager = DeviceManager()
            # Should detect accelerate first (based on order in detect_framework)
            assert device_manager.framework == "accelerate"

    def test_framework_aware_device_selection(self):
        """Test device selection with framework awareness."""
        mock_pl = MagicMock()
        with patch.dict(sys.modules, {"pytorch_lightning": mock_pl}):
            device_manager = DeviceManager()

            # Framework detection should not interfere with device selection
            device = device_manager.select_device("cpu")
            assert device.type == "cpu"

            # Should still detect framework
            assert device_manager.framework == "pytorch_lightning"

    def test_cpu_memory_pooling_with_framework(self):
        """Test CPU memory pooling behavior with framework."""
        mock_accelerate = MagicMock()
        with patch.dict(sys.modules, {"accelerate": mock_accelerate}):
            device_manager = DeviceManager(device="cpu")

            # Create CPU tensor
            tensor = torch.randn(4, 4, dtype=torch.float32)

            # Pool tensor should work regardless of framework
            pooled = device_manager.pool_tensor(tensor)
            assert isinstance(pooled, torch.Tensor)
            assert pooled.shape == tensor.shape
            assert pooled.dtype == tensor.dtype

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_memory_pooling_with_framework(self):
        """Test CUDA memory pooling behavior with framework."""
        mock_pl = MagicMock()
        with patch.dict(sys.modules, {"pytorch_lightning": mock_pl}):
            device_manager = DeviceManager(device="cuda")

            # Create CUDA tensor
            tensor = torch.randn(4, 4, device="cuda")

            # Pool tensor should work (relies on torch's built-in pooling)
            pooled = device_manager.pool_tensor(tensor)
            assert isinstance(pooled, torch.Tensor)
            assert pooled.device.type == "cuda"


class TestFrameworkIntegrationEdgeCases:
    """Test edge cases in framework integration."""

    def test_framework_detection_with_missing_module(self):
        """Test behavior when framework module exists but is incomplete."""
        # Create a mock module that's missing expected attributes
        incomplete_mock = type("MockModule", (), {})()

        with patch.dict(sys.modules, {"pytorch_lightning": incomplete_mock}):
            device_manager = DeviceManager()
            # Should still detect the framework
            assert device_manager.framework == "pytorch_lightning"

    def test_device_manager_reinitialization(self):
        """Test that DeviceManager can be reinitialized with different frameworks."""
        # First with no framework
        frameworks = ["accelerate", "pytorch_lightning", "deepspeed"]
        original_modules = {}

        for fw in frameworks:
            if fw in sys.modules:
                original_modules[fw] = sys.modules[fw]
                del sys.modules[fw]

        try:
            device_manager1 = DeviceManager()
            assert device_manager1.framework is None

            # Then with framework
            mock_pl = MagicMock()
            with patch.dict(sys.modules, {"pytorch_lightning": mock_pl}):
                device_manager2 = DeviceManager()
                assert device_manager2.framework == "pytorch_lightning"

        finally:
            # Restore original modules
            for fw, module in original_modules.items():
                sys.modules[fw] = module

    def test_plugin_behavior_consistency(self):
        """Test that plugins behave consistently across framework states."""
        # Test without framework
        frameworks = ["accelerate", "pytorch_lightning", "deepspeed"]
        original_modules = {}

        for fw in frameworks:
            if fw in sys.modules:
                original_modules[fw] = sys.modules[fw]
                del sys.modules[fw]

        try:
            device_manager_no_fw = DeviceManager(device="cpu")
            plugin_no_fw = DeviceTransferPlugin(device_manager=device_manager_no_fw)

            batch = {"data": torch.randn(2, 3)}
            result_no_fw = plugin_no_fw.transfer_to_device(batch)

            # Test with framework
            mock_accelerate = MagicMock()
            with patch.dict(sys.modules, {"accelerate": mock_accelerate}):
                device_manager_with_fw = DeviceManager(device="cpu")
                plugin_with_fw = DeviceTransferPlugin(
                    device_manager=device_manager_with_fw
                )

                result_with_fw = plugin_with_fw.transfer_to_device(batch)

                # Both should return valid results
                assert "data" in result_no_fw
                assert "data" in result_with_fw
                assert isinstance(result_no_fw["data"], torch.Tensor)
                assert isinstance(result_with_fw["data"], torch.Tensor)

        finally:
            # Restore original modules
            for fw, module in original_modules.items():
                sys.modules[fw] = module


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
