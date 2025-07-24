#!/usr/bin/env python3
"""
Test script to verify the DataLoader & Plugin System implementation.
"""

import sys
import os
import torch
import tempfile
import numpy as np

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_enhanced_dataloader():
    """Test the EnhancedCellMapDataLoader with plugin system."""
    from cellmap_data.enhanced_dataloader import EnhancedCellMapDataLoader
    from cellmap_data.dataset import CellMapDataset
    from cellmap_data.plugins import (
        PrefetchPlugin,
        AugmentationPlugin,
        DeviceTransferPlugin,
        MemoryOptimizationPlugin,
    )

    print("🔧 Testing Enhanced DataLoader with Plugin System...")

    # Create a minimal dataset for testing
    class TestDataset(CellMapDataset):
        def __new__(cls, length=5):
            return super().__new__(
                cls,
                raw_path="/tmp",
                target_path="/tmp",
                classes=["test"],
                input_arrays={"x": {"shape": (8, 8), "scale": (1.0, 1.0)}},
                target_arrays=None,
            )

        def __init__(self, length=5):
            self._length = length
            self.classes = ["test"]
            self.input_arrays = {"x": {"shape": (8, 8), "scale": (1.0, 1.0)}}
            self.target_arrays = None

        def __len__(self):
            return self._length

        def __getitem__(self, idx):
            return {"x": torch.randn(8, 8), "metadata": {"index": idx}}

        @property
        def has_data(self):
            return True

    dataset = TestDataset(10)

    # Test 1: Basic Enhanced DataLoader
    print("  ✓ Testing basic enhanced dataloader initialization...")
    loader = EnhancedCellMapDataLoader(
        dataset=dataset,
        batch_size=2,
        num_workers=0,
        device="cpu",
        enable_plugins=True,
        enable_prefetch=True,
        enable_augmentation=True,
        enable_device_transfer=True,
        enable_memory_optimization=True,
    )
    print(f"    - Initialized with {len(loader.plugin_manager.list_plugins())} plugins")

    # Test 2: Plugin Management
    print("  ✓ Testing plugin management...")
    plugins = loader.plugin_manager.list_plugins()
    expected_plugins = {
        "prefetch",
        "augmentation",
        "device_transfer",
        "memory_optimization",
    }
    assert (
        set(plugins) == expected_plugins
    ), f"Expected {expected_plugins}, got {set(plugins)}"
    print(f"    - All expected plugins present: {plugins}")

    # Test 3: Custom Plugin
    print("  ✓ Testing custom plugin addition...")

    class CustomPlugin:
        def __init__(self):
            self.name = "custom_test"
            self.priority = 75
            self.enabled = True

        def get_hook_methods(self):
            return {"post_sample": "process_sample"}

        def process_sample(self, sample, **kwargs):
            sample["custom_processed"] = True
            return sample

    custom_plugin = CustomPlugin()
    loader.add_plugin(custom_plugin)
    assert "custom_test" in loader.plugin_manager.list_plugins()
    print("    - Custom plugin added successfully")

    # Test 4: Data Loading with Plugins
    print("  ✓ Testing data loading with plugin hooks...")
    sample = loader[0]  # Get first sample
    assert isinstance(sample, dict)
    assert "x" in sample
    assert sample["x"].shape == (2, 8, 8)  # batch_size=2
    print(f"    - Sample loaded with shape: {sample['x'].shape}")

    # Test 5: Plugin Hook Execution
    print("  ✓ Testing plugin hook execution...")
    hooks = loader.plugin_manager.list_hooks()
    expected_hooks = {"pre_batch", "post_sample", "post_collate", "post_batch"}
    assert len(set(hooks) & expected_hooks) > 0, f"No expected hooks found in {hooks}"
    print(f"    - Available hooks: {hooks}")

    print("✅ Enhanced DataLoader tests passed!\n")


def test_plugin_system_components():
    """Test individual plugin system components."""
    from cellmap_data.plugins import PluginManager, PrefetchPlugin, AugmentationPlugin

    print("🔧 Testing Plugin System Components...")

    # Test 1: Plugin Manager
    print("  ✓ Testing PluginManager...")
    manager = PluginManager()

    # Test 2: PrefetchPlugin
    print("  ✓ Testing PrefetchPlugin...")
    prefetch_plugin = PrefetchPlugin(max_prefetch=4)
    manager.register_plugin(prefetch_plugin)
    assert "prefetch" in manager.list_plugins()

    # Test 3: AugmentationPlugin
    print("  ✓ Testing AugmentationPlugin...")

    def test_transform(sample):
        sample["transformed"] = True
        return sample

    aug_plugin = AugmentationPlugin(transforms=[test_transform])
    manager.register_plugin(aug_plugin)
    assert "augmentation" in manager.list_plugins()

    # Test 4: Hook Execution
    print("  ✓ Testing hook execution...")
    test_sample = {"data": torch.randn(3, 3)}
    processed_sample = manager.execute_hook("post_sample", test_sample)
    assert "transformed" in processed_sample
    print("    - Sample transformation successful")

    print("✅ Plugin system component tests passed!\n")


def test_device_manager_integration():
    """Test DeviceManager integration with the plugin system."""
    from cellmap_data.device.device_manager import DeviceManager
    from cellmap_data.plugins import DeviceTransferPlugin, MemoryOptimizationPlugin

    print("🔧 Testing DeviceManager Integration...")

    # Test 1: DeviceManager
    print("  ✓ Testing DeviceManager...")
    device_mgr = DeviceManager(device="cpu")
    test_tensor = torch.randn(2, 3)
    transferred = device_mgr.to_device(test_tensor, device="cpu")
    assert transferred.device.type == "cpu"
    print("    - Device transfer working")

    # Test 2: Memory pooling
    print("  ✓ Testing memory pooling...")
    pooled = device_mgr.pool_tensor(test_tensor)
    assert pooled.shape == test_tensor.shape
    print("    - Memory pooling working")

    # Test 3: DeviceTransferPlugin
    print("  ✓ Testing DeviceTransferPlugin...")
    device_plugin = DeviceTransferPlugin(device_manager=device_mgr)
    batch = {"x": torch.randn(2, 3), "y": torch.randn(2, 5)}
    transferred_batch = device_plugin.transfer_to_device(batch, device="cpu")
    assert all(tensor.device.type == "cpu" for tensor in transferred_batch.values())
    print("    - Device transfer plugin working")

    print("✅ DeviceManager integration tests passed!\n")


if __name__ == "__main__":
    print("🚀 Testing DataLoader & Plugin System Implementation\n")

    try:
        test_enhanced_dataloader()
        test_plugin_system_components()
        test_device_manager_integration()

        print("🎉 All DataLoader & Plugin System tests passed!")
        print("✅ Priority 3: DataLoader & Plugin System - COMPLETED")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
