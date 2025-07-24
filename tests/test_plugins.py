import pytest
import torch
from cellmap_data.plugins import (
    PluginManager,
    DataLoaderPlugin,
    PrefetchPlugin,
    AugmentationPlugin,
    DeviceTransferPlugin,
    MemoryOptimizationPlugin,
)
from unittest.mock import MagicMock, Mock


class MockPlugin(DataLoaderPlugin):
    def __init__(self, name="mock_plugin", priority=0):
        super().__init__(name, priority)
        self.hooks_called = []

    def get_hook_methods(self):
        return {
            "pre_batch": "pre_batch_hook",
            "post_sample": "post_sample_hook",
        }

    def pre_batch_hook(self, *args, **kwargs):
        self.hooks_called.append("pre_batch")

    def post_sample_hook(self, sample, *args, **kwargs):
        self.hooks_called.append("post_sample")
        sample["mock"] = True
        return sample


def test_plugin_manager_registration():
    """Test that plugins can be registered, unregistered, and listed."""
    manager = PluginManager()
    plugin = MockPlugin()

    manager.register_plugin(plugin)
    assert manager.list_plugins() == ["mock_plugin"]
    assert manager.get_plugin("mock_plugin") == plugin

    manager.unregister_plugin("mock_plugin")
    assert manager.list_plugins() == []
    assert manager.get_plugin("mock_plugin") is None


def test_plugin_manager_hook_execution():
    """Test that hooks are executed correctly."""
    manager = PluginManager()
    plugin = MockPlugin()
    manager.register_plugin(plugin)

    # Test a non-modifying hook
    manager.execute_hook("pre_batch")
    assert plugin.hooks_called == ["pre_batch"]

    # Test a modifying hook
    sample = {"data": 1}
    modified_sample = manager.execute_hook("post_sample", sample)
    assert plugin.hooks_called == ["pre_batch", "post_sample"]
    assert modified_sample == {"data": 1, "mock": True}


def test_plugin_priority():
    """Test that plugins are executed in priority order."""
    manager = PluginManager()
    plugin1 = MockPlugin(name="p1", priority=10)
    plugin2 = MockPlugin(name="p2", priority=20)

    manager.register_plugin(plugin1)
    manager.register_plugin(plugin2)

    # Rebuild cache to see the order
    manager._rebuild_hook_cache()
    pre_batch_hooks = manager._hook_cache["pre_batch"]
    assert [p.name for p, m in pre_batch_hooks] == ["p2", "p1"]


def test_prefetch_plugin():
    """Test the PrefetchPlugin."""
    dataset = MagicMock()
    dataset.__getitem__.return_value = "data"
    plugin = PrefetchPlugin(max_prefetch=2)
    plugin.setup_prefetch(dataset, indices=[0, 1, 2, 3])
    assert plugin._prefetcher is not None
    # Consume from the prefetcher to prevent deadlock
    for _ in range(2):
        plugin._prefetcher.get()
    plugin.cleanup_prefetch()
    assert plugin._prefetcher is None


def test_augmentation_plugin():
    """Test the AugmentationPlugin."""
    transform = Mock(return_value={"data": "augmented"})
    plugin = AugmentationPlugin(transforms=[transform])
    sample = {"data": "original"}
    result = plugin.apply_transforms(sample)
    transform.assert_called_once_with(sample)
    assert result == {"data": "augmented"}


def test_device_transfer_plugin():
    """Test the DeviceTransferPlugin."""
    device_manager = MagicMock()
    device_manager.to_device = Mock(side_effect=lambda x, device, non_blocking: x)
    plugin = DeviceTransferPlugin(device_manager=device_manager)
    batch = {"data": torch.tensor([1])}
    plugin.transfer_to_device(batch, device="cpu")
    device_manager.to_device.assert_called_once()


def test_memory_optimization_plugin():
    """Test the MemoryOptimizationPlugin."""
    device_manager = MagicMock()
    device_manager.pool_tensor = Mock(side_effect=lambda x: x)
    plugin = MemoryOptimizationPlugin(device_manager=device_manager)
    batch = {"data": torch.tensor([1])}
    plugin.optimize_memory(batch)
    device_manager.pool_tensor.assert_called_once()
