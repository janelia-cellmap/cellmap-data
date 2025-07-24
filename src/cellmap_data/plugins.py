"""
Plugin system for CellMapDataLoader.
Provides extensible hooks for data processing pipeline stages.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple
import torch
import logging

logger = logging.getLogger(__name__)


class DataLoaderPlugin(ABC):
    """
    Base class for all DataLoader plugins.
    Plugins can hook into various stages of the data loading pipeline.
    """

    def __init__(self, name: str, priority: int = 0):
        """
        Initialize plugin.

        Args:
            name: Plugin name for identification
            priority: Execution priority (higher runs first)
        """
        self.name = name
        self.priority = priority
        self.enabled = True

    def enable(self):
        """Enable this plugin."""
        self.enabled = True

    def disable(self):
        """Disable this plugin."""
        self.enabled = False

    @abstractmethod
    def get_hook_methods(self) -> Dict[str, str]:
        """
        Return mapping of hook names to method names this plugin implements.

        Returns:
            Dict mapping hook names to method names
        """
        pass


class PrefetchPlugin(DataLoaderPlugin):
    """Plugin for asynchronous data prefetching."""

    def __init__(self, max_prefetch: int = 8, priority: int = 100):
        super().__init__("prefetch", priority)
        self.max_prefetch = max_prefetch
        self._prefetcher = None

    def get_hook_methods(self) -> Dict[str, str]:
        return {"pre_batch": "setup_prefetch", "post_batch": "cleanup_prefetch"}

    def setup_prefetch(self, dataset, indices: Sequence[int], **kwargs):
        """Initialize prefetcher for the given indices."""
        if not self.enabled:
            return indices

        from .prefetch import Prefetcher

        if self._prefetcher is None:
            self._prefetcher = Prefetcher(
                dataset.__getitem__, max_prefetch=self.max_prefetch
            )

        logger.debug(f"PrefetchPlugin: Starting prefetch for {len(indices)} indices")
        self._prefetcher.start(indices)
        return indices

    def cleanup_prefetch(self, **kwargs):
        """Clean up prefetcher resources."""
        if self._prefetcher is not None:
            self._prefetcher.stop()
            self._prefetcher = None


class AugmentationPlugin(DataLoaderPlugin):
    """Plugin for data augmentation during loading."""

    def __init__(self, transforms: Optional[List[Any]] = None, priority: int = 50):
        super().__init__("augmentation", priority)
        self.transforms = transforms or []

    def get_hook_methods(self) -> Dict[str, str]:
        return {"post_sample": "apply_transforms"}

    def apply_transforms(self, sample: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Apply augmentation transforms to a sample."""
        if not self.enabled or not self.transforms:
            return sample

        for transform in self.transforms:
            if hasattr(transform, "__call__"):
                sample = transform(sample)

        return sample

    def add_transform(self, transform):
        """Add a new transform to the pipeline."""
        self.transforms.append(transform)

    def clear_transforms(self):
        """Remove all transforms."""
        self.transforms.clear()


class DeviceTransferPlugin(DataLoaderPlugin):
    """Plugin for intelligent device transfer with framework awareness."""

    def __init__(self, device_manager=None, priority: int = 10):
        super().__init__("device_transfer", priority)
        self.device_manager = device_manager

    def get_hook_methods(self) -> Dict[str, str]:
        return {"post_collate": "transfer_to_device"}

    def transfer_to_device(
        self, batch: Dict[str, torch.Tensor], device=None, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Transfer batch to target device using DeviceManager."""
        if not self.enabled or self.device_manager is None:
            return batch

        target_device = device or getattr(self.device_manager, "device", "cpu")

        for key, tensor in batch.items():
            if isinstance(tensor, torch.Tensor):
                batch[key] = self.device_manager.to_device(
                    tensor, device=target_device, non_blocking=True
                )

        return batch


class MemoryOptimizationPlugin(DataLoaderPlugin):
    """Plugin for memory optimization using pooling and efficient allocation."""

    def __init__(self, device_manager=None, priority: int = 20):
        super().__init__("memory_optimization", priority)
        self.device_manager = device_manager

    def get_hook_methods(self) -> Dict[str, str]:
        return {"post_collate": "optimize_memory"}

    def optimize_memory(
        self, batch: Dict[str, torch.Tensor], **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Apply memory optimizations to batch tensors."""
        if not self.enabled or self.device_manager is None:
            return batch

        for key, tensor in batch.items():
            if isinstance(tensor, torch.Tensor):
                batch[key] = self.device_manager.pool_tensor(tensor)

        return batch


class PluginManager:
    """
    Manages plugins for the DataLoader and executes hooks at appropriate stages.
    """

    def __init__(self):
        self.plugins: List[DataLoaderPlugin] = []
        self._hook_cache: Dict[str, List[Tuple[DataLoaderPlugin, str]]] = {}
        self._cache_dirty = True

    def register_plugin(self, plugin: DataLoaderPlugin):
        """Register a new plugin."""
        self.plugins.append(plugin)
        self._cache_dirty = True
        logger.debug(f"Registered plugin: {plugin.name} (priority: {plugin.priority})")

    def unregister_plugin(self, plugin_name: str):
        """Unregister a plugin by name."""
        self.plugins = [p for p in self.plugins if p.name != plugin_name]
        self._cache_dirty = True
        logger.debug(f"Unregistered plugin: {plugin_name}")

    def get_plugin(self, plugin_name: str) -> Optional[DataLoaderPlugin]:
        """Get a plugin by name."""
        for plugin in self.plugins:
            if plugin.name == plugin_name:
                return plugin
        return None

    def _rebuild_hook_cache(self):
        """Rebuild the hook cache for efficient hook execution."""
        if not self._cache_dirty:
            return

        self._hook_cache.clear()

        for plugin in self.plugins:
            if not plugin.enabled:
                continue

            hook_methods = plugin.get_hook_methods()
            for hook_name, method_name in hook_methods.items():
                if hook_name not in self._hook_cache:
                    self._hook_cache[hook_name] = []
                self._hook_cache[hook_name].append((plugin, method_name))

        # Sort plugins by priority (higher priority first)
        for hook_name in self._hook_cache:
            self._hook_cache[hook_name].sort(key=lambda x: x[0].priority, reverse=True)

        self._cache_dirty = False

    def execute_hook(self, hook_name: str, *args, **kwargs) -> Any:
        """
        Execute all plugins registered for a specific hook.

        Args:
            hook_name: Name of the hook to execute
            *args, **kwargs: Arguments to pass to hook methods

        Returns:
            Modified data (for data transformation hooks)
        """
        self._rebuild_hook_cache()

        if hook_name not in self._hook_cache:
            # If no plugins for this hook, return first arg (pass-through)
            return args[0] if args else None

        result = args[0] if args else None

        for plugin, method_name in self._hook_cache[hook_name]:
            try:
                method = getattr(plugin, method_name)
                if hook_name in [
                    "post_sample",
                    "post_collate",
                    "transfer_to_device",
                    "optimize_memory",
                ]:
                    # Data transformation hooks - pass result forward
                    result = method(result, *args[1:], **kwargs)
                else:
                    # Non-transformation hooks - call with original args
                    method(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"Error executing hook {hook_name} in plugin {plugin.name}: {e}"
                )
                # Continue with other plugins

        return result

    def list_plugins(self) -> List[str]:
        """List all registered plugin names."""
        return [plugin.name for plugin in self.plugins]

    def list_hooks(self) -> List[str]:
        """List all available hook names."""
        self._rebuild_hook_cache()
        return list(self._hook_cache.keys())
