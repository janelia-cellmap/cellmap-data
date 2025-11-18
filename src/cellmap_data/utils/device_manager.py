"""
CellMap Device Management Module

Centralized device handling logic extracted from monolithic classes to provide
consistent device management across the CellMap data loading pipeline.

This module consolidates device detection, configuration, and management that was
previously duplicated across CellMapDataset, CellMapImage, and other classes.
"""

import torch
from typing import Optional, Union, Dict, Any
from ..utils.logging_config import get_logger

logger = get_logger("device_manager")


class DeviceManager:
    """Centralized device management for CellMap data operations.
    
    Provides consistent device detection, configuration, and management
    across all CellMap components, eliminating device handling duplication.
    
    Attributes:
        current_device: Currently configured torch device
        cuda_available: Whether CUDA is available
        mps_available: Whether MPS (Apple Silicon) is available
        device_capabilities: Device-specific capabilities information
    
    Examples:
        >>> manager = DeviceManager()
        >>> manager.current_device
        device(type='cuda', index=0)
        
        >>> manager = DeviceManager(device='cpu')
        >>> manager.current_device
        device(type='cpu')
        
        >>> manager.to('cuda')
        >>> manager.get_optimal_device()
        device(type='cuda', index=0)
    """
    
    def __init__(self, device: Optional[Union[str, torch.device]] = None):
        """Initialize device manager with optional device specification.
        
        Args:
            device: Target device specification. If None, auto-detects optimal device.
                   Can be string ('cuda', 'mps', 'cpu') or torch.device object.
                   
        Examples:
            >>> DeviceManager()  # Auto-detect optimal device
            >>> DeviceManager('cuda')  # Force CUDA
            >>> DeviceManager(torch.device('cpu'))  # Force CPU
        """
        self.cuda_available = torch.cuda.is_available()
        self.mps_available = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
        
        # Set device with fallback logic
        if device is not None:
            self._device = torch.device(device)
            logger.debug(f"DeviceManager initialized with specified device: {self._device}")
        else:
            self._device = self._auto_detect_device()
            logger.debug(f"DeviceManager auto-detected device: {self._device}")
            
        # Cache device capabilities
        self.device_capabilities = self._get_device_capabilities()
        
    def _auto_detect_device(self) -> torch.device:
        """Auto-detect the optimal available device.
        
        Returns:
            torch.device: Best available device in order: CUDA > MPS > CPU
        """
        if self.cuda_available:
            device = torch.device("cuda")
            logger.debug("Auto-detected CUDA device")
            return device
        elif self.mps_available:
            device = torch.device("mps")
            logger.debug("Auto-detected MPS device")
            return device
        else:
            device = torch.device("cpu")
            logger.debug("Auto-detected CPU device")
            return device
            
    def _get_device_capabilities(self) -> Dict[str, Any]:
        """Get device-specific capabilities and information.
        
        Returns:
            Dict[str, Any]: Device capabilities including memory, compute capability, etc.
        """
        capabilities: Dict[str, Any] = {
            "device_type": self._device.type,
            "device_name": str(self._device),
        }
        
        if self._device.type == "cuda" and self.cuda_available:
            device_index = self._device.index or 0
            capabilities.update({
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_current_device": torch.cuda.current_device(),
                "cuda_device_name": torch.cuda.get_device_name(device_index),
                "cuda_memory_total": torch.cuda.get_device_properties(device_index).total_memory,
                "cuda_compute_capability": torch.cuda.get_device_properties(device_index).major,
            })
        elif self._device.type == "mps" and self.mps_available:
            capabilities.update({
                "mps_available": True,
            })
        elif self._device.type == "cpu":
            capabilities.update({
                "cpu_count": torch.get_num_threads(),
            })
            
        return capabilities
    
    @property
    def current_device(self) -> torch.device:
        """Get the currently configured device.
        
        Returns:
            torch.device: Current device configuration
        """
        return self._device
        
    def to(self, device: Union[str, torch.device]) -> 'DeviceManager':
        """Change the current device configuration.
        
        Args:
            device: New device to configure. Can be string or torch.device.
            
        Returns:
            DeviceManager: Self for method chaining
            
        Raises:
            RuntimeError: If specified device is not available
            
        Examples:
            >>> manager.to('cuda')
            >>> manager.to(torch.device('cpu'))
        """
        new_device = torch.device(device)
        
        # Validate device availability
        if new_device.type == "cuda" and not self.cuda_available:
            raise RuntimeError("CUDA device requested but CUDA is not available")
        elif new_device.type == "mps" and not self.mps_available:
            raise RuntimeError("MPS device requested but MPS is not available")
            
        self._device = new_device
        self.device_capabilities = self._get_device_capabilities()
        logger.debug(f"DeviceManager switched to device: {self._device}")
        
        return self
        
    def get_optimal_device(self) -> torch.device:
        """Get the optimal device for the current system.
        
        Returns:
            torch.device: Optimal device considering hardware availability
        """
        return self._auto_detect_device()
        
    def is_cuda_available(self) -> bool:
        """Check if CUDA is available.
        
        Returns:
            bool: True if CUDA is available
        """
        return self.cuda_available
        
    def is_mps_available(self) -> bool:
        """Check if MPS (Apple Silicon GPU) is available.
        
        Returns:
            bool: True if MPS is available
        """
        return self.mps_available
        
    def get_memory_info(self) -> Dict[str, Any]:
        """Get memory information for the current device.
        
        Returns:
            Dict[str, Any]: Memory information including total, allocated, cached, etc.
        """
        memory_info: Dict[str, Any] = {"device": str(self._device)}
        
        if self._device.type == "cuda" and self.cuda_available:
            device_index = self._device.index or 0
            memory_info.update({
                "total_memory": torch.cuda.get_device_properties(device_index).total_memory,
                "allocated_memory": torch.cuda.memory_allocated(device_index),
                "cached_memory": torch.cuda.memory_reserved(device_index),
                "free_memory": torch.cuda.get_device_properties(device_index).total_memory - torch.cuda.memory_allocated(device_index),
            })
        else:
            memory_info.update({
                "note": f"Memory info not available for {self._device.type} device"
            })
            
        return memory_info
        
    def transfer_tensor(self, tensor: torch.Tensor, non_blocking: bool = False) -> torch.Tensor:
        """Transfer tensor to the managed device.
        
        Args:
            tensor: Tensor to transfer
            non_blocking: Whether to use non-blocking transfer
            
        Returns:
            torch.Tensor: Tensor on the managed device
        """
        if tensor.device == self._device:
            return tensor
            
        return tensor.to(self._device, non_blocking=non_blocking)
        
    def synchronize(self) -> None:
        """Synchronize the current device if applicable.
        
        For CUDA devices, this calls torch.cuda.synchronize().
        For other devices, this is a no-op.
        """
        if self._device.type == "cuda" and self.cuda_available:
            torch.cuda.synchronize(self._device.index)
            
    def clear_cache(self) -> None:
        """Clear device cache if applicable.
        
        For CUDA devices, this calls torch.cuda.empty_cache().
        For other devices, this is a no-op.
        """
        if self._device.type == "cuda" and self.cuda_available:
            torch.cuda.empty_cache()
            logger.debug("Cleared CUDA cache")
            
    def get_device_stats(self) -> Dict[str, Any]:
        """Get comprehensive device statistics.
        
        Returns:
            Dict[str, Any]: Device statistics including capabilities, memory, status
        """
        stats: Dict[str, Any] = {
            "current_device": str(self._device),
            "capabilities": self.device_capabilities,
            "memory_info": self.get_memory_info(),
            "cuda_available": self.cuda_available,
            "mps_available": self.mps_available,
        }
        
        return stats
        
    def __repr__(self) -> str:
        """String representation of DeviceManager.
        
        Returns:
            str: Human-readable representation
        """
        return f"DeviceManager(device={self._device}, cuda={self.cuda_available}, mps={self.mps_available})"
        
    def __str__(self) -> str:
        """String conversion of DeviceManager.
        
        Returns:
            str: Device name
        """
        return str(self._device)


# Global device manager instance for shared access
_global_device_manager = None


def get_global_device_manager() -> DeviceManager:
    """Get the global DeviceManager instance.
    
    Returns:
        DeviceManager: Global device manager singleton
        
    Examples:
        >>> manager = get_global_device_manager()
        >>> manager.current_device
        device(type='cuda', index=0)
    """
    global _global_device_manager
    if _global_device_manager is None:
        _global_device_manager = DeviceManager()
    return _global_device_manager


def set_global_device(device: Union[str, torch.device]) -> DeviceManager:
    """Set the global device configuration.
    
    Args:
        device: Device to set as global default
        
    Returns:
        DeviceManager: Updated global device manager
        
    Examples:
        >>> set_global_device('cuda')
        >>> set_global_device(torch.device('cpu'))
    """
    global _global_device_manager
    if _global_device_manager is None:
        _global_device_manager = DeviceManager(device)
    else:
        _global_device_manager.to(device)
    return _global_device_manager
