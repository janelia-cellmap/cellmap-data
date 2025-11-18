"""
Device Management Module for CellMap Dataset Architecture.

Centralized device detection, assignment, and management for consistent device
handling across all dataset components. Extracted from monolithic CellMapDataset
to improve maintainability and reduce code duplication.
"""

import os
from typing import Optional, Union
import torch
from ..utils.logging_config import get_logger

logger = get_logger("device_manager")


class DeviceManager:
    """Centralized device management for CellMap dataset components.

    Handles device detection, assignment, and tensor operations in a consistent
    manner across all dataset classes. Reduces code duplication and provides
    optimized device selection strategies.

    Attributes:
        device: Current torch.device instance
        device_type: String representation of device type ('cuda', 'mps', 'cpu')
        is_cuda_available: Whether CUDA is available and functional
        is_mps_available: Whether MPS (Apple Silicon) is available and functional
        device_count: Number of available devices for the current device type

    Examples:
        >>> device_manager = DeviceManager()
        >>> device_manager.device
        device(type='cuda', index=0)  # If CUDA available

        >>> device_manager = DeviceManager(device='cpu')
        >>> device_manager.device_type
        'cpu'

        >>> # Move tensors to managed device
        >>> tensor = torch.randn(10, 10)
        >>> device_tensor = device_manager.to_device(tensor)
    """

    def __init__(self, device: Optional[Union[str, torch.device]] = None):
        """Initialize device manager with automatic or specified device.

        Args:
            device: Specific device to use. If None, automatically selects
                   the best available device (CUDA > MPS > CPU).

        Raises:
            RuntimeError: If specified device is not available.
            ValueError: If device specification is invalid.
        """
        self._device: torch.device
        self._device_type: str
        self._cuda_available: bool
        self._mps_available: bool
        self._device_count: int

        # Initialize device availability flags
        self._check_device_availability()

        # Set device based on input or auto-detection
        if device is not None:
            self._set_specific_device(device)
        else:
            self._auto_select_device()

        logger.info(f"DeviceManager initialized with device: {self._device}")

    def _check_device_availability(self) -> None:
        """Check availability of different device types."""
        try:
            self._cuda_available = torch.cuda.is_available()
            if self._cuda_available:
                # Test basic CUDA functionality
                try:
                    torch.cuda.device_count()
                    torch.cuda.get_device_name(0)
                except Exception:
                    self._cuda_available = False
                    logger.warning("CUDA detected but not functional, falling back")
        except Exception:
            self._cuda_available = False

        try:
            self._mps_available = (
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            )
            if self._mps_available:
                # Test basic MPS functionality
                try:
                    torch.tensor([1.0]).to("mps")
                except Exception:
                    self._mps_available = False
                    logger.warning("MPS detected but not functional, falling back")
        except Exception:
            self._mps_available = False

        logger.debug(
            f"Device availability: CUDA={self._cuda_available}, MPS={self._mps_available}"
        )

    def _auto_select_device(self) -> None:
        """Automatically select the best available device."""
        if self._cuda_available:
            self._device = torch.device("cuda")
            self._device_type = "cuda"
            self._device_count = torch.cuda.device_count()
            logger.debug(f"Auto-selected CUDA device with {self._device_count} GPUs")
        elif self._mps_available:
            self._device = torch.device("mps")
            self._device_type = "mps"
            self._device_count = 1  # MPS typically has 1 device
            logger.debug("Auto-selected MPS device")
        else:
            self._device = torch.device("cpu")
            self._device_type = "cpu"
            self._device_count = os.cpu_count() or 1
            logger.debug(f"Auto-selected CPU device with {self._device_count} cores")

    def _set_specific_device(self, device: Union[str, torch.device]) -> None:
        """Set a specific device with validation.

        Args:
            device: Device specification to validate and set.

        Raises:
            RuntimeError: If specified device is not available.
            ValueError: If device specification is invalid.
        """
        if isinstance(device, str):
            device_str = device.lower()
            if device_str == "cuda":
                if not self._cuda_available:
                    raise RuntimeError("CUDA device requested but not available")
                self._device = torch.device("cuda")
                self._device_type = "cuda"
                self._device_count = torch.cuda.device_count()
            elif device_str == "mps":
                if not self._mps_available:
                    raise RuntimeError("MPS device requested but not available")
                self._device = torch.device("mps")
                self._device_type = "mps"
                self._device_count = 1
            elif device_str == "cpu":
                self._device = torch.device("cpu")
                self._device_type = "cpu"
                self._device_count = os.cpu_count() or 1
            elif device_str.startswith("cuda:"):
                if not self._cuda_available:
                    raise RuntimeError("CUDA device requested but not available")
                try:
                    device_idx = int(device_str.split(":")[1])
                    if device_idx >= torch.cuda.device_count():
                        raise RuntimeError(f"CUDA device {device_idx} not available")
                    self._device = torch.device(device_str)
                    self._device_type = "cuda"
                    self._device_count = torch.cuda.device_count()
                except (ValueError, IndexError):
                    raise ValueError(f"Invalid CUDA device specification: {device}")
            else:
                raise ValueError(f"Unknown device type: {device}")
        elif isinstance(device, torch.device):
            device_type = device.type.lower()
            if device_type == "cuda" and not self._cuda_available:
                raise RuntimeError("CUDA device requested but not available")
            elif device_type == "mps" and not self._mps_available:
                raise RuntimeError("MPS device requested but not available")
            self._device = device
            self._device_type = device_type
            if device_type == "cuda":
                self._device_count = torch.cuda.device_count()
            elif device_type == "mps":
                self._device_count = 1
            else:
                self._device_count = os.cpu_count() or 1
        else:
            raise ValueError(
                f"Device must be string or torch.device, got {type(device)}"
            )

        logger.debug(f"Set specific device: {self._device}")

    @property
    def device(self) -> torch.device:
        """Current torch.device instance."""
        return self._device

    @property
    def device_type(self) -> str:
        """String representation of device type."""
        return self._device_type

    @property
    def is_cuda_available(self) -> bool:
        """Whether CUDA is available and functional."""
        return self._cuda_available

    @property
    def is_mps_available(self) -> bool:
        """Whether MPS is available and functional."""
        return self._mps_available

    @property
    def device_count(self) -> int:
        """Number of available devices for current device type."""
        return self._device_count

    @property
    def is_gpu_device(self) -> bool:
        """Whether current device is a GPU (CUDA or MPS)."""
        return self._device_type in ("cuda", "mps")

    def to_device(
        self, tensor: torch.Tensor, non_blocking: bool = False
    ) -> torch.Tensor:
        """Move tensor to managed device.

        Args:
            tensor: Tensor to move to device.
            non_blocking: Whether to use non-blocking transfer (GPU only).

        Returns:
            Tensor moved to managed device.
        """
        if tensor.device == self._device:
            return tensor

        # Use non-blocking transfer only for GPU devices
        use_non_blocking = non_blocking and self.is_gpu_device

        try:
            return tensor.to(self._device, non_blocking=use_non_blocking)
        except Exception as e:
            logger.warning(
                f"Failed to move tensor to {self._device}, keeping on {tensor.device}: {e}"
            )
            return tensor

    def empty_cache(self) -> None:
        """Clear GPU memory cache if using CUDA."""
        if self._device_type == "cuda":
            try:
                torch.cuda.empty_cache()
                logger.debug("CUDA cache cleared")
            except Exception as e:
                logger.warning(f"Failed to clear CUDA cache: {e}")
        elif self._device_type == "mps":
            try:
                torch.mps.empty_cache()
                logger.debug("MPS cache cleared")
            except Exception as e:
                logger.warning(f"Failed to clear MPS cache: {e}")

    def synchronize(self) -> None:
        """Synchronize device operations."""
        if self._device_type == "cuda":
            try:
                torch.cuda.synchronize(self._device)
                logger.debug("CUDA synchronized")
            except Exception as e:
                logger.warning(f"Failed to synchronize CUDA: {e}")
        elif self._device_type == "mps":
            try:
                torch.mps.synchronize()
                logger.debug("MPS synchronized")
            except Exception as e:
                logger.warning(f"Failed to synchronize MPS: {e}")

    def get_memory_info(self) -> dict:
        """Get device memory information.

        Returns:
            Dictionary with memory information or empty dict if not available.
        """
        if self._device_type == "cuda":
            try:
                allocated = torch.cuda.memory_allocated(self._device)
                reserved = torch.cuda.memory_reserved(self._device)
                return {
                    "device": str(self._device),
                    "allocated_mb": allocated // (1024 * 1024),
                    "reserved_mb": reserved // (1024 * 1024),
                    "available_mb": (reserved - allocated) // (1024 * 1024),
                }
            except Exception:
                return {}
        elif self._device_type == "mps":
            try:
                allocated = torch.mps.current_allocated_memory()
                return {
                    "device": str(self._device),
                    "allocated_mb": allocated // (1024 * 1024),
                }
            except Exception:
                return {}
        else:
            return {"device": str(self._device), "type": "cpu"}

    def optimize_for_memory(self) -> None:
        """Apply memory optimization settings for current device."""
        if self._device_type == "cuda":
            try:
                # Set memory fraction if environment variable specified
                memory_fraction = os.environ.get("CUDA_MEMORY_FRACTION")
                if memory_fraction:
                    torch.cuda.set_per_process_memory_fraction(float(memory_fraction))
                logger.debug("Applied CUDA memory optimizations")
            except Exception as e:
                logger.warning(f"Failed to apply CUDA memory optimizations: {e}")

    def __str__(self) -> str:
        """String representation of device manager."""
        return f"DeviceManager(device={self._device}, type={self._device_type}, count={self._device_count})"

    def __repr__(self) -> str:
        """Detailed representation of device manager."""
        return (
            f"DeviceManager(device={self._device}, type={self._device_type}, "
            f"count={self._device_count}, cuda_available={self._cuda_available}, "
            f"mps_available={self._mps_available})"
        )


# Global device manager instance for shared use
_global_device_manager = None


def get_global_device_manager() -> DeviceManager:
    """Get global device manager instance.

    Returns:
        Shared DeviceManager instance for consistent device handling.
    """
    global _global_device_manager
    if _global_device_manager is None:
        _global_device_manager = DeviceManager()
    return _global_device_manager


def set_global_device(device: Union[str, torch.device]) -> None:
    """Set global device for all CellMap components.

    Args:
        device: Device to set globally.
    """
    global _global_device_manager
    _global_device_manager = DeviceManager(device=device)
    logger.info(f"Global device set to: {device}")
