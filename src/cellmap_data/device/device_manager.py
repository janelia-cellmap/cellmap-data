import torch


class DeviceManager:
    """
    Centralized device and memory management for cellmap-data.
    Handles device selection, tensor transfer, and memory pooling.
    Detects framework context (Accelerate, PyTorch Lightning) and disables redundant transfers.
    """

    def __init__(self, device=None):
        self.device = device if device is not None else torch.device("cpu")
        self.framework = self.detect_framework()
        self.memory_pool = None  # Placeholder for future pooling logic

    def select_device(self, preferred=None):
        """Select device: cuda if available, else cpu. Optionally override with preferred."""
        if preferred is not None:
            return torch.device(preferred)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def to_device(self, tensor, device=None, non_blocking=True):
        """Safely transfer tensor to device, skipping if managed by external framework."""
        if self.framework in ("accelerate", "pytorch_lightning", "deepspeed"):
            # Assume external framework manages device transfers
            return tensor

        target_device = device or self.device
        if hasattr(tensor, "to"):
            return tensor.to(target_device, non_blocking=non_blocking)
        return tensor

    def pool_tensor(self, tensor):
        """Memory pooling for frequent allocations. Uses torch's caching allocator for CUDA, and a simple cache for CPU."""
        # CUDA: rely on torch's built-in caching allocator
        if tensor.device.type == "cuda":
            # Optionally, could use torch.cuda.memory_pool() for advanced pooling
            return tensor  # torch handles pooling automatically
        # CPU: implement a simple cache for repeated shapes/dtypes
        key = (tuple(tensor.shape), tensor.dtype)
        if not hasattr(self, "_cpu_pool"):
            self._cpu_pool = {}
        pool = self._cpu_pool
        if key in pool:
            # Reuse cached tensor if available
            cached = pool[key]
            if cached.shape == tensor.shape and cached.dtype == tensor.dtype:
                return cached
        # Store new tensor in pool
        pool[key] = tensor
        return tensor

    def detect_framework(self):
        """Detect if running under Accelerate, PyTorch Lightning, or DeepSpeed by checking sys.modules only."""
        import sys

        for fw in ("accelerate", "pytorch_lightning", "deepspeed"):
            if fw in sys.modules:
                return fw
        return None
