from cellpose.dynamics import masks_to_flows_gpu_3d, masks_to_flows
from cellpose.dynamics import masks_to_flows_gpu as masks_to_flows_gpu_2d
import torch


class CellposeFlow:
    def __init__(self, ndim: int, device: str | None = None):
        self.ndim = ndim
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        _device = torch.device(device)
        if device == "cuda" or device == "mps":
            if ndim == 3:
                flows_func = lambda x: masks_to_flows_gpu_3d(x, device=_device)
            elif ndim == 2:
                flows_func = lambda x: masks_to_flows_gpu_2d(x, device=_device)
            else:
                raise ValueError(f"Unsupported dimension {ndim}")
        else:
            flows_func = lambda x: masks_to_flows(x, device=_device)
        self.flows_func = flows_func
        self.device = _device

    def __call__(self, masks):
        return self.flows_func(masks)