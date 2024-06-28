import torch
from py_distance_transforms import transform_cuda, transform


class DistanceTransform(torch.nn.Module):
    # Compute the distance transform of the input
    def __init__(self):
        super(DistanceTransform, self).__init__()

    def forward(self, x: torch.Tensor):
        if x.device.type == "cuda":
            return transform_cuda(x)
        else:
            return transform(x)
