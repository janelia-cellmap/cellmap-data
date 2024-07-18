import torch
from cellmap_data.utils import torch_max_value


class RandomContrast(torch.nn.Module):
    # Randomly adjust the contrast of the input
    def __init__(self, contrast_range=(0.5, 1.5)):
        super(RandomContrast, self).__init__()
        self.contrast_range = contrast_range

    def forward(self, x: torch.Tensor):
        ratio = float(
            torch.rand(1) * (self.contrast_range[1] - self.contrast_range[0])
            + self.contrast_range[0]
        )
        bound = torch_max_value(x.dtype)
        result = (
            (ratio * x + (1.0 - ratio) * x.mean(dim=0, keepdim=True))
            .clamp(0, bound)
            .to(x.dtype)
        )
        # Hack to avoid NaNs
        torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0, out=x)
        return result
