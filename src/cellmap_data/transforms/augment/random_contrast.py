from typing import Sequence
import torch
from cellmap_data.utils import torch_max_value


class RandomContrast(torch.nn.Module):
    """
    Randomly change the contrast of the input.

    Attributes:
        contrast_range (tuple): Contrast range.

    Methods:
        forward: Forward pass.
    """

    def __init__(self, contrast_range: Sequence[float] = (0.5, 1.5)) -> None:
        """
        Initialize the random contrast.

        Args:
            contrast_range (tuple, optional): Contrast range. Defaults to (0.5, 1.5).
        """
        super().__init__()
        self.contrast_range = contrast_range

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
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
        torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0, out=result)
        return result
