from typing import Sequence
import torch
from torchvision.transforms.v2 import ToDtype

import logging

logger = logging.getLogger(__name__)


class RandomGamma(torch.nn.Module):
    """
    Apply a random gamma augmentation to the input.

    Attributes:
        gamma_range (tuple): Gamma range.

    Methods:
        forward: Forward pass.
    """

    def __init__(self, gamma_range: Sequence[float] = (0.5, 1.5)) -> None:
        """
        Initialize the random gamma augmentation.

        Args:
            gamma_range (tuple, optional): Gamma range. Defaults to (0.5, 1.5).
        """
        super().__init__()
        self.gamma_range = gamma_range

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        gamma = torch.as_tensor(
            float(
                torch.rand(1) * (self.gamma_range[1] - self.gamma_range[0])
                + self.gamma_range[0]
            )
        )
        if not torch.is_floating_point(x):
            logger.debug("Input is not a floating point tensor. Converting to float32.")
            x = ToDtype(torch.float32, scale=True)(x)

        x = (x**gamma).clamp(0.0, 1.0)

        torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0, out=x)
        return x
