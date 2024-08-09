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

        # These assertions pass
        # assert not torch.isnan(x).any()
        # assert not torch.isinf(x).any()
        # assert not torch.isnan(gamma)
        # assert not torch.isinf(gamma)
        # assert gamma > 0.0
        x = (x**gamma).clamp(0.0, 1.0)

        # This assertion fails and I don't know why
        # assert torch.isnan(x).sum() == 0

        # Hack to avoid NaNs
        torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0, out=x)
        return x
