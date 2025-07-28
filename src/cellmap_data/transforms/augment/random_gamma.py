from typing import Sequence
import torch
from torchvision.transforms.v2 import ToDtype

from ...utils.logging_config import get_logger

logger = get_logger("transforms.random_gamma")


class RandomGamma(torch.nn.Module):
    """Apply random gamma correction to input tensors for data augmentation.

    Applies gamma correction with a random gamma value sampled from a specified range.
    Gamma correction adjusts luminance and contrast by applying a power-law
    transformation: output = input^gamma.

    Args:
        gamma_range: Range [min, max] for random gamma value sampling.
            Defaults to (0.5, 1.5). Values < 1.0 brighten, values > 1.0 darken.

    Examples:
        >>> gamma_transform = RandomGamma(gamma_range=(0.8, 1.2))
        >>> x = torch.rand(2, 3, 64, 64)
        >>> corrected = gamma_transform(x)
    """

    def __init__(self, gamma_range: Sequence[float] = (0.5, 1.5)) -> None:
        """Initialize random gamma transform with specified range.

        Args:
            gamma_range: Two-element sequence [min, max] defining the range for
                random gamma value sampling. Defaults to (0.5, 1.5).
        """
        super().__init__()
        self.gamma_range = gamma_range

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random gamma correction to input tensor.

        Args:
            x: Input tensor for gamma correction. Can be any numeric dtype and shape.

        Returns:
            Gamma-corrected tensor with values clamped to [0, 1] range.
        """
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
