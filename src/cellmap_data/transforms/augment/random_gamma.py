from typing import Sequence
import torch
from torchvision.transforms.v2 import ToDtype

from ...utils.logging_config import get_logger

logger = get_logger("transforms.random_gamma")


class RandomGamma(torch.nn.Module):
    """Apply random gamma correction to input tensors for data augmentation.

    This transformation applies gamma correction with a random gamma value sampled
    from a specified range. Gamma correction adjusts the luminance and contrast
    characteristics of images by applying a power-law transformation: output = input^gamma.

    Gamma values < 1.0 brighten the image (expand dark regions), while gamma values
    > 1.0 darken the image (compress bright regions). The transformation automatically
    handles dtype conversion to floating point when needed and clamps output to [0, 1].

    Inherits from torch.nn.Module for PyTorch compatibility and parameter management.

    Parameters
    ----------
    gamma_range : sequence of float, optional
        Range [min, max] for random gamma value sampling, by default (0.5, 1.5).
        Must contain exactly two positive values with min < max.

    Attributes
    ----------
    gamma_range : tuple of float
        Configured range for gamma value sampling.

    Methods
    -------
    forward(x)
        Apply random gamma correction to input tensor.

    Examples
    --------
    Basic gamma augmentation:

    >>> import torch
    >>> from cellmap_data.transforms.augment import RandomGamma
    >>> gamma_transform = RandomGamma(gamma_range=(0.8, 1.2))
    >>> x = torch.rand(2, 3, 64, 64)
    >>> corrected = gamma_transform(x)
    >>> print(f"Input range: [{x.min():.3f}, {x.max():.3f}]")
    >>> print(f"Output range: [{corrected.min():.3f}, {corrected.max():.3f}]")

    Strong gamma variation for dramatic augmentation:

    >>> strong_gamma = RandomGamma(gamma_range=(0.3, 2.0))
    >>> dramatic = strong_gamma(x)

    See Also
    --------
    RandomContrast : Linear contrast adjustment
    GaussianNoise : Additive noise augmentation
    """

    def __init__(self, gamma_range: Sequence[float] = (0.5, 1.5)) -> None:
        """Initialize random gamma transform with specified range.

        Parameters
        ----------
        gamma_range : sequence of float, optional
            Two-element sequence [min, max] defining the range for random
            gamma value sampling, by default (0.5, 1.5).
            Values should be positive with min < max.

        Notes
        -----
        Gamma correction follows the power law: output = input^gamma.
        Values less than 1.0 brighten dark regions, values greater than 1.0
        darken bright regions, providing useful augmentation for various lighting conditions.
        """
        super().__init__()
        self.gamma_range = gamma_range

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random gamma correction to input tensor.

        Randomly samples a gamma value from the configured range and applies
        power-law gamma correction. Automatically converts to floating point
        if needed and ensures output is in valid [0, 1] range.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor for gamma correction. Can be any numeric dtype and shape.
            If not floating point, will be converted to float32 with scaling.

        Returns
        -------
        torch.Tensor
            Gamma-corrected tensor with values clamped to [0, 1] range.
            Output dtype is torch.float32 if input was converted, otherwise preserved.

        Examples
        --------
        >>> import torch
        >>> gamma_transform = RandomGamma(gamma_range=(0.8, 1.2))
        >>> x = torch.tensor([[0.1, 0.5], [0.9, 0.2]], dtype=torch.float32)
        >>> result = gamma_transform.forward(x)
        >>> print(result.shape == x.shape)  # Shape preserved
        True
        >>> print((result >= 0.0).all() and (result <= 1.0).all())  # Range [0,1]
        True

        Notes
        -----
        - Input values outside [0, 1] are clamped after gamma correction
        - NaN, positive infinity, and negative infinity values are replaced with 0.0
        - Non-floating point inputs are converted to float32 with appropriate scaling
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
