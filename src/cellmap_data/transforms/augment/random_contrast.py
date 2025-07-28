from typing import Sequence
import torch
from cellmap_data.utils import torch_max_value
from ...utils.error_handling import (
    ValidationError,
    ErrorMessages,
    validate_tensor_finite,
    validate_positive_ratio,
)


class RandomContrast(torch.nn.Module):
    """Apply random contrast adjustment to input tensors for data augmentation.

    This transformation randomly adjusts the contrast of input tensors by applying
    a linear transformation around the mean value. The contrast factor is randomly
    sampled from a specified range, providing stochastic augmentation suitable
    for training neural networks.

    The transformation applies the formula: output = ratio * input + (1 - ratio) * mean,
    where ratio is randomly sampled and mean is the spatial mean of the input.
    Includes robust error handling for numerical stability and input validation.

    Inherits from torch.nn.Module for PyTorch compatibility and device management.

    Parameters
    ----------
    contrast_range : sequence of float, optional
        Range [min, max] for random contrast factor sampling, by default (0.5, 1.5).
        Values < 1.0 reduce contrast, values > 1.0 increase contrast.
        Must contain exactly two values with min < max.

    Attributes
    ----------
    contrast_range : tuple of float
        Configured range for contrast factor sampling.

    Methods
    -------
    forward(x)
        Apply random contrast adjustment to input tensor.

    Raises
    ------
    ValidationError
        If input tensor contains invalid values or contrast transformation fails.

    Examples
    --------
    Basic contrast augmentation:

    >>> import torch
    >>> from cellmap_data.transforms.augment import RandomContrast
    >>> contrast_transform = RandomContrast(contrast_range=(0.8, 1.2))
    >>> x = torch.rand(2, 3, 64, 64)
    >>> augmented = contrast_transform(x)
    >>> print(f"Input range: [{x.min():.3f}, {x.max():.3f}]")
    >>> print(f"Output range: [{augmented.min():.3f}, {augmented.max():.3f}]")

    Strong contrast variation:

    >>> strong_contrast = RandomContrast(contrast_range=(0.2, 2.0))
    >>> high_variation = strong_contrast(x)

    See Also
    --------
    RandomGamma : Gamma correction-based contrast adjustment
    GaussianNoise : Additive noise augmentation
    """

    def __init__(self, contrast_range: Sequence[float] = (0.5, 1.5)) -> None:
        """Initialize random contrast transform with specified range.

        Parameters
        ----------
        contrast_range : sequence of float, optional
            Two-element sequence [min, max] defining the range for random
            contrast factor sampling, by default (0.5, 1.5).
            Values should be positive with min < max.

        Notes
        -----
        Contrast factors less than 1.0 reduce contrast (making the image more uniform),
        while factors greater than 1.0 increase contrast (enhancing differences).
        """
        super().__init__()
        self.contrast_range = contrast_range

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random contrast adjustment to input tensor.

        Randomly samples a contrast factor from the configured range and applies
        linear contrast adjustment around the input's spatial mean. Includes
        comprehensive validation and numerical stability handling.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor for contrast adjustment. Can be any shape and numeric dtype.
            Must contain finite values for stable computation.

        Returns
        -------
        torch.Tensor
            Contrast-adjusted tensor with same shape and dtype as input.
            Values are clamped to valid range for the input dtype.

        Raises
        ------
        ValidationError
            If input tensor contains non-finite values (NaN, inf).
            If computed contrast ratio is not positive.
            If transformation produces invalid output values.

        Examples
        --------
        >>> import torch
        >>> contrast_transform = RandomContrast(contrast_range=(0.8, 1.2))
        >>> x = torch.tensor([[0.2, 0.8], [0.4, 0.6]], dtype=torch.float32)
        >>> result = contrast_transform.forward(x)
        >>> print(result.shape == x.shape)  # Shape preserved
        True
        >>> print(result.dtype == x.dtype)  # Dtype preserved
        True

        Notes
        -----
        The transformation formula is: output = ratio * input + (1 - ratio) * mean
        where ratio is the randomly sampled contrast factor and mean is the
        spatial mean of the input tensor.
        """
        # Handle edge cases that could cause NaNs
        if x.numel() == 0:
            return x  # Return empty tensor as-is

        # Check for invalid input values
        validate_tensor_finite(x, "input tensor")

        # Generate contrast ratio
        ratio = float(
            torch.rand(1) * (self.contrast_range[1] - self.contrast_range[0])
            + self.contrast_range[0]
        )

        # Validate contrast ratio
        validate_positive_ratio(ratio, "contrast")

        # Calculate mean with numerical stability
        x_mean = x.mean(dim=0, keepdim=True)

        # Check if mean calculation produced NaN (shouldn't happen with valid input)
        if torch.any(torch.isnan(x_mean)):
            raise ValidationError(
                ErrorMessages.DATA_CORRUPTED,
                details="Mean calculation produced NaN - this indicates corrupted input data",
            )

        # Apply contrast transformation
        bound = torch_max_value(x.dtype)
        result = ratio * x + (1.0 - ratio) * x_mean

        # Clamp to valid range for the dtype
        result = result.clamp(0, bound)

        # Convert back to original dtype
        result = result.to(x.dtype)

        # Final validation - this should never trigger with proper implementation
        if torch.any(torch.isnan(result)):
            raise ValidationError(
                ErrorMessages.DATA_CORRUPTED,
                details=f"Contrast transformation produced NaN values. "
                f"Input stats: min={x.min()}, max={x.max()}, mean={x_mean.mean()}, "
                f"contrast_ratio={ratio}, dtype={x.dtype}",
            )

        return result
