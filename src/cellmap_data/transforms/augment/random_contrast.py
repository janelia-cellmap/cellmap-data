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

    Randomly adjusts the contrast of input tensors by applying a linear transformation
    around the mean value. The contrast factor is randomly sampled from a specified
    range, providing stochastic augmentation suitable for training neural networks.

    Args:
        contrast_range: Range [min, max] for random contrast factor sampling.
            Defaults to (0.5, 1.5). Values < 1.0 reduce contrast, values > 1.0
            increase contrast.

    Examples:
        >>> contrast_transform = RandomContrast(contrast_range=(0.8, 1.2))
        >>> x = torch.rand(2, 3, 64, 64)
        >>> augmented = contrast_transform(x)
    """

    def __init__(self, contrast_range: Sequence[float] = (0.5, 1.5)) -> None:
        """Initialize random contrast transform with specified range.

        Args:
            contrast_range: Two-element sequence [min, max] defining the range for
                random contrast factor sampling. Defaults to (0.5, 1.5).
        """
        super().__init__()
        self.contrast_range = contrast_range

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random contrast adjustment to input tensor.

        Args:
            x: Input tensor for contrast adjustment. Must contain finite values.

        Returns:
            Contrast-adjusted tensor with same shape and dtype as input.

        Raises:
            ValidationError: If input tensor contains non-finite values.
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
