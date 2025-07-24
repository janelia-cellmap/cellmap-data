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
        """Forward pass with proper numerical stability handling."""
        # Handle edge cases that could cause NaNs
        if x.numel() == 0:
            return x  # Return empty tensor as-is

        # Check for invalid input values
        if torch.any(torch.isnan(x)):
            raise ValueError("Input tensor contains NaN values")

        if torch.any(torch.isinf(x)):
            raise ValueError("Input tensor contains infinite values")

        # Generate contrast ratio
        ratio = float(
            torch.rand(1) * (self.contrast_range[1] - self.contrast_range[0])
            + self.contrast_range[0]
        )

        # Validate contrast ratio
        if ratio <= 0:
            raise ValueError(f"Invalid contrast ratio: {ratio}. Must be positive.")

        # Calculate mean with numerical stability
        x_mean = x.mean(dim=0, keepdim=True)

        # Check if mean calculation produced NaN (shouldn't happen with valid input)
        if torch.any(torch.isnan(x_mean)):
            raise RuntimeError(
                "Mean calculation produced NaN - this indicates corrupted input data"
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
            raise RuntimeError(
                f"Contrast transformation produced NaN values. "
                f"Input stats: min={x.min()}, max={x.max()}, mean={x_mean.mean()}, "
                f"contrast_ratio={ratio}, dtype={x.dtype}"
            )

        return result
