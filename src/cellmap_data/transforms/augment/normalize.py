from typing import Any, Dict
import torch
import torchvision.transforms.v2 as T


class Normalize(T.Transform):
    """Normalize tensor values with configurable shift and scale parameters.

    Applies a linear transformation (shift + scale) to input tensors and converts
    the result to float dtype. Useful for normalizing uint8 image data to [0, 1]
    range or applying standardization with custom parameters.

    Args:
        shift: Value to add before scaling. Defaults to 0.
        scale: Multiplicative scaling factor. Defaults to 1/255.

    Examples:
        >>> normalize = Normalize(shift=0, scale=1/255)
        >>> x = torch.tensor([[0, 255]], dtype=torch.uint8)
        >>> result = normalize.transform(x, {})
        >>> print(result)
        tensor([[0.0000, 1.0000]])
    """

    def __init__(self, shift=0, scale=1 / 255) -> None:
        """Initialize the normalization transformation.

        Args:
            shift: Shift values before scaling. Defaults to 0.
            scale: Scale values after shifting. Defaults to 1/255.
        """
        super().__init__()
        self.shift = shift
        self.scale = scale

    def _transform(self, x: Any, params: Dict[str, Any] | None = None) -> Any:
        """Apply normalization transformation to input tensor.

        Args:
            x: Input tensor to normalize.
            params: Additional parameters (unused). Defaults to None.

        Returns:
            Normalized tensor with applied shift and scale.
        """
        return (x + self.shift) * self.scale

    def __repr__(self) -> str:
        """Return string representation of the transformation.

        Returns:
            Class name for debugging and logging purposes.
        """
        return self.__class__.__name__

    def transform(self, x: Any, params: Dict[str, Any] | None = None) -> Any:
        """Apply normalization with automatic float conversion.

        Args:
            x: Input tensor to normalize and convert to float.
            params: Additional parameters (unused). Defaults to None.

        Returns:
            Normalized float tensor.
        """
        return self._transform(x.to(torch.float), params)
