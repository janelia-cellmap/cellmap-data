from typing import Any, Dict
import torch
import torchvision.transforms.v2 as T


class Normalize(T.Transform):
    """Normalize the input tensor by given shift and scale, and convert to float. Subclasses torchvision.transforms.Transform.

    Methods:
        _transform: Transform the input.
    """

    def __init__(self, shift=0, scale=1 / 255) -> None:
        """Initialize the normalization transformation.
        Args:
            shift (float, optional): Shift values, before scaling. Defaults to 0.
            scale (float, optional): Scale values after shifting. Defaults to 1/255.

        This is helpful in normalizing the input to the range [0, 1], especially for data saved as uint8 which is scaled to [0, 255].

        Example:
            >>> import torch
            >>> from cellmap_data.transforms import Normalize
            >>> x = torch.tensor([[0, 255], [2, 3]], dtype=torch.uint8)
            >>> Normalize(shift=0, scale=1/255).transform(x, {})
            tensor([[0.0000, 1],
                    [0.0078, 0.0118]])

        """
        super().__init__()
        self.shift = shift
        self.scale = scale

    def _transform(self, x: Any, params: Dict[str, Any] | None = None) -> Any:
        """Transform the input."""
        return (x + self.shift) * self.scale

    def __repr__(self) -> str:
        """Return a string representation of the transformation."""
        return self.__class__.__name__

    def transform(self, x: Any, params: Dict[str, Any] | None = None) -> Any:
        """Transform the input."""
        return self._transform(x.to(torch.float), params)
