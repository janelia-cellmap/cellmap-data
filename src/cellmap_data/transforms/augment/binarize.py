from typing import Any, Dict
import torchvision.transforms.v2 as T


class Binarize(T.Transform):
    """Binarize the input tensor. Subclasses torchvision.transforms.Transform.

    Methods:
        _transform: Transform the input.
    """

    def __init__(self, threshold=0) -> None:
        """Initialize the normalization transformation."""
        super().__init__()
        self.threshold = threshold

    def _transform(self, x: Any, params: Dict[str, Any]) -> Any:
        """Transform the input."""
        return (x > self.threshold).to(x.dtype)
