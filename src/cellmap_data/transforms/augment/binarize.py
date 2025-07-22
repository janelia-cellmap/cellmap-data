from typing import Any, Dict
import torchvision.transforms.v2 as T
import torch


class Binarize(T.Transform):
    """Binarize the input tensor. Subclasses torchvision.transforms.Transform.

    Methods:
        _transform: Transform the input.
    """

    def __init__(self, threshold=0) -> None:
        """Initialize the normalization transformation."""
        super().__init__()
        self.threshold = threshold

    def _transform(self, x: Any, params: Dict[str, Any] | None = None) -> Any:
        """Transform the input."""
        out = (x > self.threshold).to(x.dtype)
        hasnans = x.isnan().any()
        if not torch.is_floating_point(out) and hasnans:
            out = out.to(torch.float32)
        if hasnans:
            out[x.isnan()] = torch.nan
        return out

    def __repr__(self) -> str:
        """Return a string representation of the transformation."""
        return f"{self.__class__.__name__}(threshold={self.threshold})"

    def transform(self, x: Any, params: Dict[str, Any] | None = None) -> Any:
        """Transform the input."""
        return self._transform(x, params)
