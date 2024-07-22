from typing import Any, Dict
import torchvision.transforms.v2 as T


class Normalize(T.Transform):
    """Normalize the input tensor. Subclasses torchvision.transforms.Transform.

    Methods:
        _transform: Transform the input.
    """

    def __init__(self):
        """Initialize the normalization transformation."""
        super().__init__()

    def _transform(self, x: Any, params: Dict[str, Any]) -> Any:
        """Transform the input."""
        min_val = x.nan_to_num().min()
        diff = x.nan_to_num().max() - min_val
        if diff == 0:
            return x
        else:
            return (x - min_val) / diff
