from typing import Any, Dict
import torchvision.transforms.v2 as T


class NaNtoNum(T.Transform):
    """Replace NaNs with zeros in the input tensor. Subclasses torchvision.transforms.Transform.

    Attributes:
        params (Dict[str, Any]): Parameters for the transformation. Defaults to {}, see https://pytorch.org/docs/stable/generated/torch.nan_to_num.html for details.

    Methods:
        _transform: Transform the input.
    """

    def __init__(self, params: Dict[str, Any]):
        """Initialize the NaN to number transformation.

        Args:
            params (Dict[str, Any]): Parameters for the transformation. Defaults to {}, see https://pytorch.org/docs/stable/generated/torch.nan_to_num.html for details.
        """
        super().__init__()
        self.params = params

    def _transform(self, x: Any, params: Dict[str, Any]) -> Any:
        """Transform the input."""
        return x.nan_to_num(**self.params)
