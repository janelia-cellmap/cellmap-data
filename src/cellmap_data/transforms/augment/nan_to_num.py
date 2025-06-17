from typing import Any, Dict
import torchvision.transforms.v2 as T


class NaNtoNum(T.Transform):
    """Replace NaNs with zeros in the input tensor. Subclasses torchvision.transforms.Transform.

    Attributes:
        params (Dict[str, Any]): Parameters for the transformation. Defaults to {}, see https://pytorch.org/docs/stable/generated/torch.nan_to_num.html for details.

    Methods:
        _transform: Transform the input.
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        """Initialize the NaN to number transformation.

        Args:
            params (Dict[str, Any]): Parameters for the transformation. Defaults to {}, see https://pytorch.org/docs/stable/generated/torch.nan_to_num.html for details.
        """
        super().__init__()
        self.params = params

    def _transform(self, x: Any, params: Dict[str, Any] | None = None) -> Any:
        """Transform the input."""
        return x.nan_to_num(**self.params)

    def __repr__(self) -> str:
        """Return a string representation of the transformation."""
        return f"{self.__class__.__name__}(params={self.params})"

    def transform(self, inpt: Any, params: Dict[str, Any] | None = None) -> Any:
        """Transform the input."""
        return self._transform(inpt, params)
