from typing import Any, Dict
import torchvision.transforms.v2 as T


class NaNtoNum(T.Transform):
    """Replace NaN values with finite numbers in input tensors.

    This transformation provides a convenient wrapper around PyTorch's nan_to_num
    function for use in torchvision transformation pipelines. It replaces NaN values
    with zeros (or other specified values), positive infinity with large finite numbers,
    and negative infinity with small finite numbers.

    Args:
        params: Parameters passed to torch.nan_to_num function. See PyTorch documentation
            for torch.nan_to_num for complete parameter details.
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        """Initialize the NaN to number transformation.

        Args:
            params: Parameters for the torch.nan_to_num transformation. Common parameters:
                - 'nan': replacement value for NaN (default: 0.0)
                - 'posinf': replacement value for positive infinity (default: large finite number)
                - 'neginf': replacement value for negative infinity (default: small finite number)
                See torch.nan_to_num documentation for complete parameter details.
        """
        super().__init__()
        self.params = params

    def _transform(self, inpt: Any, params: Dict[str, Any] | None = None) -> Any:
        """Apply nan_to_num transformation to input tensor.

        Args:
            inpt: Input tensor or other data structure to process.
            params: Additional parameters (not used, maintained for interface compatibility).

        Returns:
            Input with NaN values replaced according to configured parameters.
        """
        return inpt.nan_to_num(**self.params)

    def __repr__(self) -> str:
        """Return a string representation of the transformation."""
        return f"{self.__class__.__name__}(params={self.params})"

    def transform(self, inpt: Any, params: Dict[str, Any] | None = None) -> Any:
        """Transform the input tensor by replacing invalid values.

        Public interface for applying the nan_to_num transformation.

        Args:
            inpt: Input tensor or data structure to transform.
            params: Additional parameters (not used, maintained for interface compatibility).

        Returns:
            Transformed input with NaN and infinity values replaced.
        """
        return self._transform(inpt, params)
