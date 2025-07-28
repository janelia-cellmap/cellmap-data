from typing import Any, Dict
import torchvision.transforms.v2 as T


class NaNtoNum(T.Transform):
    """Replace NaN values with finite numbers in input tensors.

    This transformation provides a convenient wrapper around PyTorch's nan_to_num
    function for use in torchvision transformation pipelines. It replaces NaN values
    with zeros (or other specified values), positive infinity with large finite numbers,
    and negative infinity with small finite numbers.

    Inherits from torchvision.transforms.v2.Transform to provide consistent API
    and integration with transformation pipelines.

    Parameters
    ----------
    params : dict of str to any
        Parameters passed to torch.nan_to_num function. See PyTorch documentation
        for torch.nan_to_num for complete parameter details.

    Attributes
    ----------
    params : dict
        Stored parameters for the nan_to_num transformation.

    Methods
    -------
    _transform(x, params)
        Apply nan_to_num transformation to input tensor.
    transform(inpt, params)
        Public interface for applying the transformation.

    Examples
    --------
    Replace NaNs with zeros (default behavior):

    >>> import torch
    >>> from cellmap_data.transforms.augment import NaNtoNum
    >>> nan_transform = NaNtoNum(params={})
    >>> x = torch.tensor([[1.0, float('nan')], [float('inf'), 2.0]])
    >>> result = nan_transform.transform(x)
    >>> print(result)
    tensor([[1.0000e+00, 0.0000e+00],
            [3.4028e+38, 2.0000e+00]])

    Custom replacement values:

    >>> custom_transform = NaNtoNum(params={
    ...     'nan': -1.0,
    ...     'posinf': 999.0,
    ...     'neginf': -999.0
    ... })
    >>> x_custom = torch.tensor([float('nan'), float('inf'), float('-inf')])
    >>> result_custom = custom_transform.transform(x_custom)
    >>> print(result_custom)  # [-1.0, 999.0, -999.0]

    See Also
    --------
    torch.nan_to_num : Underlying PyTorch function
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        """Initialize the NaN to number transformation.

        Parameters
        ----------
        params : dict of str to any
            Parameters for the torch.nan_to_num transformation. Common parameters:
            - 'nan': replacement value for NaN (default: 0.0)
            - 'posinf': replacement value for positive infinity (default: large finite number)
            - 'neginf': replacement value for negative infinity (default: small finite number)
            See torch.nan_to_num documentation for complete parameter details.

        Examples
        --------
        Default behavior (replace NaNs with 0, infinities with finite extremes):
        >>> transform = NaNtoNum(params={})

        Custom replacement values:
        >>> transform = NaNtoNum(params={'nan': -1.0, 'posinf': 1e6})
        """
        super().__init__()
        self.params = params

    def _transform(self, inpt: Any, params: Dict[str, Any] | None = None) -> Any:
        """Apply nan_to_num transformation to input tensor.

        Parameters
        ----------
        inpt : Any
            Input tensor or other data structure to process.
        params : dict of str to any, optional
            Additional parameters (not used, maintained for interface compatibility).

        Returns
        -------
        Any
            Input with NaN values replaced according to configured parameters.
        """
        return inpt.nan_to_num(**self.params)

    def __repr__(self) -> str:
        """Return a string representation of the transformation."""
        return f"{self.__class__.__name__}(params={self.params})"

    def transform(self, inpt: Any, params: Dict[str, Any] | None = None) -> Any:
        """Transform the input tensor by replacing invalid values.

        Public interface for applying the nan_to_num transformation.

        Parameters
        ----------
        inpt : Any
            Input tensor or data structure to transform.
        params : dict of str to any, optional
            Additional parameters (not used, maintained for interface compatibility).

        Returns
        -------
        Any
            Transformed input with NaN and infinity values replaced.
        """
        return self._transform(inpt, params)
