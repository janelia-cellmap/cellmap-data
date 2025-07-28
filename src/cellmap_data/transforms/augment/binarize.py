from typing import Any, Dict
import torchvision.transforms.v2 as T
import torch


class Binarize(T.Transform):
    """Binarize input tensors using a threshold-based transformation.

    This transformation converts input tensors to binary values by applying
    a threshold comparison. Values above the threshold become 1 (or dtype max),
    values below become 0. NaN values are preserved in the output when present
    in floating-point tensors.

    Inherits from torchvision.transforms.v2.Transform to provide consistent API
    and integration with transformation pipelines.

    Parameters
    ----------
    threshold : float or int, optional
        Threshold value for binarization, by default 0.
        Values greater than threshold become 1, others become 0.

    Attributes
    ----------
    threshold : float or int
        The threshold value used for binarization.

    Methods
    -------
    _transform(x, params)
        Apply binarization to input tensor.

    Examples
    --------
    Basic thresholding at zero:

    >>> import torch
    >>> from cellmap_data.transforms.augment import Binarize
    >>> binarize = Binarize(threshold=0)
    >>> x = torch.tensor([[-1.0, 0.5], [2.0, -0.3]])
    >>> result = binarize._transform(x)
    >>> print(result)
    tensor([[0., 1.],
            [1., 0.]])

    Custom threshold with NaN handling:

    >>> binarize_half = Binarize(threshold=0.5)
    >>> x_nan = torch.tensor([[0.3, 0.8], [float('nan'), 1.2]])
    >>> result_nan = binarize_half._transform(x_nan)
    >>> print(result_nan)  # [0., 1.], [nan, 1.]

    See Also
    --------
    Normalize : Value normalization transformation
    """

    def __init__(self, threshold=0) -> None:
        """Initialize the binarization transformation.

        Parameters
        ----------
        threshold : float or int, optional
            Threshold value for the binary comparison, by default 0.
            Values strictly greater than this threshold become 1,
            others become 0.
        """
        super().__init__()
        self.threshold = threshold

    def _transform(self, inpt: Any, params: Dict[str, Any] | None = None) -> Any:
        """Apply binarization transformation to input tensor.

        Parameters
        ----------
        inpt : Any
            Input tensor to binarize. Should be numeric tensor.
        params : dict of str to any, optional
            Additional parameters (not used, maintained for interface compatibility).

        Returns
        -------
        Any
            Binarized tensor where values > threshold become 1 and others become 0.
            NaN values are preserved when present in floating-point inputs.

        Notes
        -----
        For integer inputs containing NaN, the output is converted to float32
        to accommodate NaN preservation.
        """
        out = (inpt > self.threshold).to(inpt.dtype)
        hasnans = inpt.isnan().any()
        if not torch.is_floating_point(out) and hasnans:
            out = out.to(torch.float32)
        if hasnans:
            out[inpt.isnan()] = torch.nan
        return out

    def __repr__(self) -> str:
        """Return a string representation of the transformation."""
        return f"{self.__class__.__name__}(threshold={self.threshold})"

    def transform(self, x: Any, params: Dict[str, Any] | None = None) -> Any:
        """Transform the input."""
        return self._transform(x, params)
