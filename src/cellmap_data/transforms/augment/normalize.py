from typing import Any, Dict
import torch
import torchvision.transforms.v2 as T


class Normalize(T.Transform):
    """Normalize tensor values with configurable shift and scale parameters.

    This transformation normalizes input tensors by applying a linear transformation
    (shift + scale) and converts the result to float dtype. It's particularly useful
    for normalizing uint8 image data to [0, 1] range or applying standardization
    with custom parameters.

    Inherits from torchvision.transforms.v2.Transform to provide consistent API
    and integration with torchvision transformation pipelines.

    Parameters
    ----------
    shift : float, optional
        Value to add before scaling, by default 0.
        Applied as: (input + shift) * scale.
    scale : float, optional
        Multiplicative scaling factor, by default 1/255.
        Common values: 1/255 for uint8→[0,1], 1/std for standardization.

    Methods
    -------
    _transform(x, params)
        Apply the normalization transformation to input tensor.
    transform(x, params)
        Public interface applying normalization with float conversion.

    Examples
    --------
    Normalize uint8 image to [0, 1] range:

    >>> import torch
    >>> from cellmap_data.transforms.augment import Normalize
    >>> x = torch.tensor([[0, 255], [128, 64]], dtype=torch.uint8)
    >>> normalize = Normalize(shift=0, scale=1/255)
    >>> result = normalize.transform(x, {})
    >>> print(result)
    tensor([[0.0000, 1.0000],
            [0.5020, 0.2510]])

    Custom normalization with shift:

    >>> # Normalize to [-1, 1] range from uint8
    >>> normalize = Normalize(shift=-127.5, scale=1/127.5)
    >>> x = torch.tensor([0, 127, 255], dtype=torch.uint8)
    >>> result = normalize.transform(x, {})
    >>> print(result)
    tensor([-1.0000, -0.0039,  1.0000])

    Standardization-style normalization:

    >>> # Simulate (x - mean) / std with shift and scale
    >>> mean, std = 128.0, 64.0
    >>> normalize = Normalize(shift=-mean, scale=1/std)
    >>> x = torch.tensor([64, 128, 192], dtype=torch.float32)
    >>> result = normalize.transform(x, {})
    >>> print(result)
    tensor([-1.0000,  0.0000,  1.0000])

    Notes
    -----
    The transformation always converts input to float dtype regardless of input type.
    This ensures numerical stability and compatibility with neural network training.

    Parameters are applied as: output = (input + shift) * scale, which allows for
    flexible normalization schemes including offset correction and scaling.

    For integration with torchvision pipelines, use the standard torchvision
    Compose functionality or call transform() method directly.

    See Also
    --------
    torchvision.transforms.v2.Normalize : Standard normalization with mean/std
    cellmap_data.transforms.augment.RandomContrast : Contrast-based augmentation
    """

    def __init__(self, shift=0, scale=1 / 255) -> None:
        """Initialize the normalization transformation.

        Parameters
        ----------
        shift : float, optional
            Shift values before scaling, by default 0.
        scale : float, optional
            Scale values after shifting, by default 1/255.

        This is helpful in normalizing the input to the range [0, 1], especially for data saved as uint8 which is scaled to [0, 255].

        Examples
        --------
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
        """Apply normalization transformation to input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to normalize.
        params : dict, optional
            Additional parameters (unused in this transform), by default None.

        Returns
        -------
        torch.Tensor
            Normalized tensor with applied shift and scale.
        """
        return (x + self.shift) * self.scale

    def __repr__(self) -> str:
        """Return string representation of the transformation.

        Returns
        -------
        str
            Class name for debugging and logging purposes.
        """
        return self.__class__.__name__

    def transform(self, x: Any, params: Dict[str, Any] | None = None) -> Any:
        """Apply normalization with automatic float conversion.

        Public interface that converts input to float before applying normalization.
        This ensures numerical stability and compatibility with training pipelines.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to normalize and convert to float.
        params : dict, optional
            Additional parameters (unused in this transform), by default None.

        Returns
        -------
        torch.Tensor
            Normalized float tensor.
        """
        return self._transform(x.to(torch.float), params)
