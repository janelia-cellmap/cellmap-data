import torch
import torch.nn.functional as F


class GaussianBlur(torch.nn.Module):
    """Apply Gaussian blur filter to input tensors for smoothing augmentation.

    This transformation applies a Gaussian blur filter using convolution operations
    for 2D or 3D input tensors. The blur is implemented via separable Gaussian kernels
    with configurable kernel size and standard deviation, providing efficient smoothing
    augmentation suitable for neural network training.

    The module creates a fixed Gaussian kernel and applies it via grouped convolution
    to preserve channel independence. Supports both 2D and 3D inputs with automatic
    device placement and batch dimension handling.

    Inherits from torch.nn.Module for PyTorch compatibility and parameter management.

    Parameters
    ----------
    kernel_size : int, optional
        Size of the Gaussian kernel in pixels/voxels, by default 3.
        Must be an odd positive integer to ensure symmetric filtering.
    sigma : float, optional
        Standard deviation of the Gaussian distribution, by default 0.1.
        Controls the blur intensity - larger values create stronger blur.
    dim : int, optional
        Spatial dimensionality for blur operation, by default 2.
        Supports 2 (for images) or 3 (for volumes).
    channels : int, optional
        Number of input channels to process, by default 1.
        Each channel is blurred independently using grouped convolution.

    Attributes
    ----------
    kernel_size : int
        Size of the Gaussian kernel.
    kernel_shape : tuple
        Shape of the kernel for the specified dimensionality.
    sigma : float
        Standard deviation of the Gaussian distribution.
    dim : int
        Spatial dimensionality.
    kernel : torch.Tensor
        The computed Gaussian kernel values.
    conv : torch.nn.Conv2d or torch.nn.Conv3d
        Convolution layer for applying the blur.

    Methods
    -------
    _create_gaussian_kernel()
        Generate normalized Gaussian kernel for convolution.
    forward(x)
        Apply Gaussian blur to input tensor.

    Raises
    ------
    AssertionError
        If dim is not 2 or 3, or if kernel_size is not odd.

    Examples
    --------
    Basic 2D image blurring:

    >>> import torch
    >>> from cellmap_data.transforms.augment import GaussianBlur
    >>> blur_transform = GaussianBlur(kernel_size=5, sigma=1.0, dim=2)
    >>> x = torch.rand(1, 64, 64)  # Single-channel 2D image
    >>> blurred = blur_transform(x)
    >>> print(blurred.shape)
    torch.Size([1, 64, 64])

    3D volume processing with multiple channels:

    >>> blur_3d = GaussianBlur(kernel_size=3, sigma=0.5, dim=3, channels=2)
    >>> volume = torch.rand(2, 32, 32, 32)  # Two-channel 3D volume
    >>> smoothed = blur_3d(volume)

    See Also
    --------
    GaussianNoise : Additive noise augmentation
    RandomContrast : Contrast-based augmentation
    """

    def __init__(
        self, kernel_size: int = 3, sigma: float = 0.1, dim: int = 2, channels: int = 1
    ):
        """Initialize Gaussian blur transform with specified parameters.

        Parameters
        ----------
        kernel_size : int, optional
            Size of the Gaussian kernel, by default 3.
            Must be an odd positive integer for symmetric filtering.
        sigma : float, optional
            Standard deviation of the Gaussian distribution, by default 0.1.
            Larger values produce stronger blur effects.
        dim : int, optional
            Spatial dimensionality (2 or 3), by default 2.
            Determines whether to use 2D or 3D convolution.
        channels : int, optional
            Number of input channels, by default 1.
            Each channel is processed independently.

        Raises
        ------
        AssertionError
            If dim is not 2 or 3.
            If kernel_size is not an odd number.

        Notes
        -----
        The Gaussian kernel is pre-computed and frozen (no gradient computation).
        Convolution uses 'replicate' padding to minimize edge artifacts and
        grouped convolution to maintain channel independence.
        """
        super().__init__()
        assert dim in (2, 3), "Only 2D or 3D Gaussian blur is supported."
        assert kernel_size % 2 == 1, "Kernel size should be an odd number."

        self.kernel_size = kernel_size
        self.kernel_shape = (kernel_size,) * dim
        self.sigma = sigma
        self.dim = dim
        self.kernel = self._create_gaussian_kernel()
        self.conv = {2: torch.nn.Conv2d, 3: torch.nn.Conv3d}[dim](
            in_channels=channels,
            out_channels=channels,
            kernel_size=self.kernel_shape,
            bias=False,
            padding="same",  # Automatically pads to keep output size same as input
            groups=channels,  # Apply the same kernel to each channel independently
            padding_mode="replicate",  # Use 'replicate' padding to avoid artifacts
        )
        kernel = self.kernel.view(1, 1, *self.kernel_shape)
        kernel = kernel.repeat(channels, 1, *(1,) * self.dim)
        self.conv.weight.data = kernel
        self.conv.weight.requires_grad = False  # Freeze the kernel weights

    def _create_gaussian_kernel(self):
        """Create normalized Gaussian kernel for convolution filtering.

        Generates a multi-dimensional Gaussian kernel based on the configured
        kernel size, sigma, and dimensionality. The kernel is normalized to
        sum to 1.0 for proper filtering behavior.

        Returns
        -------
        torch.Tensor
            Normalized Gaussian kernel with shape matching kernel_shape.
            Values represent the Gaussian distribution discretized over
            the kernel grid centered at zero.

        Notes
        -----
        The kernel is computed using the formula:
        K(x) = exp(-||x||^2 / (2 * sigma^2)) / normalization_factor
        where x represents the coordinate offsets from kernel center.
        """
        coords = torch.arange(self.kernel_size) - self.kernel_size // 2
        axes_coords = torch.meshgrid(*[[coords] * self.dim], indexing="ij")
        kernel = torch.exp(
            -torch.sum(torch.stack([coord**2 for coord in axes_coords]), dim=0)
            / (2 * self.sigma**2)
        )

        kernel /= kernel.sum()  # Normalize
        return kernel

    def forward(self, x: torch.Tensor):
        """Apply Gaussian blur to input tensor.

        Performs Gaussian blurring via convolution with the pre-computed kernel.
        Automatically handles device placement and batch dimension management
        for both 2D and 3D inputs.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to blur. Expected shapes:
            - 2D: (height, width) or (channels, height, width) or (batch, channels, height, width)
            - 3D: (depth, height, width) or (channels, depth, height, width) or (batch, channels, depth, height, width)

        Returns
        -------
        torch.Tensor
            Blurred tensor with same shape and device as input.
            Dtype is converted to float for convolution, then back to original dtype.

        Examples
        --------
        >>> import torch
        >>> blur = GaussianBlur(kernel_size=5, sigma=1.0, dim=2)
        >>> x = torch.rand(64, 64)  # 2D input without batch/channel dims
        >>> result = blur.forward(x)
        >>> print(result.shape)
        torch.Size([64, 64])

        Notes
        -----
        - Convolution layer is automatically moved to input tensor's device
        - Input is converted to float32 for convolution operations
        - Batch dimension is added/removed automatically when needed
        """
        self.conv.to(x.device, non_blocking=True)
        if len(x.shape) == self.dim:
            # For 2D or 3D input without batch dimension
            x = x.view(1, *x.shape)  # Add batch dimension
            out = self.conv(x.to(torch.float))
            out = out.view(*x.shape[1:])  # Remove batch dimension
        else:
            out = self.conv(x.to(torch.float))

        return out
