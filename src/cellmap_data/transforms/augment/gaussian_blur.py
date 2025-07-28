import torch
import torch.nn.functional as F


class GaussianBlur(torch.nn.Module):
    """Apply Gaussian blur filter to input tensors for smoothing augmentation.

    Applies a Gaussian blur filter using convolution operations for 2D or 3D inputs.
    The blur is implemented via separable Gaussian kernels with configurable kernel
    size and standard deviation.

    Args:
        kernel_size: Size of the Gaussian kernel in pixels/voxels. Defaults to 3.
            Must be an odd positive integer.
        sigma: Standard deviation of the Gaussian distribution. Defaults to 0.1.
            Controls blur intensity - larger values create stronger blur.
        dim: Spatial dimensionality for blur operation. Defaults to 2.
            Supports 2 (for images) or 3 (for volumes).
        channels: Number of input channels to process. Defaults to 1.

    Examples:
        >>> blur_transform = GaussianBlur(kernel_size=5, sigma=1.0, dim=2)
        >>> x = torch.rand(1, 64, 64)  # Single-channel 2D image
        >>> blurred = blur_transform(x)
    """

    def __init__(
        self, kernel_size: int = 3, sigma: float = 0.1, dim: int = 2, channels: int = 1
    ):
        """Initialize Gaussian blur transform with specified parameters.

        Args:
            kernel_size: Size of the Gaussian kernel. Defaults to 3.
                Must be an odd positive integer for symmetric filtering.
            sigma: Standard deviation of the Gaussian distribution. Defaults to 0.1.
            dim: Spatial dimensionality (2 or 3). Defaults to 2.
            channels: Number of input channels. Defaults to 1.

        Raises:
            AssertionError: If dim is not 2 or 3, or if kernel_size is not odd.
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

        Returns:
            Normalized Gaussian kernel with shape matching kernel_shape.
            Values represent the Gaussian distribution discretized over
            the kernel grid centered at zero.
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

        Args:
            x: Input tensor to blur. Expected shapes:
                - 2D: (height, width) or (channels, height, width) or (batch, channels, height, width)
                - 3D: (depth, height, width) or (channels, depth, height, width) or (batch, channels, depth, height, width)

        Returns:
            Blurred tensor with same shape and device as input.
            Dtype is converted to float for convolution, then back to original dtype.
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
