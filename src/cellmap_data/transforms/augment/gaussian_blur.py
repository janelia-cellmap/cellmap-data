import torch
import torch.nn.functional as F


class GaussianBlur(torch.nn.Module):
    def __init__(
        self, kernel_size: int = 3, sigma: float = 0.1, dim: int = 2, channels: int = 1
    ):
        """
        Initialize a Gaussian Blur module.

        Args:
            kernel_size (int): Size of the Gaussian kernel (should be odd).
            sigma (float): Standard deviation of the Gaussian distribution.
            dim (int): Dimensionality (2 or 3) for applying the blur.
            channels (int): Number of input channels (default is 1).
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
        """Create a Gaussian kernel for 2D or 3D convolution."""
        coords = torch.arange(self.kernel_size) - self.kernel_size // 2
        axes_coords = torch.meshgrid(*[[coords] * self.dim], indexing="ij")
        kernel = torch.exp(
            -torch.sum(torch.stack([coord**2 for coord in axes_coords]), dim=0)
            / (2 * self.sigma**2)
        )

        kernel /= kernel.sum()  # Normalize
        return kernel

    def forward(self, x: torch.Tensor):
        """Apply Gaussian blur to the input tensor."""
        self.conv.to(x.device, non_blocking=True)
        if len(x.shape) == self.dim:
            # For 2D or 3D input without batch dimension
            x = x.view(1, *x.shape)  # Add batch dimension
            out = self.conv(x.to(torch.float))
            out = out.view(*x.shape[1:])  # Remove batch dimension
        else:
            out = self.conv(x.to(torch.float))

        return out
