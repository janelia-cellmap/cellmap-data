import torch
import torch.nn.functional as F


class GaussianBlur(torch.nn.Module):
    def __init__(self, kernel_size: int = 3, sigma: float = 0.1, dim: int = 2):
        """
        Initialize a Gaussian Blur module.

        Args:
            kernel_size (int): Size of the Gaussian kernel (should be odd).
            sigma (float): Standard deviation of the Gaussian distribution.
            dim (int): Dimensionality (2 or 3) for applying the blur.
        """
        super().__init__()
        assert dim in (2, 3), "Only 2D or 3D Gaussian blur is supported."
        assert kernel_size % 2 == 1, "Kernel size should be an odd number."

        self.kernel_size = kernel_size
        self.kernel_shape = (kernel_size,) * dim
        self.sigma = sigma
        self.dim = dim
        self.kernel = self._create_gaussian_kernel()
        padding = self.kernel_size // 2
        if dim == 2:
            self.conv = lambda x, kernel: F.conv2d(
                x, kernel, padding=padding, groups=x.shape[1]
            )
        else:
            self.conv = lambda x, kernel: F.conv3d(
                x, kernel, padding=padding, groups=x.shape[1]
            )

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
        device = x.device
        kernel = self.kernel.to(device, non_blocking=True)

        # Add batch and channel dimensions
        kernel = kernel.view(1, 1, *self.kernel_shape)
        # Repeat for all channels
        kernel = kernel.repeat(x.shape[1], 1, *(1,) * self.dim)
        return self.conv(x, kernel)


if __name__ == "__main__":
    # Example usage
    image_2d = torch.rand(4, 3, 128, 128)  # Batch of 2D images with 3 channels
    image_3d = torch.rand(2, 3, 32, 32, 32)  # Batch of 3D volumes with 3 channels

    blur_2d = GaussianBlur(kernel_size=5, sigma=1.0, dim=2)
    blur_3d = GaussianBlur(kernel_size=5, sigma=1.0, dim=3)

    blurred_2d = blur_2d(image_2d)
    blurred_3d = blur_3d(image_3d)
