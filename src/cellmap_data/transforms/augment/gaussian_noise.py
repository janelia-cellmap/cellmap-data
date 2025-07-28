import torch


class GaussianNoise(torch.nn.Module):
    """Add Gaussian noise to input tensors for data augmentation.

    Applies additive Gaussian noise to tensors with configurable mean and standard
    deviation. Noise is automatically placed on the same device as input.

    Args:
        mean: Mean of the Gaussian noise distribution. Defaults to 0.0.
        std: Standard deviation of the noise distribution. Defaults to 0.1.

    Examples:
        >>> noise_transform = GaussianNoise(mean=0.0, std=0.1)
        >>> x = torch.randn(2, 3, 64, 64)
        >>> noisy_x = noise_transform(x)
    """

    def __init__(self, mean: float = 0.0, std: float = 0.1) -> None:
        """Initialize the Gaussian noise transform.

        Args:
            mean: Mean of the Gaussian noise distribution. Defaults to 0.0.
            std: Standard deviation of the noise distribution. Defaults to 0.1.
        """
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian noise to the input tensor.

        Args:
            x: Input tensor to add noise to.

        Returns:
            Input tensor with additive Gaussian noise applied.
        """
        noise = torch.normal(mean=self.mean, std=self.std, size=x.size())
        return x + noise.to(x.device, non_blocking=True)
