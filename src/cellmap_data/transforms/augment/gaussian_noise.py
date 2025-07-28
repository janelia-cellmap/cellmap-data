import torch


class GaussianNoise(torch.nn.Module):
    """Add Gaussian noise to input tensors for data augmentation.

    This class applies additive Gaussian noise to input tensors, commonly used
    as a data augmentation technique in neural network training. The noise is
    generated with configurable mean and standard deviation, and is automatically
    placed on the same device as the input tensor.

    Inherits from torch.nn.Module to provide standard PyTorch module functionality
    including device placement and parameter management.

    Parameters
    ----------
    mean : float, optional
        Mean value of the Gaussian noise distribution, by default 0.0.
        Positive values shift the input brighter, negative values darker.
    std : float, optional
        Standard deviation of the Gaussian noise distribution, by default 0.1.
        Controls the magnitude of noise variation added to the input.

    Attributes
    ----------
    mean : float
        Configured mean value for noise generation.
    std : float
        Configured standard deviation for noise generation.

    Methods
    -------
    forward(x)
        Apply Gaussian noise to input tensor.

    Examples
    --------
    Basic noise application:

    >>> import torch
    >>> from cellmap_data.transforms.augment import GaussianNoise
    >>> noise_transform = GaussianNoise(mean=0.0, std=0.1)
    >>> x = torch.randn(2, 3, 64, 64)
    >>> noisy_x = noise_transform(x)
    >>> print(f"Original range: [{x.min():.3f}, {x.max():.3f}]")
    >>> print(f"Noisy range: [{noisy_x.min():.3f}, {noisy_x.max():.3f}]")

    Higher noise for strong augmentation:

    >>> strong_noise = GaussianNoise(mean=0.0, std=0.5)
    >>> very_noisy = strong_noise(x)

    See Also
    --------
    RandomContrast : Contrast-based augmentation
    RandomGamma : Gamma correction augmentation
    """

    def __init__(self, mean: float = 0.0, std: float = 0.1) -> None:
        """Initialize the Gaussian noise transform.

        Parameters
        ----------
        mean : float, optional
            Mean value of the Gaussian noise distribution, by default 0.0.
            The noise will be centered around this value.
        std : float, optional
            Standard deviation of the Gaussian noise distribution, by default 0.1.
            Controls the spread and magnitude of the noise.

        Notes
        -----
        The default standard deviation of 0.1 is suitable for normalized image data
        in the range [0, 1]. For different data ranges, adjust accordingly.
        """
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian noise to the input tensor.

        Generates Gaussian noise with the configured mean and standard deviation,
        matches the input tensor's shape and device, then adds it to the input.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to which noise will be added. Can be any shape and dtype.
            The tensor device is preserved in the output.

        Returns
        -------
        torch.Tensor
            Input tensor with additive Gaussian noise applied. Same shape and
            device as input, but may have modified dtype based on noise operation.

        Examples
        --------
        >>> import torch
        >>> noise_transform = GaussianNoise(mean=0.0, std=0.1)
        >>> x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> noisy = noise_transform.forward(x)
        >>> print(noisy.shape == x.shape)
        True
        """
        noise = torch.normal(mean=self.mean, std=self.std, size=x.size())
        return x + noise.to(x.device, non_blocking=True)
