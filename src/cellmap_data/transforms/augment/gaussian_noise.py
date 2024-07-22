import torch


class GaussianNoise(torch.nn.Module):
    """
    Add Gaussian noise to the input. Subclasses torch.nn.Module.

    Attributes:
        mean (float): Mean of the noise.
        std (float): Standard deviation of the noise.

    Methods:
        forward: Forward pass.
    """

    def __init__(self, mean=0.0, std=1.0):
        """
        Initialize the Gaussian noise.

        Args:
            mean (float, optional): Mean of the noise. Defaults to 0.0.
            std (float, optional): Standard deviation of the noise. Defaults to 1.0.
        """
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        """Forward pass."""
        noise = torch.normal(mean=self.mean, std=self.std, size=x.size())
        return x + noise.to(x.device)
