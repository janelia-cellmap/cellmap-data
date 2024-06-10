import torch


class GaussianNoise(torch.nn.Module):
    def __init__(self, mean=0.0, std=1.0):
        super(GaussianNoise, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        noise = torch.normal(mean=self.mean, std=self.std, size=x.size())
        return x + noise.to(x.device)
