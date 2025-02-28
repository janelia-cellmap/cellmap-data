# from py_distance_transforms import transform_cuda, transform
from typing import Any, Dict
import torch

from scipy.ndimage import distance_transform_edt as edt


def transform(x: torch.Tensor) -> torch.Tensor:
    return torch.tensor(edt(x.cpu().numpy())).to(x.device, non_blocking=True)


class DistanceTransform(torch.nn.Module):
    """
    Compute the distance transform of the input.

    Attributes:
        use_cuda (bool): Use CUDA.
        clip (list): Clip the output to the specified range.

    Methods:
        _transform: Transform the input.
        forward: Forward pass.
    """

    def __init__(self, use_cuda: bool = False, clip=[-torch.inf, torch.inf]) -> None:
        """
        Initialize the distance transform.

        Args:
            use_cuda (bool, optional): Use CUDA. Defaults to False.
            clip (list, optional): Clip the output to the specified range. Defaults to [-torch.inf, torch.inf].

        Raises:
            NotImplementedError: CUDA is not supported yet.
        """
        UserWarning("This is still in development and may not work as expected")
        super().__init__()
        self.use_cuda = use_cuda
        self.clip = clip
        if self.use_cuda:
            raise NotImplementedError(
                "CUDA is not supported yet because testing did not return expected results."
            )

    def _transform(self, x: torch.Tensor, params: Any | None = None) -> torch.Tensor:
        """Transform the input."""
        if self.use_cuda and x.device.type == "cuda":
            raise NotImplementedError(
                "CUDA is not supported yet because testing did not return expected results."
            )
            # return transform_cuda(x)
        else:
            return transform(x).clip(self.clip[0], self.clip[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # TODO: Need to figure out how to prevent having inaccurate distance values at the edges --> precompute
        # distance = self._transform(x.nan_to_num(0))
        distance = self._transform(x)
        distance[x.isnan()] = torch.nan
        x = distance
        return x

    def transform(self, x: Any, params: Dict[str, Any] | None = None) -> Any:
        """Transform the input."""
        return self._transform(x, params)


class SignedDistanceTransform(torch.nn.Module):
    """
    Compute the signed distance transform of the input - positive within objects and negative outside.

    Attributes:
        use_cuda (bool): Use CUDA.
        clip (list): Clip the output to the specified range.

    Methods:
        _transform: Transform the input.
        forward: Forward pass.
    """

    def __init__(self, use_cuda: bool = False, clip=[-torch.inf, torch.inf]) -> None:
        """
        Initialize the signed distance transform.

        Args:
            use_cuda (bool, optional): Use CUDA. Defaults to False.
            clip (list, optional): Clip the output to the specified range. Defaults to [-torch.inf, torch.inf].

        Raises:
            NotImplementedError: CUDA is not supported yet.
        """
        UserWarning("This is still in development and may not work as expected")
        super().__init__()
        self.use_cuda = use_cuda
        self.clip = clip
        if self.use_cuda:
            raise NotImplementedError(
                "CUDA is not supported yet because testing did not return expected results."
            )

    def _transform(
        self, x: torch.Tensor, params: Dict[str, Any] | None = None
    ) -> torch.Tensor:
        """Transform the input."""
        if self.use_cuda and x.device.type == "cuda":
            raise NotImplementedError(
                "CUDA is not supported yet because testing did not return expected results."
            )
            # return transform_cuda(x) - transform_cuda(x.logical_not())
        else:
            # TODO: Fix this to be correct

            return transform(x).clip(self.clip[0], self.clip[1]) - transform(
                x.logical_not()
            ).clip(self.clip[0], self.clip[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # TODO: Need to figure out how to prevent having inaccurate distance values at the edges --> precompute
        # distance = self._transform(x.nan_to_num(0))
        distance = self._transform(x)
        distance[x.isnan()] = torch.nan
        x = distance
        return x

    def transform(self, x: Any, params: Dict[str, Any] | None = None) -> Any:
        """Transform the input."""
        return self._transform(x, params)
