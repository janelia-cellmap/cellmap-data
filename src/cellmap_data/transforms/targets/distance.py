# from py_distance_transforms import transform_cuda, transform
import torch

from scipy.ndimage import distance_transform_edt as edt

transform = lambda x: torch.tensor(edt(x.cpu().numpy())).to(x.device)


class DistanceTransform(torch.nn.Module):
    """
    Compute the distance transform of the input.

    Attributes:
        use_cuda (bool): Use CUDA.

    Methods:
        _transform: Transform the input.
        forward: Forward pass.
    """

    def __init__(self, use_cuda: bool = False):
        """
        Initialize the distance transform.

        Args:
            use_cuda (bool, optional): Use CUDA. Defaults to False.

        Raises:
            NotImplementedError: CUDA is not supported yet.
        """
        UserWarning("This is still in development and may not work as expected")
        super().__init__()
        self.use_cuda = use_cuda
        if self.use_cuda:
            raise NotImplementedError(
                "CUDA is not supported yet because testing did not return expected results."
            )

    def _transform(self, x: torch.Tensor):
        """Transform the input."""
        if self.use_cuda and x.device.type == "cuda":
            raise NotImplementedError(
                "CUDA is not supported yet because testing did not return expected results."
            )
            return transform_cuda(x)
        else:
            return transform(x)

    def forward(self, x: torch.Tensor):
        """Forward pass."""
        # TODO: Need to figure out how to prevent having inaccurate distance values at the edges --> precompute
        for b in range(x.shape[0]):
            for class_ind in range(x.shape[1]):
                # distance = self._transform(x[b, class_ind].nan_to_num(0))
                distance = self._transform(x[b, class_ind])
                distance[x[b, class_ind].isnan()] = torch.nan
                x[b, class_ind] = distance
        return x


class SignedDistanceTransform(torch.nn.Module):
    """
    Compute the signed distance transform of the input - positive within objects and negative outside.

    Attributes:
        use_cuda (bool): Use CUDA.

    Methods:
        _transform: Transform the input.
        forward: Forward pass.
    """

    def __init__(self, use_cuda: bool = False):
        """
        Initialize the signed distance transform.

        Args:
            use_cuda (bool, optional): Use CUDA. Defaults to False.

        Raises:
            NotImplementedError: CUDA is not supported yet.
        """
        UserWarning("This is still in development and may not work as expected")
        super().__init__()
        self.use_cuda = use_cuda
        if self.use_cuda:
            raise NotImplementedError(
                "CUDA is not supported yet because testing did not return expected results."
            )

    def _transform(self, x: torch.Tensor):
        """Transform the input."""
        if self.use_cuda and x.device.type == "cuda":
            raise NotImplementedError(
                "CUDA is not supported yet because testing did not return expected results."
            )
            return transform_cuda(x) - transform_cuda(x.logical_not())
        else:
            return transform(x) - transform(x.logical_not())

    def forward(self, x: torch.Tensor):
        """Forward pass."""
        # TODO: Need to figure out how to prevent having inaccurate distance values at the edges --> precompute
        for b in range(x.shape[0]):
            for class_ind in range(x.shape[1]):
                # distance = self._transform(x[b, class_ind].nan_to_num(0))
                distance = self._transform(x[b, class_ind])
                distance[x[b, class_ind].isnan()] = torch.nan
                x[b, class_ind] = distance
        return x
