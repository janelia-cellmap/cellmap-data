# from py_distance_transforms import transform_cuda, transform
import torch

from scipy.ndimage import distance_transform_edt as edt

transform = lambda x: torch.tensor(edt(x.cpu().numpy())).to(x.device)


class DistanceTransform(torch.nn.Module):
    # Compute the distance transform of the input
    def __init__(self, use_cuda: bool = False):
        super(DistanceTransform, self).__init__()
        self.use_cuda = use_cuda

    def _transform(self, x: torch.Tensor):
        if self.use_cuda and x.device.type == "cuda":
            UserWarning("This is still in development and may not work as expected")
            return transform_cuda(x)
        else:
            return transform(x)

    def forward(self, x: torch.Tensor):
        # TODO: Need to figure out how to prevent having inaccurate distance values at the edges --> precompute
        for b in range(x.shape[0]):
            for class_ind in range(x.shape[1]):
                # distance = self._transform(x[b, class_ind].nan_to_num(0))
                distance = self._transform(x[b, class_ind])
                distance[x[b, class_ind].isnan()] = torch.nan
                x[b, class_ind] = distance
        return x


class SignedDistanceTransform(torch.nn.Module):
    # Compute the distance transform of the input
    def __init__(self, use_cuda: bool = False):
        super(SignedDistanceTransform, self).__init__()
        self.use_cuda = use_cuda

    def _transform(self, x: torch.Tensor):
        if self.use_cuda and x.device.type == "cuda":
            UserWarning("This is still in development and may not work as expected")
            return transform_cuda(x) - transform_cuda(x.logical_not())
        else:
            return transform(x) - transform(x.logical_not())

    def forward(self, x: torch.Tensor):
        # TODO: Need to figure out how to prevent having inaccurate distance values at the edges --> precompute
        for b in range(x.shape[0]):
            for class_ind in range(x.shape[1]):
                # distance = self._transform(x[b, class_ind].nan_to_num(0))
                distance = self._transform(x[b, class_ind])
                distance[x[b, class_ind].isnan()] = torch.nan
                x[b, class_ind] = distance
        return x
