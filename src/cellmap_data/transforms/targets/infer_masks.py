from typing import Mapping, Sequence
import torch

import logging

logger = logging.getLogger(__name__)


class InferMasks(torch.nn.Module):
    """
    Transform for constructing partial masks based on true-negatives of class presence based on other class masks
    """

    def __init__(
        self,
        class_relation_dict: Mapping[int, Sequence[int]],
        empty_value: float | int = -100,
    ) -> None:
        super().__init__()
        self.class_relation_dict = class_relation_dict
        self.empty_value = empty_value

    def forward(self, x: torch.Tensor):
        out = torch.ones_like(x) * self.empty_value
        # TODO: Do with slices (no for loops)
        for b in range(x.shape[0]):
            for class_ind, related_inds in self.class_relation_dict.items():
                if not (x[b, class_ind] == self.empty_value).all():
                    continue
                for ind in related_inds:
                    if (x[b, class_ind] == self.empty_value).any():
                        continue
                    out[b, class_ind][x[b, ind] > 0] = 0
        return out
