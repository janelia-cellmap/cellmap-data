from typing import Callable, Optional, Sequence, Sized
import torch
from torch.utils.data import Sampler


class CellMapSampler(Sampler):
    def __init__(
        self,
        data_source: Sized,
        indices: Sequence[int],
        check_function: Optional[Callable] = None,
        generator: Optional[torch.Generator] = None,
    ):
        super().__init__()
        self.data_source = data_source
        self.indices = indices
        self.check_function = check_function
        self.generator = generator

    def __iter__(self):
        for i in torch.randperm(len(self.indices), generator=self.generator):
            if self.check_function is None or self.check_function(self.indices[i]):
                yield self.indices[i]

    def __len__(self):
        return len(self.indices)
