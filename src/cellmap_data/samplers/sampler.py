from typing import Callable, Optional, Sequence, Sized
import torch
from torch.utils.data import Sampler


class CellMapSampler(Sampler):
    """CellMap sampler.

    Attributes:
        data_source (Sized): Data source.
        indices (Sequence[int]): Indices.
        check_function (Optional[Callable]): Check function.
        generator (Optional[torch.Generator]): Generator.

    Methods:
        __iter__: Iterate over the indices.
        __len__: Return the length of the indices
    """

    def __init__(
        self,
        data_source: Sized,
        indices: Sequence[int],
        check_function: Optional[Callable] = None,
        generator: Optional[torch.Generator] = None,
    ):
        """Initialize the CellMap sampler.

        Args:
            data_source (Sized): Data source.
            indices (Sequence[int]): Indices.
            check_function (Optional[Callable], optional): Check function. Defaults to None.
            generator (Optional[torch.Generator], optional): Generator. Defaults to None.
        """
        super().__init__()
        self.data_source = data_source
        self.indices = indices
        self.check_function = check_function
        self.generator = generator

    def __iter__(self):
        """Iterate over the indices

        Yields:
            int: Index
        """
        for i in torch.randperm(len(self.indices), generator=self.generator):
            if self.check_function is None or self.check_function(self.indices[i]):
                yield self.indices[i]

    def __len__(self):
        """Return the number of indices"""
        return len(self.indices)
