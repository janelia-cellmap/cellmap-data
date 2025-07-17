from collections.abc import Iterator, Sequence
from typing import Callable, Optional
import torch


class MutableSubsetRandomSampler(torch.utils.data.Sampler[int]):
    """A mutable version of SubsetRandomSampler that allows changing the indices after initialization.

    Args:
        indices_generator (Callable[[], Sequence[int]]): A callable that returns a sequence of indices to sample from.
        rng (Optional[torch.Generator]): Generator used in sampling.
    """

    indices: Sequence[int]
    indices_generator: Callable
    rng: Optional[torch.Generator]

    def __init__(
        self, indices_generator: Callable, rng: Optional[torch.Generator] = None
    ):
        self.indices_generator = indices_generator
        self.indices = list(self.indices_generator())
        self.rng = rng

    def __iter__(self) -> Iterator[int]:
        for i in torch.randperm(len(self.indices), generator=self.rng):
            yield self.indices[i]

    def __len__(self) -> int:
        return len(self.indices)

    def refresh(self) -> None:
        """Redraw the indices used by the sampler by calling the indices generator."""
        self.indices = list(self.indices_generator())
