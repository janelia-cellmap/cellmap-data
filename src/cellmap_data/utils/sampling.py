import warnings
from typing import Optional, Sequence
import torch


def min_redundant_inds(
    size: int, num_samples: int, rng: Optional[torch.Generator] = None
) -> torch.Tensor:
    """
    Returns a list of indices that will sample `num_samples` from a dataset of size `size` with minimal redundancy.
    If `num_samples` is greater than `size`, it will sample with replacement.
    """
    if num_samples > size:
        warnings.warn(
            f"Requested num_samples={num_samples} exceeds available samples={size}. "
            "Sampling with replacement using repeated permutations to minimize duplicates."
        )
    # Determine how many full permutations and remainder are needed
    full_iters = num_samples // size
    remainder = num_samples % size

    inds_list = []
    for _ in range(full_iters):
        inds_list.append(torch.randperm(size, generator=rng))
    if remainder > 0:
        inds_list.append(torch.randperm(size, generator=rng)[:remainder])
    return torch.cat(inds_list, dim=0)
