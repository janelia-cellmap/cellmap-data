import warnings
from typing import Optional

import torch

MAX_SIZE = (
    512 * 1024 * 1024
)  # 512 million - increased from 64M to handle larger datasets efficiently


def min_redundant_inds(
    size: int, num_samples: int, rng: Optional[torch.Generator] = None
) -> torch.Tensor:
    """
    Returns a list of indices that will sample `num_samples` from a dataset of size `size` with minimal redundancy.
    If `num_samples` is greater than `size`, it will sample with replacement.
    """
    if size <= 0:
        raise ValueError("Size must be a positive integer.")
    elif size > MAX_SIZE:
        warnings.warn(
            f"Size={size} exceeds MAX_SIZE={MAX_SIZE}. Using faster sampling strategy that doesn't ensure minimal redundancy."
        )
        return torch.randint(0, size, (num_samples,), generator=rng)
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
