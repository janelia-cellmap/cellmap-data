import math
import torch
from typing import Optional


def _feistel_prp_pow2(x: torch.Tensor, rounds: int, key: int, k: int) -> torch.Tensor:
    """
    Pseudorandom permutation over {0..2^k-1} using a Feistel network.
    Splits k bits into L/R halves and runs a few rounds with simple XOR/mix.
    """
    # split sizes
    r_bits = k // 2
    l_bits = k - r_bits
    r_mask = (1 << r_bits) - 1
    l_mask = (1 << l_bits) - 1

    L = (x >> r_bits) & l_mask
    R = x & r_mask

    # simple round function: mix R with key & round constant
    # all ops are invertible mod 2^n when used in Feistel
    for r in range(rounds):
        # cheap mix; use 64-bit for safety then mask back down
        F = R
        F = (F ^ ((F << 13) & r_mask)) & r_mask
        F = (F ^ (F >> 7)) & r_mask
        F = (F ^ ((F << 17) & r_mask)) & r_mask
        F = (F + ((key + 0x9E3779B97F4A7C15 + r) & r_mask)) & r_mask

        L, R = R, (L ^ F) & l_mask

        # swap roles/sizes midway if halves differ
        if l_bits != r_bits:
            L, R = R & r_mask, L & l_mask
            l_bits, r_bits = r_bits, l_bits
            l_mask, r_mask = r_mask, l_mask

    # recombine (reverse last swap if halves flipped odd times)
    if l_bits >= r_bits:
        y = ((L & l_mask) << r_bits) | (R & r_mask)
    else:
        y = ((R & r_mask) << l_bits) | (L & l_mask)
    return y & ((1 << k) - 1)


def _permute_to_range(
    x: torch.Tensor, rounds: int, key: int, M: int, k: int
) -> torch.Tensor:
    """
    Cycle-walk the PRP over 2^k until result < M.
    (Guaranteed to terminate; average ~1 iteration when 2^k close to M)
    """
    y = _feistel_prp_pow2(x, rounds, key, k)
    # cycle-walk for the small fraction mapping outside [0, M)
    mask = y >= M
    # rarely true; loop while any out-of-range remains
    while mask.any():
        y2 = _feistel_prp_pow2(y[mask], rounds, key, k)
        y[mask] = y2
        mask = y >= M
    return y


def min_redundant_inds(
    size: int,
    num_samples: int,
    rng: Optional[torch.Generator] = None,
    *,
    device: torch.device | str = "cpu",
    rounds: int = 5,
    chunk_size: int = 1_000_000,
) -> torch.Tensor:
    """
    Memory-efficient sampler with minimal redundancy.

    - Streams a pseudorandom permutation of [0, size).
    - If num_samples > size, emits full permutation(s) back-to-back with new keys.
    - Uses O(1) extra memory (besides the O(N) output tensor).
    - Works when size is huge (e.g., 314,157,057), avoids randperm(size).

    Args:
        size: dataset size (M)
        num_samples: number of indices to produce (N)
        rng: optional torch.Generator for reproducibility
        device: output device
        rounds: Feistel rounds (5–8 is plenty)
        chunk_size: processing batch size (tune for throughput/memory)

    Returns:
        Tensor of shape (num_samples,) with minimal duplicates.
    """
    if size <= 0:
        raise ValueError("Dataset size must be greater than 0.")
    if rng is None:
        rng = torch.Generator(device="cpu")
        rng.seed()

    M = int(size)
    N = int(num_samples)

    # ceil log2(M)
    k = math.ceil(math.log2(M))
    two_k = 1 << k

    out = torch.empty(N, device=device)

    def _new_key() -> int:
        # draw a 64-bit-ish key from rng without large tensors
        # use two int32 draws to form a 64-bit key
        a = int(torch.randint(0, 2**31, (1,), generator=rng, dtype=torch.int64))
        b = int(torch.randint(0, 2**31, (1,), generator=rng, dtype=torch.int64))
        return ((a << 32) ^ b) | 1  # make key odd

    filled = 0
    need = N
    perm_index = 0  # position within current permutation [0, 2^k)
    key = _new_key()

    while need > 0:
        # produce up to the remainder of current permutation or need, in chunks
        remain_in_perm = two_k - perm_index
        to_emit = min(need, remain_in_perm)

        start = perm_index
        end = start + to_emit
        perm_index = end

        # Process in sub-chunks to keep peak memory flat
        sub_start = 0
        while sub_start < to_emit:
            sub_end = min(sub_start + chunk_size, to_emit)
            n = sub_end - sub_start

            xs = torch.arange(
                start + sub_start, start + sub_end, dtype=torch.int64, device=device
            )
            ys = _permute_to_range(xs, rounds=rounds, key=key, M=M, k=k)
            out[filled : filled + n] = ys
            filled += n
            need -= n
            sub_start = sub_end

        # If we exhausted the 2^k domain, start a fresh permutation with a new key.
        if perm_index >= two_k and need > 0:
            perm_index = 0
            key = _new_key()

    return out.to(torch.long)
