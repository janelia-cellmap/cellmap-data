from difflib import SequenceMatcher

import torch


def torch_max_value(dtype: torch.dtype) -> int:
    """
    Get the maximum value for a given torch dtype.

    Args:
        dtype (torch.dtype): Data type.

    Returns:
        int: Maximum value.
    """
    if dtype == torch.uint8:
        return 255
    elif dtype == torch.int8:
        return 127
    elif dtype == torch.int16:
        return 32767
    elif dtype == torch.int32:
        return 2147483647
    elif dtype == torch.int64:
        return 9223372036854775807
    else:
        # This is only here for completeness. This value is implicitly assumed in a lot of places so changing it is not
        # easy.
        return 1


def longest_common_substring(a: str, b: str) -> str:
    m = SequenceMatcher(None, a, b)
    match = m.find_longest_match(0, len(a), 0, len(b))
    return a[match.a : match.a + match.size]
