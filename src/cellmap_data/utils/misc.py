from difflib import SequenceMatcher
import os
from typing import Any, Mapping, Sequence

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


def split_target_path(path: str) -> tuple[str, list[str]]:
    """Splits a path to groundtruth data into the main path string, and the classes supplied for it."""
    try:
        path_prefix, path_rem = path.split("[")
        classes, path_suffix = path_rem.split("]")
        classes = classes.split(",")
        path_string = path_prefix + "{label}" + path_suffix
    except ValueError:
        path_string = path
        classes = [path.split(os.path.sep)[-1]]
    return path_string, classes


def is_array_2D(
    array_info: Mapping[str, Any] | None, summary: bool = True
) -> bool | Mapping[str, bool]:
    """Checks if the array has only 2 dimensions of shape specified."""
    if array_info is None or len(array_info) == 0:
        return False
    elif "shape" in array_info:
        return len(array_info["shape"]) == 2
    else:
        arrays = {}
        for key, value in array_info.items():
            arrays[key] = is_array_2D(value)
        if summary:
            return all(arrays.values())
        else:
            return arrays


def array_has_singleton_dim(
    array_info: Mapping[str, Any] | None, summary: bool = True
) -> bool | Mapping[str, bool]:
    """Checks if the array has a singleton dimension."""
    if array_info is None or len(array_info) == 0:
        return False
    elif "shape" in array_info:
        return 1 in array_info["shape"]
    else:
        arrays = {}
        for key, value in array_info.items():
            arrays[key] = array_has_singleton_dim(value)
        if summary:
            return any(arrays.values())
        else:
            return arrays


def get_sliced_shape(shape: Sequence[int], axis: int) -> Sequence[int]:
    """Returns a shape expanded from 2D and sliced along the specified axis."""
    shape = list(shape)
    if 1 in shape:
        singleton_idx = shape.index(1)
        if singleton_idx != axis:
            # Move singleton to the current axis
            shape.insert(axis, shape.pop(singleton_idx))
    else:
        # If no singleton, just add a singleton dimension at the current axis
        shape.insert(axis, 1)
    return tuple(shape)


def permute_singleton_dimension(arr_dict, axis):
    for arr_name, arr_info in arr_dict.items():
        arr_info["shape"] = get_sliced_shape(arr_info["shape"], axis)
