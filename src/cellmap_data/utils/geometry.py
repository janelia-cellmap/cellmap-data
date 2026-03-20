"""Spatial bounding box utilities for world-coordinate arithmetic."""

from __future__ import annotations


def box_intersection(a: dict, b: dict) -> dict | None:
    """Intersection of two bounding boxes.

    Each box is ``{axis: (min, max), ...}`` in world coordinates (nm).
    Returns ``None`` if there is no overlap on any shared axis.
    """
    result = {}
    for ax in a:
        if ax not in b:
            continue
        lo = max(a[ax][0], b[ax][0])
        hi = min(a[ax][1], b[ax][1])
        if lo >= hi:
            return None
        result[ax] = (lo, hi)
    return result if result else None


def box_union(a: dict, b: dict) -> dict:
    """Bounding box that contains both *a* and *b*."""
    axes = set(a) | set(b)
    result = {}
    for ax in axes:
        if ax in a and ax in b:
            result[ax] = (min(a[ax][0], b[ax][0]), max(a[ax][1], b[ax][1]))
        elif ax in a:
            result[ax] = a[ax]
        else:
            result[ax] = b[ax]
    return result


def box_shape(box: dict, scale: dict) -> dict:
    """Convert a world bounding box to a voxel count per axis.

    Args:
        box:   ``{axis: (min, max)}`` in nm.
        scale: ``{axis: voxel_size}`` in nm/voxel.

    Returns:
        ``{axis: int}`` — number of voxels per axis (at least 1).
    """
    return {ax: max(1, int(round((box[ax][1] - box[ax][0]) / scale[ax]))) for ax in box}
