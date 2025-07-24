#!/usr/bin/env python3
"""
Debug script to investigate the coordinate transformation index out of bounds issue.
This script reproduces the conditions that cause the ValueError in np.unravel_index.
"""

import numpy as np
import logging
from typing import Mapping

# Set up logging to match what the dataset classes use
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def simulate_index_issue(
    idx: int, sampling_box_shape: Mapping[str, int], axis_order: str = "zyx"
):
    """
    Simulate the exact conditions that cause the ValueError in the dataset classes.
    """
    print(f"\n=== Debugging Index {idx} ===")
    print(f"Sampling box shape: {sampling_box_shape}")
    print(f"Axis order: {axis_order}")

    # Create the shape array as done in the actual code
    shape_array = [sampling_box_shape[c] for c in axis_order]
    print(f"Shape array for np.unravel_index: {shape_array}")

    # Calculate expected valid range
    total_size = np.prod(shape_array)
    print(f"Total valid indices: 0 to {total_size - 1} (size: {total_size})")

    if idx >= total_size:
        print(f"ERROR: Index {idx} >= total size {total_size}")
        print(f"This will cause ValueError in np.unravel_index")
        return None

    try:
        center = np.unravel_index(idx, shape_array)
        print(f"Successfully unraveled index {idx} to center: {center}")
        return center
    except ValueError as e:
        print(f"ERROR: {e}")
        return None


def test_edge_cases():
    """Test various edge cases that might cause issues."""

    print("=== Testing Edge Cases ===")

    # Case 1: Normal case
    simulate_index_issue(5, {"z": 10, "y": 10, "x": 10})

    # Case 2: Index at boundary
    simulate_index_issue(999, {"z": 10, "y": 10, "x": 10})

    # Case 3: Index out of bounds
    simulate_index_issue(1000, {"z": 10, "y": 10, "x": 10})

    # Case 4: Very small sampling box
    simulate_index_issue(1, {"z": 1, "y": 1, "x": 1})

    # Case 5: Zero or negative sampling box (pathological)
    simulate_index_issue(0, {"z": 0, "y": 1, "x": 1})

    # Case 6: Large index with small box
    simulate_index_issue(100, {"z": 2, "y": 2, "x": 2})


def analyze_sampling_box_calculation():
    """
    Analyze how sampling_box_shape might end up with incorrect values.
    """
    print("\n=== Analyzing Sampling Box Calculation ===")

    # Simulate the calculation from _get_box_shape
    def simulate_get_box_shape(source_box, largest_voxel_sizes):
        box_shape = {}
        for c, (start, stop) in source_box.items():
            size = stop - start
            size /= largest_voxel_sizes[c]
            box_shape[c] = int(np.floor(size))
        return box_shape

    # Test case 1: Normal values
    source_box = {"z": [0, 100], "y": [0, 100], "x": [0, 100]}
    largest_voxel_sizes = {"z": 8.0, "y": 8.0, "x": 8.0}
    result = simulate_get_box_shape(source_box, largest_voxel_sizes)
    print(f"Normal case: {result}")

    # Test case 2: Very small box
    source_box = {"z": [0, 1], "y": [0, 1], "x": [0, 1]}
    largest_voxel_sizes = {"z": 8.0, "y": 8.0, "x": 8.0}
    result = simulate_get_box_shape(source_box, largest_voxel_sizes)
    print(f"Small box case: {result}")

    # Test case 3: Large voxel sizes
    source_box = {"z": [0, 100], "y": [0, 100], "x": [0, 100]}
    largest_voxel_sizes = {"z": 200.0, "y": 200.0, "x": 200.0}
    result = simulate_get_box_shape(source_box, largest_voxel_sizes)
    print(f"Large voxel case: {result}")


def check_length_calculation_consistency():
    """
    Check if __len__ calculation is consistent with actual sampling_box_shape.
    """
    print("\n=== Checking Length Calculation Consistency ===")

    sampling_box_shapes = [
        {"z": 10, "y": 10, "x": 10},
        {"z": 0, "y": 10, "x": 10},  # Pathological case
        {"z": 1, "y": 1, "x": 1},  # Minimal case
    ]

    for shape in sampling_box_shapes:
        axis_order = "zyx"
        calculated_len = int(np.prod([shape[c] for c in axis_order]))
        print(f"Shape: {shape}, Calculated length: {calculated_len}")

        # Test last valid index
        if calculated_len > 0:
            last_valid_idx = calculated_len - 1
            print(f"  Last valid index: {last_valid_idx}")
            simulate_index_issue(last_valid_idx, shape, axis_order)

            # Test first invalid index
            first_invalid_idx = calculated_len
            print(f"  First invalid index: {first_invalid_idx}")
            simulate_index_issue(first_invalid_idx, shape, axis_order)


if __name__ == "__main__":
    test_edge_cases()
    analyze_sampling_box_calculation()
    check_length_calculation_consistency()
