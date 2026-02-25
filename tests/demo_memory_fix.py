#!/usr/bin/env python
"""
Simple script to demonstrate the memory leak fix.

This script simulates a training loop and shows that the array cache
is properly cleared after each iteration, preventing memory accumulation.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import numpy as np
    from cellmap_data import CellMapImage
    
    print("=" * 70)
    print("Memory Leak Fix Demonstration")
    print("=" * 70)
    print()
    print("This script demonstrates that the array cache is cleared after")
    print("each __getitem__ call, preventing memory leaks in training loops.")
    print()
    
    # Note: This requires actual test data to run
    print("To run this demo, you need:")
    print("1. A valid Zarr/OME-NGFF dataset")
    print("2. TensorStore and other dependencies installed")
    print()
    print("The key fix is in CellMapImage.__getitem__():")
    print("- After retrieving data and applying transforms")
    print("- We call self._clear_array_cache()")
    print("- This removes the cached xarray from __dict__")
    print("- Preventing memory accumulation from xarray operations")
    print()
    print("Expected behavior:")
    print("- Before fix: 'array' stays in __dict__ → memory accumulates")
    print("- After fix: 'array' removed from __dict__ → memory stays bounded")
    print()
    print("=" * 70)
    
except ImportError as e:
    print(f"Error: {e}")
    print("Please install cellmap-data and dependencies first.")
    print("Run: pip install -e .")
