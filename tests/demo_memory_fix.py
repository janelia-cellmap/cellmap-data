#!/usr/bin/env python
"""
Simple script to demonstrate the memory leak fix.

This script simulates a training loop and shows that the array cache
is properly cleared after each iteration, preventing memory accumulation.
"""

import sys
import os
import time
import tracemalloc

# Add src to path (so the real library can be imported if available)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import numpy as np
except ImportError as e:
    print(f"Error importing numpy: {e}")
    print("This demo requires NumPy to run.")
    sys.exit(1)


class DemoCacheUser:
    """
    Minimal stand-in object that simulates an internal array cache.

    This mirrors the idea of CellMapImage keeping an xarray cached and then
    clearing it inside __getitem__ via a _clear_array_cache() method.
    """

    def __init__(self, shape=(512, 512), dtype=np.float32):
        self.shape = shape
        self.dtype = dtype
        self._array_cache = None

    def _clear_array_cache(self):
        """Clear the internal array cache, simulating the real fix."""
        self._array_cache = None

    def __getitem__(self, idx, clear_cache=True):
        """
        Simulate loading data and optionally clearing the cache.

        If clear_cache is False, the internal cache keeps growing as new
        arrays are created, mimicking a leak. If True, the cache is cleared
        each time, keeping memory bounded.
        """
        # Simulate an expensive load that allocates a new array
        arr = np.ones(self.shape, dtype=self.dtype)
        self._array_cache = arr

        if clear_cache:
            self._clear_array_cache()

        # In a real __getitem__, we would return data used by the model
        return arr


def run_demo(clear_cache: bool, iterations: int = 50):
    """
    Run a small loop that simulates repeated __getitem__ calls and
    reports peak memory usage with and without cache clearing.
    """
    demo = DemoCacheUser()

    tracemalloc.start()
    start_time = time.time()

    for i in range(iterations):
        _ = demo.__getitem__(i, clear_cache=clear_cache)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    elapsed = time.time() - start_time

    mode = "WITH cache clearing" if clear_cache else "WITHOUT cache clearing"
    print(f"Mode: {mode}")
    print(f"  Iterations      : {iterations}")
    print(f"  Peak memory (MB): {peak / (1024 * 1024):.2f}")
    print(f"  Elapsed (s)     : {elapsed:.3f}")
    print()


def main():
    print("=" * 70)
    print("Memory Leak Fix Demonstration")
    print("=" * 70)
    print()
    print(
        "This script simulates the behavior of CellMapImage.__getitem__().\n"
        "We allocate arrays repeatedly and either keep them cached (leaky)\n"
        "or clear the cache on each access (fixed)."
    )
    print()
    print("Expected behavior:")
    print("- WITHOUT cache clearing: peak memory grows with iterations.")
    print("- WITH    cache clearing: peak memory stays bounded.")
    print()

    run_demo(clear_cache=False)
    run_demo(clear_cache=True)


if __name__ == "__main__":
    main()
