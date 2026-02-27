#!/usr/bin/env python
"""
Memory profiling demo for the CellMapImage array cache fix.

Demonstrates two levels of profiling:
  1. Mock class — fast, no real data needed, shows the principle.
  2. Real CellMapImage — uses a temporary Zarr dataset to profile
     actual xarray/TensorStore allocations.

Profiling tools used:
  - tracemalloc (built-in): snapshot comparison shows *what* is growing,
    not just peak usage.
  - objgraph (optional, pip install objgraph): counts live Python objects
    by type, confirming whether xarray DataArrays accumulate.

Usage:
  python tests/demo_memory_fix.py
  DEMO_ITERS=200 python tests/demo_memory_fix.py
"""

import gc
import io
import os
import sys
import tempfile
import tracemalloc
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

try:
    import objgraph

    HAS_OBJGRAPH = True
except ImportError:
    HAS_OBJGRAPH = False


# ---------------------------------------------------------------------------
# Profiling helpers
# ---------------------------------------------------------------------------


def profile_iters(label, call_fn, iterations=100, snapshot_every=25):
    """
    Run call_fn(i) for `iterations` steps and track memory growth.

    Prints a tracemalloc snapshot diff every `snapshot_every` steps showing
    which allocation sites are growing (not just peak usage). If objgraph is
    available, also prints live object-type counts so you can confirm whether
    xarray DataArrays or numpy arrays are accumulating.

    Args:
        label: Description printed as the section header.
        call_fn: Callable taking iteration index, e.g. lambda i: img[center].
        iterations: Total number of iterations to run.
        snapshot_every: How often to print an intermediate snapshot diff.
    """
    print(f"\n{'─' * 64}")
    print(f"  {label}")
    print(f"{'─' * 64}")

    gc.collect()
    tracemalloc.start()
    baseline = tracemalloc.take_snapshot()

    # objgraph.show_growth() tracks growth relative to the previous call;
    # calling it once here establishes the baseline object counts.
    if HAS_OBJGRAPH:
        objgraph.show_growth(
            limit=10, file=io.StringIO()
        )  # prime state, discard output

    for i in range(iterations):
        call_fn(i)

        if (i + 1) % snapshot_every == 0:
            gc.collect()
            snap = tracemalloc.take_snapshot()
            stats = snap.compare_to(baseline, "lineno")
            growing = [s for s in stats if s.size_diff > 0]

            print(f"\n  [iter {i + 1}/{iterations}] Allocations grown vs. baseline:")
            if growing:
                for s in growing[:6]:
                    kb = s.size_diff / 1024
                    loc = s.traceback[0]
                    print(f"    {kb:+8.1f} KB  {loc.filename}:{loc.lineno}")
            else:
                print("    (none — memory is stable)")

            if HAS_OBJGRAPH:
                print(
                    f"\n  [iter {i + 1}/{iterations}] New object types since last check:"
                )
                objgraph.show_growth(limit=5, shortnames=False)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\n  Summary — current: {current / 1024:.1f} KB, peak: {peak / 1024:.1f} KB")


# ---------------------------------------------------------------------------
# Section 1: Mock demo (no real data needed)
# ---------------------------------------------------------------------------


class _MockCacheUser:
    """
    Minimal stand-in simulating CellMapImage's cached_property array pattern.

    Each __getitem__ allocates a new array into self._array_cache, mirroring
    how CellMapImage builds an xarray DataArray on every access. With
    clear_cache=True the cache is dropped immediately (the fix); without it
    the reference accumulates.
    """

    def __init__(self, shape=(512, 512)):
        self.shape = shape
        self._array_cache = None

    def _clear_array_cache(self):
        self._array_cache = None

    def __getitem__(self, idx, clear_cache=True):
        self._array_cache = np.ones(self.shape, dtype=np.float32)
        result = self._array_cache
        if clear_cache:
            self._clear_array_cache()
        return result


def run_mock_demo(iterations):
    print("\n" + "=" * 64)
    print("SECTION 1: Mock demo (no real data, illustrates the principle)")
    print("=" * 64)

    leaky = _MockCacheUser()
    fixed = _MockCacheUser()

    profile_iters(
        "WITHOUT cache clearing (leaky)",
        lambda i: leaky.__getitem__(i, clear_cache=False),
        iterations=iterations,
    )
    profile_iters(
        "WITH cache clearing (fixed)",
        lambda i: fixed.__getitem__(i, clear_cache=True),
        iterations=iterations,
    )


# ---------------------------------------------------------------------------
# Section 2: Real CellMapImage with a temporary Zarr store
# ---------------------------------------------------------------------------


def _build_test_zarr(root_path: Path, shape=(32, 32, 32), scale=(4.0, 4.0, 4.0)):
    """Create a minimal OME-NGFF Zarr array for profiling."""
    import zarr
    from pydantic_ome_ngff.v04.axis import Axis
    from pydantic_ome_ngff.v04.multiscale import (
        Dataset as MultiscaleDataset,
        MultiscaleMetadata,
    )
    from pydantic_ome_ngff.v04.transform import VectorScale

    root_path.mkdir(parents=True, exist_ok=True)
    data = np.random.rand(*shape).astype(np.float32)
    store = zarr.DirectoryStore(str(root_path))
    root = zarr.group(store=store, overwrite=True)
    chunks = tuple(min(16, s) for s in shape)
    root.create_dataset("s0", data=data, chunks=chunks, overwrite=True)

    axes = [Axis(name=n, type="space", unit="nanometer") for n in ["z", "y", "x"]]
    datasets = (
        MultiscaleDataset(
            path="s0",
            coordinateTransformations=(VectorScale(type="scale", scale=scale),),
        ),
    )
    root.attrs["multiscales"] = [
        MultiscaleMetadata(
            version="0.4", name="test", axes=axes, datasets=datasets
        ).model_dump(mode="json", exclude_none=True)
    ]
    return str(root_path)


def run_real_demo(iterations):
    print("\n" + "=" * 64)
    print("SECTION 2: Real CellMapImage with a temporary Zarr dataset")
    print("=" * 64)

    try:
        from cellmap_data.image import CellMapImage
    except ImportError as e:
        print(f"\n  Skipping — could not import CellMapImage: {e}")
        return

    try:
        with tempfile.TemporaryDirectory() as tmp:
            # Larger array so each DataArray is meaningfully sized (~2 MB)
            shape = (64, 64, 64)
            scale = [4.0, 4.0, 4.0]
            voxel_shape = [16, 16, 16]
            img_path = _build_test_zarr(
                Path(tmp) / "raw", shape=shape, scale=tuple(scale)
            )

            # Volume spans 0–256 nm per axis; vary centers to exercise interp/reindex
            rng = np.random.default_rng(42)
            half = voxel_shape[0] * scale[0] / 2  # 32 nm margin
            lo, hi = half, shape[0] * scale[0] - half  # 32 to 224 nm

            def random_center(i):
                coords = rng.uniform(lo, hi, size=3)
                return {
                    "z": float(coords[0]),
                    "y": float(coords[1]),
                    "x": float(coords[2]),
                }

            def make_image():
                return CellMapImage(
                    path=img_path,
                    target_class="raw",
                    target_scale=scale,
                    target_voxel_shape=voxel_shape,
                    device="cpu",
                )

            # Warmup: load all heavy imports and initialize TensorStore context
            # before profiling either mode, so the comparison is not confounded
            # by import costs.
            print("\n  Warming up (pre-loading imports and TensorStore context)...")
            _warmup = make_image()
            for _ in range(5):
                _warmup[{"z": 128.0, "y": 128.0, "x": 128.0}]
            del _warmup
            gc.collect()
            print("  Done.\n")

            # Leaky first (no imports to pay), then fixed — equal footing.
            img_leaky = make_image()
            img_leaky._clear_array_cache = lambda: None
            profile_iters(
                "CellMapImage — WITHOUT cache clearing (leaky)",
                lambda i: img_leaky[random_center(i)],
                iterations=iterations,
            )

            img_fixed = make_image()
            profile_iters(
                "CellMapImage — WITH cache clearing (fixed)",
                lambda i: img_fixed[random_center(i)],
                iterations=iterations,
            )

    except Exception as e:
        print(f"\n  Error during real demo: {e}")
        raise


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    iterations = int(os.environ.get("DEMO_ITERS", "100"))

    print("=" * 64)
    print("CellMapImage Memory Profiling Demo")
    print("=" * 64)
    print(
        f"\n  iterations : {iterations} (set DEMO_ITERS env var to change)\n"
        f"  tracemalloc: built-in\n"
        f"  objgraph   : {'available' if HAS_OBJGRAPH else 'not installed — pip install objgraph'}"
    )

    run_mock_demo(iterations=iterations)
    run_real_demo(iterations=iterations)

    print("\n" + "=" * 64)
    print("Done.")
    print("=" * 64)


if __name__ == "__main__":
    main()
