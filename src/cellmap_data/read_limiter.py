"""
Global TensorStore read limiter for Windows crash prevention.

On Windows, concurrent TensorStore materializations from multiple threads
(triggered by source[center], .interp, ._TensorStoreAdapter.__array__, etc.)
cause native hard crashes / aborts. This module provides a semaphore-backed
context manager that serializes those reads on Windows+TensorStore while
acting as a no-op on all other platforms.

Configuration
-------------
CELLMAP_DATA_BACKEND : str
    Set to "tensorstore" (default) to enable the limiter on Windows.
    Set to anything else (e.g. "dask") to disable it entirely.

CELLMAP_MAX_CONCURRENT_READS : int
    Maximum concurrent TensorStore reads allowed on Windows.
    Defaults to 1 (fully serialized). Increase cautiously.

Notes
-----
Both environment variables must be set **before** this module is imported,
as the semaphore is created once at import time.
"""

import os
import platform
import threading
from contextlib import contextmanager

_IS_WINDOWS = platform.system() == "Windows"
_IS_TENSORSTORE = (
    os.environ.get("CELLMAP_DATA_BACKEND", "tensorstore").lower() == "tensorstore"
)

MAX_CONCURRENT_READS: int | None
_read_semaphore: threading.Semaphore | None

if _IS_WINDOWS and _IS_TENSORSTORE:
    MAX_CONCURRENT_READS = int(os.environ.get("CELLMAP_MAX_CONCURRENT_READS", "1"))
    _read_semaphore = threading.Semaphore(MAX_CONCURRENT_READS)
else:
    MAX_CONCURRENT_READS = None
    _read_semaphore = None


@contextmanager
def limit_tensorstore_reads():
    """Context manager that gates TensorStore reads on Windows.

    On Windows with the TensorStore backend, at most ``MAX_CONCURRENT_READS``
    threads may be inside this context at once.  On all other platforms (or
    when using the Dask backend) this is a true no-op with zero overhead.

    Usage
    -----
    ::

        with limit_tensorstore_reads():
            array = source[center]          # the unsafe read
        # torch-only work continues here unconstrained
    """
    if _read_semaphore is not None:
        _read_semaphore.acquire()
        try:
            yield
        finally:
            _read_semaphore.release()
    else:
        yield
