"""
Stress test for concurrent TensorStore reads and the read limiter.

Verifies that the TensorStore read limiter (read_limiter.py) prevents crashes
under high concurrency on Windows and does not cause deadlocks or correctness
issues on any other platform.

On Windows, running many concurrent __getitem__ calls without the limiter
triggers native hard-crashes (abort / SEH) inside TensorStore.  This test
catches those regressions as non-zero exit codes in CI.

On Linux/macOS the limiter is a no-op, but the concurrency tests still
exercise the same code paths and would expose deadlocks introduced in the
limiter itself.
"""

import os
import platform
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from typing import List

import pytest

from cellmap_data import CellMapDataset
from cellmap_data.read_limiter import (
    MAX_CONCURRENT_READS,
    _read_semaphore,
    limit_tensorstore_reads,
)

from .test_helpers import create_minimal_test_dataset, create_test_dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextmanager
def collect_worker_errors():
    """Context manager for collecting errors from concurrent workers.

    Yields:
        List[Exception]: Empty list that workers can append errors to.

    Raises:
        AssertionError: If any errors were collected during execution.
    """
    errors: List[Exception] = []
    try:
        yield errors
    finally:
        assert not errors, f"{len(errors)} errors occurred: {errors[:3]}"


_IS_WINDOWS = platform.system() == "Windows"
_IS_TENSORSTORE = (
    os.environ.get("CELLMAP_DATA_BACKEND", "tensorstore").lower() == "tensorstore"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def stress_dataset_config(tmp_path):
    """Dataset with multiple classes, suitable for heavy concurrent access."""
    return create_test_dataset(
        tmp_path,
        raw_shape=(32, 32, 32),
        num_classes=3,
        raw_scale=(4.0, 4.0, 4.0),
    )


@pytest.fixture
def stress_dataset(stress_dataset_config):
    """CellMapDataset configured for concurrent stress testing."""
    config = stress_dataset_config
    input_arrays = {"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}}
    target_arrays = {"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}}

    return CellMapDataset(
        raw_path=config["raw_path"],
        target_path=config["gt_path"],
        classes=config["classes"],
        input_arrays=input_arrays,
        target_arrays=target_arrays,
        is_train=True,
        force_has_data=True,
    )


@pytest.fixture
def raw_only_stress_dataset(tmp_path):
    """raw_only CellMapDataset for testing the raw-only target read path."""
    config = create_minimal_test_dataset(tmp_path)
    input_arrays = {"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}}
    target_arrays = {"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}}

    return CellMapDataset(
        raw_path=config["raw_path"],
        target_path=config["raw_path"],  # raw as target too → raw_only path
        classes=None,
        input_arrays=input_arrays,
        target_arrays=target_arrays,
        is_train=True,
        force_has_data=True,
    )


# ---------------------------------------------------------------------------
# Unit tests for the read_limiter module itself
# ---------------------------------------------------------------------------


class TestReadLimiterUnit:
    """Unit tests for cellmap_data.read_limiter."""

    def test_semaphore_state_matches_platform(self):
        """Semaphore is active on Windows+tensorstore; None elsewhere."""
        if _IS_WINDOWS and _IS_TENSORSTORE:
            assert (
                _read_semaphore is not None
            ), "Expected a semaphore on Windows+TensorStore"
            assert isinstance(MAX_CONCURRENT_READS, int)
            assert MAX_CONCURRENT_READS >= 1
        else:
            assert (
                _read_semaphore is None
            ), "Expected no semaphore on non-Windows or non-TensorStore"
            assert MAX_CONCURRENT_READS is None

    def test_env_override_respected(self):
        """CELLMAP_MAX_CONCURRENT_READS is reflected in MAX_CONCURRENT_READS."""
        if _IS_WINDOWS and _IS_TENSORSTORE:
            # The env var was read at import time; just verify the value is sane.
            expected = int(os.environ.get("CELLMAP_MAX_CONCURRENT_READS", "1"))
            assert MAX_CONCURRENT_READS == expected

    def test_context_manager_completes_without_error(self):
        """A single entry/exit of limit_tensorstore_reads() does not raise."""
        with limit_tensorstore_reads():
            pass

    def test_context_manager_reraises_exceptions(self):
        """Exceptions inside the context manager propagate and release the lock."""
        with pytest.raises(RuntimeError, match="boom"):
            with limit_tensorstore_reads():
                raise RuntimeError("boom")

        # Semaphore must be released: a second entry must not block.
        acquired = threading.Event()

        def try_acquire():
            with limit_tensorstore_reads():
                acquired.set()

        t = threading.Thread(target=try_acquire)
        t.start()
        t.join(timeout=5)
        assert acquired.is_set(), "Semaphore was not released after exception"

    def test_concurrent_access_does_not_deadlock(self):
        """50 threads entering the context manager concurrently must all finish."""
        errors: list[Exception] = []
        finished = threading.Barrier(50)

        def task():
            try:
                with limit_tensorstore_reads():
                    finished.wait(timeout=30)
            except Exception as exc:  # BrokenBarrierError counts too
                errors.append(exc)

        threads = [threading.Thread(target=task) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        assert not errors, f"Errors during concurrent access: {errors}"


# ---------------------------------------------------------------------------
# Dataset close() and atexit integration
# ---------------------------------------------------------------------------


class TestExecutorLifecycle:
    """Tests for the close() method and atexit registration."""

    def test_close_shuts_down_executor(self, stress_dataset):
        """close() shuts down the executor and sets it to None."""
        # Force executor creation
        _ = stress_dataset.executor
        assert stress_dataset._executor is not None

        stress_dataset.close()
        assert stress_dataset._executor is None

    def test_close_is_idempotent(self, stress_dataset):
        """Calling close() multiple times does not raise."""
        stress_dataset.close()
        stress_dataset.close()  # second call must be safe

    def test_getitem_after_close_recreates_executor(self, stress_dataset):
        """After close(), __getitem__ can still run (executor is re-created)."""
        stress_dataset.close()
        assert stress_dataset._executor is None

        # __getitem__ internally accesses .executor which lazily re-creates it
        result = stress_dataset[0]
        assert result is not None
        assert stress_dataset._executor is not None


# ---------------------------------------------------------------------------
# __getitem__ stress tests
# ---------------------------------------------------------------------------


def _make_stress_dataset(config: dict) -> CellMapDataset:
    """Create a fresh CellMapDataset from a config dict (for per-thread use)."""
    input_arrays = {"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}}
    target_arrays = {"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}}
    return CellMapDataset(
        raw_path=config["raw_path"],
        target_path=config["gt_path"],
        classes=config["classes"],
        input_arrays=input_arrays,
        target_arrays=target_arrays,
        is_train=True,
        force_has_data=True,
    )


class TestConcurrentGetitem:
    """Stress tests for dataset.__getitem__ under sustained load.

    Design note on concurrency model
    ---------------------------------
    CellMapDataset uses an *internal* ThreadPoolExecutor to parallelize
    per-array and per-label reads within a single ``__getitem__`` call.
    The real DataLoader usage pattern is:

    * ``num_workers=0``  → the main process calls ``__getitem__`` sequentially;
      the dataset's internal pool handles within-item parallelism.
    * ``num_workers>0``  → each worker *process* gets its own pickle-restored
      copy of the dataset (and therefore its own executor); calls are still
      sequential within each worker.

    Sharing one dataset instance across multiple threads that each call
    ``__getitem__`` simultaneously is NOT how DataLoader uses the dataset and
    creates a deadlock: ``get_target_array`` (running in a worker slot) blocks
    waiting for ``get_label_array`` sub-futures on the same pool, starving
    those sub-futures of worker slots.

    The concurrent tests below therefore give each outer thread its own dataset
    instance, accurately simulating ``num_workers>0`` DataLoader workers.  The
    TensorStore read limiter is still exercised because ``limit_tensorstore_reads``
    uses a *process-wide* semaphore, shared across all threads (and datasets) in
    the same process.
    """

    NUM_ITERATIONS_PER_THREAD = 50  # iterations each "worker" runs
    NUM_OUTER_THREADS = 4  # simulated DataLoader num_workers

    # ------------------------------------------------------------------
    # serial baseline
    # ------------------------------------------------------------------

    def test_serial_getitem_with_classes(self, stress_dataset):
        """Sequential __getitem__ calls (multi-class) complete without error."""
        n = min(200, len(stress_dataset))
        for i in range(n):
            result = stress_dataset[i % len(stress_dataset)]
            assert result is not None
            assert "raw" in result
            assert "gt" in result

    def test_serial_getitem_raw_only(self, raw_only_stress_dataset):
        """Sequential __getitem__ calls (raw-only) complete without error."""
        ds = raw_only_stress_dataset
        n = min(200, len(ds))
        for i in range(n):
            result = ds[i % len(ds)]
            assert result is not None

    # ------------------------------------------------------------------
    # concurrent stress — each thread owns its dataset (mirrors DataLoader workers)
    # ------------------------------------------------------------------

    def test_concurrent_workers_with_classes(self, stress_dataset_config):
        """Multiple simulated DataLoader workers (each with its own dataset) run concurrently."""
        with collect_worker_errors() as errors:

            def worker(thread_id: int) -> None:
                ds = _make_stress_dataset(stress_dataset_config)
                try:
                    n = min(self.NUM_ITERATIONS_PER_THREAD, len(ds))
                    for i in range(n):
                        result = ds[i % len(ds)]
                        if result is None:
                            errors.append(
                                RuntimeError(f"thread {thread_id}: got None at idx {i}")
                            )
                except Exception as exc:
                    errors.append(exc)
                finally:
                    ds.close()

            threads = [
                threading.Thread(target=worker, args=(tid,))
                for tid in range(self.NUM_OUTER_THREADS)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=120)

            alive = [t for t in threads if t.is_alive()]
            assert (
                not alive
            ), f"{len(alive)} threads are still alive (possible deadlock)"

    def test_concurrent_workers_raw_only(self, tmp_path):
        """Multiple simulated workers with raw-only datasets run concurrently."""
        with collect_worker_errors() as errors:
            # Build a shared config for the raw-only variant
            config = create_minimal_test_dataset(tmp_path)
            input_arrays = {"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}}
            target_arrays = {"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}}

            def worker(thread_id: int) -> None:
                ds = CellMapDataset(
                    raw_path=config["raw_path"],
                    target_path=config["raw_path"],
                    classes=None,
                    input_arrays=input_arrays,
                    target_arrays=target_arrays,
                    is_train=True,
                    force_has_data=True,
                )
                try:
                    n = min(self.NUM_ITERATIONS_PER_THREAD, len(ds))
                    for i in range(n):
                        ds[i % len(ds)]
                except Exception as exc:
                    errors.append(exc)
                finally:
                    ds.close()

            threads = [
                threading.Thread(target=worker, args=(tid,))
                for tid in range(self.NUM_OUTER_THREADS)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=120)

            alive = [t for t in threads if t.is_alive()]
            assert (
                not alive
            ), f"{len(alive)} threads are still alive (possible deadlock)"

    @pytest.mark.skipif(
        not _IS_WINDOWS,
        reason="Windows-specific crash regression test; skipped on non-Windows",
    )
    def test_windows_high_concurrency_no_crash(self, stress_dataset_config):
        """
        Windows-specific: many simulated workers must not hard-crash the process.

        Each worker has its own dataset (matching DataLoader num_workers behavior).
        A native TensorStore abort appears as a non-zero pytest exit code, which
        CI catches even without a Python exception being raised.
        """
        num_workers = 8
        iters = 100

        with collect_worker_errors() as errors:

            def worker(thread_id: int) -> None:
                ds = _make_stress_dataset(stress_dataset_config)
                try:
                    for i in range(iters):
                        ds[i % len(ds)]
                except Exception as exc:
                    errors.append(exc)
                finally:
                    ds.close()

            threads = [
                threading.Thread(target=worker, args=(tid,))
                for tid in range(num_workers)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=300)

            alive = [t for t in threads if t.is_alive()]
            assert not alive, f"{len(alive)} threads still alive"
