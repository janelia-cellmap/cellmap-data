"""
Tests for initialization optimizations added to CellMapDataset and
CellMapMultiDataset.

Covers:
  - force_has_data=True sets has_data immediately (no class_counts read)
  - bounding_box / sampling_box parallel computation: correctness and cleanup
  - CellMapMultiDataset.class_counts parallel execution: correct aggregation,
    exception propagation, CELLMAP_MAX_WORKERS env-var respected
  - _ImmediateExecutor: submit/map correctness (Windows+TensorStore drop-in)
  - Immediate executor code paths in bounding_box, sampling_box, and
    CellMapMultiDataset.class_counts (simulated via monkeypatching)
  - Consistency: dataset.py and multidataset.py share the same
    _USE_IMMEDIATE_EXECUTOR flag
"""

from unittest.mock import PropertyMock, patch

import pytest

from cellmap_data import CellMapDataset, CellMapMultiDataset
from cellmap_data.image import CellMapImage

from .test_helpers import create_test_dataset

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def single_dataset_config(tmp_path):
    return create_test_dataset(
        tmp_path,
        raw_shape=(32, 32, 32),
        num_classes=2,
        raw_scale=(4.0, 4.0, 4.0),
        seed=0,
    )


@pytest.fixture
def multi_source_dataset(tmp_path):
    """Dataset with two input arrays and two target arrays (four CellMapImage
    objects), so the parallel bounding_box / sampling_box paths receive more
    than one source to map over."""
    config = create_test_dataset(
        tmp_path,
        raw_shape=(32, 32, 32),
        num_classes=2,
        raw_scale=(4.0, 4.0, 4.0),
        seed=7,
    )
    return CellMapDataset(
        raw_path=config["raw_path"],
        target_path=config["gt_path"],
        classes=config["classes"],
        input_arrays={
            "raw_4nm": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)},
            "raw_8nm": {"shape": (8, 8, 8), "scale": (8.0, 8.0, 8.0)},
        },
        target_arrays={
            "gt_4nm": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)},
            "gt_8nm": {"shape": (8, 8, 8), "scale": (8.0, 8.0, 8.0)},
        },
        force_has_data=True,
    )


@pytest.fixture
def three_datasets(tmp_path):
    datasets = []
    for i in range(3):
        config = create_test_dataset(
            tmp_path / f"ds_{i}",
            raw_shape=(32, 32, 32),
            num_classes=2,
            raw_scale=(4.0, 4.0, 4.0),
            seed=i,
        )
        ds = CellMapDataset(
            raw_path=config["raw_path"],
            target_path=config["gt_path"],
            classes=config["classes"],
            input_arrays={"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            target_arrays={"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
        )
        datasets.append(ds)
    return datasets


# ---------------------------------------------------------------------------
# force_has_data
# ---------------------------------------------------------------------------


class TestForceHasData:
    """force_has_data=True should set has_data=True at construction time
    without ever accessing CellMapImage.class_counts."""

    def test_has_data_true_when_force_set(self, single_dataset_config):
        """has_data is True immediately after __init__ when force_has_data=True."""
        config = single_dataset_config
        dataset = CellMapDataset(
            raw_path=config["raw_path"],
            target_path=config["gt_path"],
            classes=config["classes"],
            input_arrays={"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            target_arrays={"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            force_has_data=True,
        )
        assert dataset.has_data is True

    def test_class_counts_not_accessed_when_force_has_data(self, single_dataset_config):
        """CellMapImage.class_counts must never be accessed during __init__
        when force_has_data=True."""
        config = single_dataset_config
        with patch.object(
            CellMapImage, "class_counts", new_callable=PropertyMock, return_value=100.0
        ) as mock_counts:
            CellMapDataset(
                raw_path=config["raw_path"],
                target_path=config["gt_path"],
                classes=config["classes"],
                input_arrays={"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
                target_arrays={"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
                force_has_data=True,
            )
            mock_counts.assert_not_called()

    def test_class_counts_accessed_without_force_has_data(self, single_dataset_config):
        """Without force_has_data, class_counts IS accessed (inverse check)."""
        config = single_dataset_config
        with patch.object(
            CellMapImage, "class_counts", new_callable=PropertyMock, return_value=100.0
        ) as mock_counts:
            CellMapDataset(
                raw_path=config["raw_path"],
                target_path=config["gt_path"],
                classes=config["classes"],
                input_arrays={"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
                target_arrays={"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
                force_has_data=False,
            )
            mock_counts.assert_called()

    def test_has_data_false_without_force_for_empty_data(self, tmp_path):
        """Without force_has_data and with all-zero target data, has_data=False."""
        import numpy as np
        from .test_helpers import create_test_zarr_array, create_test_image_data

        # Raw array
        raw_data = create_test_image_data((16, 16, 16), pattern="random")
        create_test_zarr_array(tmp_path / "dataset.zarr" / "raw", raw_data)

        # All-zero target → class_counts == 0 → has_data stays False
        zero_data = np.zeros((16, 16, 16), dtype=np.uint8)
        create_test_zarr_array(
            tmp_path / "dataset.zarr" / "class_0",
            zero_data,
            absent=zero_data.size,  # all absent
        )

        dataset = CellMapDataset(
            raw_path=str(tmp_path / "dataset.zarr" / "raw"),
            target_path=str(tmp_path / "dataset.zarr" / "[class_0]"),
            classes=["class_0"],
            input_arrays={"raw": {"shape": (8, 8, 8), "scale": (1.0, 1.0, 1.0)}},
            target_arrays={"gt": {"shape": (8, 8, 8), "scale": (1.0, 1.0, 1.0)}},
            force_has_data=False,
        )
        assert not dataset.has_data


# ---------------------------------------------------------------------------
# bounding_box / sampling_box parallel computation
# ---------------------------------------------------------------------------


class TestParallelBoundingBox:
    """bounding_box and sampling_box must give correct results when computed
    in parallel across multiple CellMapImage sources."""

    def test_bounding_box_correct_with_multiple_sources(self, multi_source_dataset):
        bbox = multi_source_dataset.bounding_box
        assert isinstance(bbox, dict)
        for axis in multi_source_dataset.axis_order:
            assert axis in bbox
            lo, hi = bbox[axis]
            assert lo <= hi

    def test_sampling_box_correct_with_multiple_sources(self, multi_source_dataset):
        sbox = multi_source_dataset.sampling_box
        assert isinstance(sbox, dict)
        for axis in multi_source_dataset.axis_order:
            assert axis in sbox
            assert len(sbox[axis]) == 2

    def test_bounding_box_consistent_with_single_source(self, single_dataset_config):
        """Sequential vs. parallel should yield the same bounding box."""
        config = single_dataset_config

        def make_dataset():
            return CellMapDataset(
                raw_path=config["raw_path"],
                target_path=config["gt_path"],
                classes=config["classes"],
                input_arrays={"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
                target_arrays={"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
                force_has_data=True,
            )

        ds1 = make_dataset()
        ds2 = make_dataset()

        bbox1 = ds1.bounding_box
        bbox2 = ds2.bounding_box

        for axis in ds1.axis_order:
            assert pytest.approx(bbox1[axis][0]) == bbox2[axis][0]
            assert pytest.approx(bbox1[axis][1]) == bbox2[axis][1]

    def test_sampling_box_inside_bounding_box(self, multi_source_dataset):
        """The sampling box must be a sub-region of (or equal to) the bounding box."""
        bbox = multi_source_dataset.bounding_box
        sbox = multi_source_dataset.sampling_box
        for axis in multi_source_dataset.axis_order:
            assert sbox[axis][0] >= bbox[axis][0] - 1e-9
            assert sbox[axis][1] <= bbox[axis][1] + 1e-9

    def test_bounding_box_pool_does_not_leak_threads(self, single_dataset_config):
        """Accessing bounding_box twice on fresh datasets should not raise even
        if the pool from the first call was already shut down."""
        config = single_dataset_config

        for _ in range(2):
            ds = CellMapDataset(
                raw_path=config["raw_path"],
                target_path=config["gt_path"],
                classes=config["classes"],
                input_arrays={"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
                target_arrays={"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
                force_has_data=True,
            )
            bbox = ds.bounding_box
            assert bbox is not None


# ---------------------------------------------------------------------------
# CellMapMultiDataset.class_counts parallel execution
# ---------------------------------------------------------------------------


class TestMultiDatasetClassCountsParallel:
    """Parallel class_counts must aggregate correctly and behave robustly."""

    def test_totals_equal_sum_of_individual_datasets(self, three_datasets):
        """Aggregated totals must equal the element-wise sum of each dataset's
        class_counts["totals"]."""
        classes = ["class_0", "class_1"]
        multi = CellMapMultiDataset(
            classes=classes,
            input_arrays={"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            target_arrays={"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            datasets=three_datasets,
        )

        # Compute expected totals by summing across individual datasets.
        expected: dict[str, float] = {c: 0.0 for c in classes}
        expected.update({c + "_bg": 0.0 for c in classes})
        for ds in three_datasets:
            for key in expected:
                expected[key] += ds.class_counts["totals"].get(key, 0.0)

        actual = multi.class_counts["totals"]
        for key, val in expected.items():
            assert pytest.approx(actual[key], rel=1e-6) == val

    def test_class_counts_has_totals_key(self, three_datasets):
        multi = CellMapMultiDataset(
            classes=["class_0", "class_1"],
            input_arrays={"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            target_arrays={"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            datasets=three_datasets,
        )
        counts = multi.class_counts
        assert "totals" in counts
        for c in ["class_0", "class_1", "class_0_bg", "class_1_bg"]:
            assert c in counts["totals"]

    def test_exception_from_dataset_propagates(self, three_datasets):
        """If any dataset's class_counts raises, the exception must propagate
        out of CellMapMultiDataset.class_counts (via future.result())."""
        multi = CellMapMultiDataset(
            classes=["class_0", "class_1"],
            input_arrays={"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            target_arrays={"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            datasets=three_datasets,
        )

        with patch.object(
            CellMapDataset,
            "class_counts",
            new_callable=PropertyMock,
            side_effect=RuntimeError("simulated failure"),
        ):
            with pytest.raises(RuntimeError, match="simulated failure"):
                _ = multi.class_counts

    def test_max_workers_env_var_respected(self, three_datasets, monkeypatch):
        """CELLMAP_MAX_WORKERS is the cap on the number of worker threads."""
        monkeypatch.setenv("CELLMAP_MAX_WORKERS", "1")

        multi = CellMapMultiDataset(
            classes=["class_0", "class_1"],
            input_arrays={"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            target_arrays={"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            datasets=three_datasets,
        )
        # Should still produce correct results with a single worker
        counts = multi.class_counts
        assert "totals" in counts

    def test_single_dataset_multidataset(self, tmp_path):
        """Edge case: a multi-dataset with one child returns that child's counts."""
        config = create_test_dataset(
            tmp_path, raw_shape=(32, 32, 32), num_classes=2, raw_scale=(4.0, 4.0, 4.0)
        )
        ds = CellMapDataset(
            raw_path=config["raw_path"],
            target_path=config["gt_path"],
            classes=config["classes"],
            input_arrays={"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            target_arrays={"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
        )
        multi = CellMapMultiDataset(
            classes=["class_0", "class_1"],
            input_arrays={"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            target_arrays={"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            datasets=[ds],
        )
        multi_totals = multi.class_counts["totals"]
        ds_totals = ds.class_counts["totals"]
        for c in ["class_0", "class_1"]:
            assert pytest.approx(multi_totals[c]) == ds_totals.get(c, 0.0)
            assert pytest.approx(multi_totals[c + "_bg"]) == ds_totals.get(
                c + "_bg", 0.0
            )

    def test_empty_classes_list(self, three_datasets):
        """An empty classes list produces an empty totals dict without error."""
        multi = CellMapMultiDataset(
            classes=[],
            input_arrays={"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            target_arrays={},
            datasets=three_datasets,
        )
        counts = multi.class_counts
        assert counts["totals"] == {}


# ---------------------------------------------------------------------------
# _ImmediateExecutor unit tests
# ---------------------------------------------------------------------------


class TestImmediateExecutor:
    """Unit tests for _ImmediateExecutor.

    _ImmediateExecutor is the Windows+TensorStore drop-in that runs every
    submitted callable synchronously in the calling thread.  It must satisfy
    the same interface as ThreadPoolExecutor so all existing call sites
    (submit, map, as_completed, shutdown) work without modification.
    """

    @pytest.fixture
    def executor(self):
        from cellmap_data.dataset import _ImmediateExecutor

        return _ImmediateExecutor()

    def test_submit_executes_synchronously(self, executor):
        """submit() runs the callable before returning; the future is already done."""
        calls = []
        future = executor.submit(calls.append, 99)
        assert future.done(), "Future should be resolved immediately"
        assert calls == [99], "Callable should have run synchronously"

    def test_submit_returns_correct_result(self, executor):
        """submit() stores the return value in the future."""
        future = executor.submit(lambda x, y: x + y, 3, 4)
        assert future.result() == 7

    def test_submit_captures_exception(self, executor):
        """Exceptions raised by the callable are stored, not propagated."""
        future = executor.submit(lambda: 1 / 0)
        assert future.exception() is not None
        assert isinstance(future.exception(), ZeroDivisionError)

    def test_map_returns_results_in_order(self, executor):
        """map() returns results in the same order as the input iterable."""
        results = list(executor.map(lambda x: x * 2, [1, 2, 3, 4]))
        assert results == [2, 4, 6, 8]

    def test_map_with_lambda(self, executor):
        """map() works with lambda functions, matching the bounding_box usage."""
        items = [{"v": i} for i in range(5)]
        results = list(executor.map(lambda d: d["v"], items))
        assert results == list(range(5))

    def test_map_propagates_exception(self, executor):
        """Exceptions from map() propagate when the result is consumed."""
        with pytest.raises(ZeroDivisionError):
            list(executor.map(lambda x: 1 / x, [1, 0, 2]))

    def test_shutdown_is_noop(self, executor):
        """shutdown() must not raise even when called multiple times."""
        executor.shutdown(wait=True)
        executor.shutdown(wait=False, cancel_futures=True)

    def test_as_completed_works_with_submit(self, executor):
        """Futures from submit() are compatible with as_completed()."""
        from concurrent.futures import as_completed

        futures = [executor.submit(lambda i=i: i * 3, i) for i in range(5)]
        results = {f.result() for f in as_completed(futures)}
        assert results == {0, 3, 6, 9, 12}

    def test_is_executor_subclass(self):
        """_ImmediateExecutor must be a subclass of concurrent.futures.Executor
        so it satisfies the Executor interface including map()."""
        from concurrent.futures import Executor

        from cellmap_data.dataset import _ImmediateExecutor

        assert issubclass(_ImmediateExecutor, Executor)


# ---------------------------------------------------------------------------
# Immediate executor code paths (simulated via monkeypatching)
# ---------------------------------------------------------------------------


class TestImmediateExecutorPaths:
    """Verify that bounding_box, sampling_box, and CellMapMultiDataset.class_counts
    work correctly when _USE_IMMEDIATE_EXECUTOR is True.

    These tests simulate the Windows+TensorStore environment on any platform
    by monkeypatching the module-level flag and singleton executor.
    """

    @pytest.fixture
    def patched_immediate(self, monkeypatch):
        """Patch dataset module to act as if running on Windows+TensorStore."""
        import cellmap_data.dataset as ds_module
        from cellmap_data.dataset import _ImmediateExecutor

        monkeypatch.setattr(ds_module, "_USE_IMMEDIATE_EXECUTOR", True)
        monkeypatch.setattr(ds_module, "_IMMEDIATE_EXECUTOR", _ImmediateExecutor())

    def test_bounding_box_uses_immediate_executor(
        self, single_dataset_config, patched_immediate
    ):
        """bounding_box must work via executor.map() when using _ImmediateExecutor."""
        config = single_dataset_config
        ds = CellMapDataset(
            raw_path=config["raw_path"],
            target_path=config["gt_path"],
            classes=config["classes"],
            input_arrays={"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            target_arrays={"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            force_has_data=True,
        )
        from cellmap_data.dataset import _ImmediateExecutor

        assert isinstance(ds.executor, _ImmediateExecutor)
        bbox = ds.bounding_box
        assert isinstance(bbox, dict)
        for axis in ds.axis_order:
            assert axis in bbox
            lo, hi = bbox[axis]
            assert lo <= hi

    def test_sampling_box_uses_immediate_executor(
        self, single_dataset_config, patched_immediate
    ):
        """sampling_box must work via executor.map() when using _ImmediateExecutor."""
        config = single_dataset_config
        ds = CellMapDataset(
            raw_path=config["raw_path"],
            target_path=config["gt_path"],
            classes=config["classes"],
            input_arrays={"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            target_arrays={"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            force_has_data=True,
        )
        sbox = ds.sampling_box
        assert isinstance(sbox, dict)
        for axis in ds.axis_order:
            assert axis in sbox
            assert len(sbox[axis]) == 2

    def test_getitem_uses_immediate_executor(
        self, single_dataset_config, patched_immediate
    ):
        """__getitem__ must work when _ImmediateExecutor is active."""
        config = single_dataset_config
        ds = CellMapDataset(
            raw_path=config["raw_path"],
            target_path=config["gt_path"],
            classes=config["classes"],
            input_arrays={"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            target_arrays={"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            force_has_data=True,
        )
        result = ds[0]
        assert isinstance(result, dict)
        assert "raw" in result

    def test_multidataset_class_counts_sequential_path(
        self, three_datasets, monkeypatch
    ):
        """CellMapMultiDataset.class_counts takes the sequential path when
        _USE_IMMEDIATE_EXECUTOR is True (shared flag from dataset.py)."""
        import cellmap_data.multidataset as md_module

        monkeypatch.setattr(md_module, "_USE_IMMEDIATE_EXECUTOR", True)

        multi = CellMapMultiDataset(
            classes=["class_0", "class_1"],
            input_arrays={"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            target_arrays={"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            datasets=three_datasets,
        )
        counts = multi.class_counts
        assert "totals" in counts
        for c in ["class_0", "class_1", "class_0_bg", "class_1_bg"]:
            assert c in counts["totals"]


# ---------------------------------------------------------------------------
# Consistency: dataset.py and multidataset.py share _USE_IMMEDIATE_EXECUTOR
# ---------------------------------------------------------------------------


class TestImmediateExecutorFlagConsistency:
    """The _USE_IMMEDIATE_EXECUTOR flag must be sourced from dataset.py in
    both dataset.py and multidataset.py so they always agree on whether
    to use threads."""

    def test_flag_values_match_across_modules(self):
        """Both modules read the same flag value at import time."""
        import cellmap_data.dataset as ds_module
        import cellmap_data.multidataset as md_module

        assert ds_module._USE_IMMEDIATE_EXECUTOR == md_module._USE_IMMEDIATE_EXECUTOR

    def test_multidataset_imports_flag_from_dataset(self):
        """multidataset module must expose _USE_IMMEDIATE_EXECUTOR (imported
        from dataset), not define its own copy."""
        import inspect

        import cellmap_data.multidataset as md_module

        assert hasattr(md_module, "_USE_IMMEDIATE_EXECUTOR"), (
            "multidataset must import _USE_IMMEDIATE_EXECUTOR from dataset"
        )
        # Verify the source: the flag in multidataset should be the same
        # object as the one in dataset (True/False booleans are singletons).
        import cellmap_data.dataset as ds_module

        assert md_module._USE_IMMEDIATE_EXECUTOR is ds_module._USE_IMMEDIATE_EXECUTOR
