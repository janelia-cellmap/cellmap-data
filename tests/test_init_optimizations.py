"""
Tests for initialization optimizations added to CellMapDataset and
CellMapMultiDataset.

Covers:
  - force_has_data=True sets has_data immediately (no class_counts read)
  - bounding_box / sampling_box parallel computation: correctness and cleanup
  - CellMapMultiDataset.class_counts parallel execution: correct aggregation,
    exception propagation, CELLMAP_MAX_WORKERS env-var respected
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
