"""
Tests for CellMapDataLoader class.

Tests data loading, batching, and optimization features using real data.
"""

import pytest
import tensorstore as ts
import torch

from cellmap_data import CellMapDataLoader, CellMapDataset, CellMapMultiDataset
from .test_helpers import create_test_dataset


class TestCellMapDataLoader:
    """Test suite for CellMapDataLoader class."""

    @pytest.fixture
    def test_dataset(self, tmp_path):
        """Create a test dataset for loader tests."""
        config = create_test_dataset(
            tmp_path,
            raw_shape=(32, 32, 32),
            num_classes=2,
            raw_scale=(4.0, 4.0, 4.0),
        )

        input_arrays = {
            "raw": {
                "shape": (16, 16, 16),
                "scale": (4.0, 4.0, 4.0),
            }
        }

        target_arrays = {
            "gt": {
                "shape": (16, 16, 16),
                "scale": (4.0, 4.0, 4.0),
            }
        }

        dataset = CellMapDataset(
            raw_path=config["raw_path"],
            target_path=config["gt_path"],
            classes=config["classes"],
            input_arrays=input_arrays,
            target_arrays=target_arrays,
            is_train=True,
            force_has_data=True,
            # Force dataset to have data for testing
        )

        return dataset

    def test_initialization_basic(self, test_dataset):
        """Test basic DataLoader initialization."""
        loader = CellMapDataLoader(
            test_dataset,
            batch_size=2,
            num_workers=0,  # Use 0 for testing
        )

        assert loader is not None
        assert loader.batch_size == 2

    def test_batch_size_parameter(self, test_dataset):
        """Test different batch sizes."""
        for batch_size in [1, 2, 4, 8]:
            loader = CellMapDataLoader(
                test_dataset,
                batch_size=batch_size,
                num_workers=0,
            )
            assert loader.batch_size == batch_size

    def test_num_workers_parameter(self, test_dataset):
        """Test num_workers parameter."""
        for num_workers in [0, 1, 2]:
            loader = CellMapDataLoader(
                test_dataset,
                batch_size=2,
                num_workers=num_workers,
            )
            # Loader should be created successfully
            assert loader is not None

    def test_weighted_sampler_parameter(self, test_dataset):
        """Test weighted sampler option."""
        # With weighted sampler
        loader_weighted = CellMapDataLoader(
            test_dataset,
            batch_size=2,
            weighted_sampler=True,
            num_workers=0,
        )
        assert loader_weighted is not None

        # Without weighted sampler
        loader_no_weight = CellMapDataLoader(
            test_dataset,
            batch_size=2,
            weighted_sampler=False,
            num_workers=0,
        )
        assert loader_no_weight is not None

    def test_is_train_parameter(self, test_dataset):
        """Test is_train parameter."""
        # Training loader
        train_loader = CellMapDataLoader(
            test_dataset,
            batch_size=2,
            is_train=True,
            force_has_data=True,
            num_workers=0,
        )
        assert train_loader is not None

        # Validation loader
        val_loader = CellMapDataLoader(
            test_dataset,
            batch_size=2,
            is_train=False,
            force_has_data=True,
            num_workers=0,
        )
        assert val_loader is not None

    def test_device_parameter(self, test_dataset):
        """Test device parameter."""
        loader_cpu = CellMapDataLoader(
            test_dataset,
            batch_size=2,
            device="cpu",
            num_workers=0,
        )
        assert loader_cpu is not None

    def test_pin_memory_parameter(self, test_dataset):
        """Test pin_memory parameter."""
        loader = CellMapDataLoader(
            test_dataset,
            batch_size=2,
            pin_memory=True,
            num_workers=0,
        )
        assert loader is not None

    def test_persistent_workers_parameter(self, test_dataset):
        """Test persistent_workers parameter."""
        # Only works with num_workers > 0
        loader = CellMapDataLoader(
            test_dataset,
            batch_size=2,
            num_workers=1,
            persistent_workers=True,
        )
        assert loader is not None

    def test_prefetch_factor_parameter(self, test_dataset):
        """Test prefetch_factor parameter."""
        # Only works with num_workers > 0
        for prefetch in [2, 4, 8]:
            loader = CellMapDataLoader(
                test_dataset,
                batch_size=2,
                num_workers=1,
                prefetch_factor=prefetch,
            )
            assert loader is not None

    def test_iterations_per_epoch_parameter(self, test_dataset):
        """Test iterations_per_epoch parameter."""
        loader = CellMapDataLoader(
            test_dataset,
            batch_size=2,
            iterations_per_epoch=10,
            num_workers=0,
        )
        assert loader is not None

    def test_shuffle_parameter(self, test_dataset):
        """Test shuffle parameter."""
        # With shuffle
        loader_shuffle = CellMapDataLoader(
            test_dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0,
        )
        assert loader_shuffle is not None

        # Without shuffle
        loader_no_shuffle = CellMapDataLoader(
            test_dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
        )
        assert loader_no_shuffle is not None

    def test_drop_last_parameter(self, test_dataset):
        """Test drop_last parameter."""
        loader = CellMapDataLoader(
            test_dataset,
            batch_size=3,
            drop_last=True,
            num_workers=0,
        )
        assert loader is not None

    def test_timeout_parameter(self, test_dataset):
        """Test timeout parameter."""
        loader = CellMapDataLoader(
            test_dataset,
            batch_size=2,
            num_workers=1,
            timeout=30,
        )
        assert loader is not None


class TestDataLoaderOperations:
    """Test DataLoader operations and functionality."""

    @pytest.fixture
    def simple_loader(self, tmp_path):
        """Create a simple loader for operation tests."""
        config = create_test_dataset(
            tmp_path,
            raw_shape=(24, 24, 24),
            num_classes=2,
            raw_scale=(4.0, 4.0, 4.0),
        )

        input_arrays = {"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}}
        target_arrays = {"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}}

        dataset = CellMapDataset(
            raw_path=config["raw_path"],
            target_path=config["gt_path"],
            classes=config["classes"],
            input_arrays=input_arrays,
            target_arrays=target_arrays,
        )

        print(config)
        assert len(dataset) > 0

        return CellMapDataLoader(dataset, batch_size=2, num_workers=0)

    def test_length(self, simple_loader):
        """Test that loader has a length."""
        # Loader should implement __len__
        length = len(simple_loader)
        assert isinstance(length, int)
        assert length > 0

    def test_device_transfer(self, simple_loader):
        """Test transferring loader to device."""
        # Test CPU transfer
        loader_cpu = simple_loader.to("cpu")
        assert loader_cpu is not None

    def test_non_blocking_transfer(self, simple_loader):
        """Test non-blocking device transfer."""
        loader = simple_loader.to("cpu", non_blocking=True)
        assert loader is not None


class TestDataLoaderIntegration:
    """Integration tests for DataLoader with datasets."""

    def test_loader_with_transforms(self, tmp_path):
        """Test loader with dataset that has transforms."""
        import torchvision.transforms.v2 as T

        from cellmap_data.transforms import Binarize

        config = create_test_dataset(
            tmp_path,
            raw_shape=(32, 32, 32),
            num_classes=2,
        )

        input_arrays = {"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}}
        target_arrays = {"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}}

        raw_transforms = T.Compose([T.ToDtype(torch.float, scale=True)])
        target_transforms = T.Compose([Binarize(threshold=0.5)])

        dataset = CellMapDataset(
            raw_path=config["raw_path"],
            target_path=config["gt_path"],
            classes=config["classes"],
            input_arrays=input_arrays,
            target_arrays=target_arrays,
            raw_value_transforms=raw_transforms,
            target_value_transforms=target_transforms,
        )

        loader = CellMapDataLoader(dataset, batch_size=2, num_workers=0)
        assert loader is not None

    def test_loader_with_spatial_transforms(self, tmp_path):
        """Test loader with spatial transforms."""
        config = create_test_dataset(
            tmp_path,
            raw_shape=(32, 32, 32),
            num_classes=2,
        )

        input_arrays = {"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}}
        target_arrays = {"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}}

        spatial_transforms = {
            "mirror": {"axes": {"x": 0.5}},
            "rotate": {"axes": {"z": [-30, 30]}},
        }

        dataset = CellMapDataset(
            raw_path=config["raw_path"],
            target_path=config["gt_path"],
            classes=config["classes"],
            input_arrays=input_arrays,
            target_arrays=target_arrays,
            spatial_transforms=spatial_transforms,
            is_train=True,
            force_has_data=True,
        )

        loader = CellMapDataLoader(dataset, batch_size=2, num_workers=0)
        assert loader is not None

    def test_loader_reproducibility(self, tmp_path):
        """Test loader reproducibility with fixed seed."""
        config = create_test_dataset(
            tmp_path,
            raw_shape=(24, 24, 24),
            num_classes=2,
            seed=42,
        )

        input_arrays = {"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}}
        target_arrays = {"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}}

        # Create two loaders with same seed
        torch.manual_seed(42)
        dataset1 = CellMapDataset(
            raw_path=config["raw_path"],
            target_path=config["gt_path"],
            classes=config["classes"],
            input_arrays=input_arrays,
            target_arrays=target_arrays,
        )
        loader1 = CellMapDataLoader(dataset1, batch_size=2, num_workers=0)

        torch.manual_seed(42)
        dataset2 = CellMapDataset(
            raw_path=config["raw_path"],
            target_path=config["gt_path"],
            classes=config["classes"],
            input_arrays=input_arrays,
            target_arrays=target_arrays,
        )
        loader2 = CellMapDataLoader(dataset2, batch_size=2, num_workers=0)

        # Both loaders should be created successfully
        assert loader1 is not None
        assert loader2 is not None

    def test_multiple_loaders_same_dataset(self, tmp_path):
        """Test multiple loaders for same dataset."""
        config = create_test_dataset(
            tmp_path,
            raw_shape=(32, 32, 32),
            num_classes=2,
        )

        input_arrays = {"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}}
        target_arrays = {"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}}

        dataset = CellMapDataset(
            raw_path=config["raw_path"],
            target_path=config["gt_path"],
            classes=config["classes"],
            input_arrays=input_arrays,
            target_arrays=target_arrays,
        )

        # Create multiple loaders
        loader1 = CellMapDataLoader(dataset, batch_size=2, num_workers=0)
        loader2 = CellMapDataLoader(dataset, batch_size=4, num_workers=0)

        assert loader1.batch_size == 2
        assert loader2.batch_size == 4

    def test_loader_memory_optimization(self, tmp_path):
        """Test memory optimization settings."""
        config = create_test_dataset(
            tmp_path,
            raw_shape=(32, 32, 32),
            num_classes=2,
        )

        input_arrays = {"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}}
        target_arrays = {"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}}

        dataset = CellMapDataset(
            raw_path=config["raw_path"],
            target_path=config["gt_path"],
            classes=config["classes"],
            input_arrays=input_arrays,
            target_arrays=target_arrays,
        )

        # Test with memory optimization settings
        loader = CellMapDataLoader(
            dataset,
            batch_size=2,
            num_workers=1,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
        )

        assert loader is not None


# ---------------------------------------------------------------------------
# Helper: collect every CellMapImage from a CellMapDataset's sources
# ---------------------------------------------------------------------------
def _all_images(dataset: CellMapDataset):
    """Yield every CellMapImage in a dataset's input and target sources."""
    from cellmap_data.image import CellMapImage

    for source in list(dataset.input_sources.values()) + list(
        dataset.target_sources.values()
    ):
        if isinstance(source, CellMapImage):
            yield source
        elif isinstance(source, dict):
            for sub in source.values():
                if isinstance(sub, CellMapImage):
                    yield sub


class TestTensorStoreCacheBounding:
    """Tests for the tensorstore_cache_bytes cache-bounding feature."""

    @pytest.fixture
    def dataset(self, tmp_path):
        config = create_test_dataset(
            tmp_path,
            raw_shape=(24, 24, 24),
            num_classes=2,
            raw_scale=(4.0, 4.0, 4.0),
        )
        input_arrays = {"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}}
        target_arrays = {"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}}
        return CellMapDataset(
            raw_path=config["raw_path"],
            target_path=config["gt_path"],
            classes=config["classes"],
            input_arrays=input_arrays,
            target_arrays=target_arrays,
            force_has_data=True,
        )

    # -- parameter stored on loader ------------------------------------------

    def test_cache_bytes_stored_on_loader(self, dataset):
        """tensorstore_cache_bytes is stored as an attribute."""
        loader = CellMapDataLoader(
            dataset, num_workers=0, tensorstore_cache_bytes=100_000_000
        )
        assert loader.tensorstore_cache_bytes == 100_000_000

    def test_default_limit(self, dataset):
        """Without the parameter (or env var), cache bytes is set by default."""
        from cellmap_data.dataloader import _DEFAULT_TENSORSTORE_CACHE_BYTES

        loader = CellMapDataLoader(dataset, num_workers=0)
        assert loader.tensorstore_cache_bytes == _DEFAULT_TENSORSTORE_CACHE_BYTES

    # -- per-worker byte math ------------------------------------------------

    def test_per_worker_division(self, dataset):
        """per_worker = total // num_workers is applied to every CellMapImage."""
        total = 400_000_000  # 400 MB
        num_workers = 3
        CellMapDataLoader(
            dataset, num_workers=num_workers, tensorstore_cache_bytes=total
        )
        # 133_333_333 each, if total = 400_000_000 and num_workers = 3
        expected = total // num_workers
        for img in _all_images(dataset):
            assert isinstance(img.context, ts.Context)
            assert img.context["cache_pool"].to_json() == {
                "total_bytes_limit": expected
            }

    def test_single_process_uses_full_budget(self, dataset):
        """With num_workers=0 the whole budget goes to the single process (÷ 1)."""
        total = 200_000_000
        CellMapDataLoader(dataset, num_workers=0, tensorstore_cache_bytes=total)
        for img in _all_images(dataset):
            assert img.context["cache_pool"].to_json() == {"total_bytes_limit": total}

    def test_context_set_on_target_images(self, dataset):
        """Cache limit is applied to target-source images, not just input-source images."""
        from cellmap_data.image import CellMapImage

        CellMapDataLoader(dataset, num_workers=2, tensorstore_cache_bytes=200_000_000)
        expected = 200_000_000 // 2
        for sources in dataset.target_sources.values():
            if isinstance(sources, dict):
                for src in sources.values():
                    if isinstance(src, CellMapImage):
                        assert src.context["cache_pool"].to_json() == {
                            "total_bytes_limit": expected
                        }

    # -- env var fallback ----------------------------------------------------

    def test_env_var_sets_cache_limit(self, dataset, monkeypatch):
        """CELLMAP_TENSORSTORE_CACHE_BYTES env var is used when parameter is not set."""
        monkeypatch.setenv("CELLMAP_TENSORSTORE_CACHE_BYTES", "300000000")
        loader = CellMapDataLoader(dataset, num_workers=3)
        assert loader.tensorstore_cache_bytes == 300_000_000
        expected = 300_000_000 // 3  # 100 MB per worker
        for img in _all_images(dataset):
            assert img.context["cache_pool"].to_json() == {
                "total_bytes_limit": expected
            }

    def test_param_overrides_env_var(self, dataset, monkeypatch):
        """Explicit parameter takes precedence over the env var."""
        monkeypatch.setenv("CELLMAP_TENSORSTORE_CACHE_BYTES", "999999999")
        CellMapDataLoader(dataset, num_workers=2, tensorstore_cache_bytes=200_000_000)
        expected = 200_000_000 // 2  # 100 MB — param wins
        for img in _all_images(dataset):
            assert img.context["cache_pool"].to_json() == {
                "total_bytes_limit": expected
            }

    # -- validation ----------------------------------------------------------

    def test_negative_cache_bytes_raises_error(self, dataset):
        """Negative tensorstore_cache_bytes values are rejected."""
        with pytest.raises(ValueError, match="must be >= 0"):
            CellMapDataLoader(dataset, num_workers=1, tensorstore_cache_bytes=-100)

    def test_negative_env_var_raises_error(self, dataset, monkeypatch):
        """Negative values in CELLMAP_TENSORSTORE_CACHE_BYTES are rejected."""
        monkeypatch.setenv("CELLMAP_TENSORSTORE_CACHE_BYTES", "-500")
        with pytest.raises(ValueError, match="must be >= 0"):
            CellMapDataLoader(dataset, num_workers=1)

    def test_invalid_env_var_raises_error(self, dataset, monkeypatch):
        """Non-integer values in CELLMAP_TENSORSTORE_CACHE_BYTES are rejected."""
        monkeypatch.setenv("CELLMAP_TENSORSTORE_CACHE_BYTES", "not_a_number")
        with pytest.raises(ValueError, match="Invalid value for environment variable"):
            CellMapDataLoader(dataset, num_workers=1)

    def test_warning_when_cache_less_than_workers(self, dataset, caplog):
        """A warning is logged when tensorstore_cache_bytes < num_workers."""
        import logging

        with caplog.at_level(logging.WARNING, logger="cellmap_data.dataloader"):
            CellMapDataLoader(dataset, num_workers=4, tensorstore_cache_bytes=2)

        # Check that warning was emitted
        assert any(
            "per-worker cache limit of 0 bytes" in r.message for r in caplog.records
        )
        # Each worker gets 1 byte (the minimum)
        for img in _all_images(dataset):
            assert img.context["cache_pool"].to_json() == {"total_bytes_limit": 1}

    # -- CellMapMultiDataset traversal ---------------------------------------

    def test_multidataset_all_images_bounded(self, tmp_path):
        """Recursive traversal reaches images in every sub-dataset."""
        datasets = []
        for i in range(2):
            config = create_test_dataset(
                tmp_path / f"ds{i}",
                raw_shape=(24, 24, 24),
                num_classes=2,
                raw_scale=(4.0, 4.0, 4.0),
            )
            input_arrays = {"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}}
            target_arrays = {"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}}
            datasets.append(
                CellMapDataset(
                    raw_path=config["raw_path"],
                    target_path=config["gt_path"],
                    classes=config["classes"],
                    input_arrays=input_arrays,
                    target_arrays=target_arrays,
                    force_has_data=True,
                )
            )

        multi = CellMapMultiDataset(
            classes=["class_0", "class_1"],
            input_arrays={"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            target_arrays={"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            datasets=datasets,
        )

        CellMapDataLoader(multi, num_workers=2, tensorstore_cache_bytes=200_000_000)
        expected = 200_000_000 // 2

        for ds in datasets:
            for img in _all_images(ds):
                assert img.context["cache_pool"].to_json() == {
                    "total_bytes_limit": expected
                }

    # -- warning when array already open ------------------------------------

    def test_warning_when_array_already_open(self, dataset, caplog):
        """A warning is logged when _array is already cached on an image."""
        import logging

        img = next(iter(dataset.input_sources.values()))
        _ = img.array  # force-open the TensorStore array

        with caplog.at_level(logging.WARNING, logger="cellmap_data.dataloader"):
            CellMapDataLoader(
                dataset, num_workers=1, tensorstore_cache_bytes=100_000_000
            )

        assert any(
            "cache_pool limit will not apply" in r.message for r in caplog.records
        )
        # context is still updated on the image object (even though the open array isn't affected)
        assert img.context["cache_pool"].to_json() == {"total_bytes_limit": 100_000_000}

    # -- functional: data still loads ----------------------------------------

    def test_data_loads_with_bounded_cache(self, dataset):
        """A bounded-cache loader can still produce a valid batch."""
        loader = CellMapDataLoader(
            dataset, batch_size=2, num_workers=0, tensorstore_cache_bytes=50_000_000
        )
        batch = next(iter(loader))
        assert "raw" in batch
        assert isinstance(batch["raw"], torch.Tensor)
        assert batch["raw"].shape[0] == 2
