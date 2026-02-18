"""
Tests for CellMapMultiDataset and CellMapDataSplit classes.

Tests combining multiple datasets and train/validation splits.
"""

import csv
import os

import pytest
import torch
import torchvision.transforms.v2 as T

from cellmap_data import CellMapDataset, CellMapDataSplit, CellMapMultiDataset

from .test_helpers import create_test_dataset


class TestCellMapMultiDataset:
    """Test suite for CellMapMultiDataset class."""

    @pytest.fixture
    def multiple_datasets(self, tmp_path):
        """Create multiple test datasets."""
        datasets = []

        for i in range(3):
            config = create_test_dataset(
                tmp_path / f"dataset_{i}",
                raw_shape=(32, 32, 32),
                num_classes=2,
                raw_scale=(4.0, 4.0, 4.0),
                seed=42 + i,
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
            datasets.append(dataset)

        return datasets

    def test_initialization_basic(self, multiple_datasets):
        """Test basic MultiDataset initialization."""
        multi_dataset = CellMapMultiDataset(
            classes=["class_0", "class_1"],
            input_arrays={"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            target_arrays={"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            datasets=multiple_datasets,
        )

        assert multi_dataset is not None
        assert len(multi_dataset.datasets) == 3

    def test_classes_parameter(self, multiple_datasets):
        """Test classes parameter."""
        classes = ["class_0", "class_1", "class_2"]

        multi_dataset = CellMapMultiDataset(
            classes=classes,
            input_arrays={"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            target_arrays={"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            datasets=multiple_datasets,
        )

        assert multi_dataset.classes == classes

    def test_input_arrays_configuration(self, multiple_datasets):
        """Test input arrays configuration."""
        input_arrays = {
            "raw_4nm": {"shape": (16, 16, 16), "scale": (4.0, 4.0, 4.0)},
            "raw_8nm": {"shape": (8, 8, 8), "scale": (8.0, 8.0, 8.0)},
        }

        multi_dataset = CellMapMultiDataset(
            classes=["class_0", "class_1"],
            input_arrays=input_arrays,
            target_arrays={"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            datasets=multiple_datasets,
        )

        assert "raw_4nm" in multi_dataset.input_arrays
        assert "raw_8nm" in multi_dataset.input_arrays

    def test_target_arrays_configuration(self, multiple_datasets):
        """Test target arrays configuration."""
        target_arrays = {
            "labels": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)},
            "distances": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)},
        }

        multi_dataset = CellMapMultiDataset(
            classes=["class_0", "class_1"],
            input_arrays={"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            target_arrays=target_arrays,
            datasets=multiple_datasets,
        )

        assert "labels" in multi_dataset.target_arrays
        assert "distances" in multi_dataset.target_arrays

    def test_empty_datasets_list(self):
        """Test with empty datasets list."""
        with pytest.raises(ValueError):
            CellMapDataSplit(
                classes=["class_0"],
                input_arrays={"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
                target_arrays={"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
                datasets={"train": []},
            )

    def test_single_dataset(self, multiple_datasets):
        """Test with single dataset."""
        multi_dataset = CellMapMultiDataset(
            classes=["class_0", "class_1"],
            input_arrays={"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            target_arrays={"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            datasets=[multiple_datasets[0]],
        )

        assert len(multi_dataset.datasets) == 1

    def test_spatial_transforms(self, multiple_datasets):
        """Test spatial transforms configuration."""
        spatial_transforms = {
            "mirror": {"axes": {"x": 0.5, "y": 0.5}},
            "rotate": {"axes": {"z": [-45, 45]}},
        }

        datasplit = CellMapDataSplit(
            classes=["class_0", "class_1"],
            input_arrays={"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            target_arrays={"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            datasets={"train": multiple_datasets},
            spatial_transforms=spatial_transforms,
            force_has_data=True,
        )

        assert datasplit.spatial_transforms is not None


class TestCellMapDataSplit:
    """Test suite for CellMapDataSplit class."""

    @pytest.fixture
    def datasplit_paths(self, tmp_path):
        """Create paths for train and validation datasets."""
        # Create training datasets
        train_configs = []
        for i in range(2):
            config = create_test_dataset(
                tmp_path / f"train_{i}",
                raw_shape=(32, 32, 32),
                num_classes=2,
                seed=42 + i,
            )
            train_configs.append(config)

        # Create validation datasets
        val_configs = []
        for i in range(1):
            config = create_test_dataset(
                tmp_path / f"val_{i}",
                raw_shape=(32, 32, 32),
                num_classes=2,
                seed=100 + i,
            )
            val_configs.append(config)

        return train_configs, val_configs

    def test_initialization_with_dict(self, datasplit_paths):
        """Test DataSplit initialization with dictionary."""
        train_configs, val_configs = datasplit_paths

        dataset_dict = {
            "train": [
                {"raw": tc["raw_path"], "gt": tc["gt_path"]} for tc in train_configs
            ],
            "validate": [
                {"raw": vc["raw_path"], "gt": vc["gt_path"]} for vc in val_configs
            ],
        }

        datasplit = CellMapDataSplit(
            dataset_dict=dataset_dict,
            classes=["class_0", "class_1"],
            input_arrays={"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            target_arrays={"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
        )

        assert datasplit is not None

    def test_train_validation_split(self, datasplit_paths):
        """Test accessing train and validation datasets."""
        train_configs, val_configs = datasplit_paths

        dataset_dict = {
            "train": [
                {"raw": tc["raw_path"], "gt": tc["gt_path"]} for tc in train_configs
            ],
            "validate": [
                {"raw": vc["raw_path"], "gt": vc["gt_path"]} for vc in val_configs
            ],
        }

        datasplit = CellMapDataSplit(
            dataset_dict=dataset_dict,
            classes=["class_0", "class_1"],
            input_arrays={"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            target_arrays={"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            force_has_data=True,
        )

        # Should have train and validation datasets
        assert hasattr(datasplit, "train_datasets") or hasattr(
            datasplit, "train_datasets_combined"
        )
        assert hasattr(datasplit, "validation_datasets") or hasattr(
            datasplit, "validation_datasets_combined"
        )

    def test_classes_parameter(self, datasplit_paths):
        """Test classes parameter."""
        train_configs, val_configs = datasplit_paths

        dataset_dict = {
            "train": [
                {"raw": tc["raw_path"], "gt": tc["gt_path"]} for tc in train_configs
            ],
            "validate": [
                {"raw": vc["raw_path"], "gt": vc["gt_path"]} for vc in val_configs
            ],
        }

        classes = ["class_0", "class_1", "class_2"]

        datasplit = CellMapDataSplit(
            dataset_dict=dataset_dict,
            classes=classes,
            input_arrays={"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            target_arrays={"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            force_has_data=True,
        )

        assert datasplit.classes == classes

    def test_input_arrays_configuration(self, datasplit_paths):
        """Test input arrays configuration."""
        train_configs, val_configs = datasplit_paths

        dataset_dict = {
            "train": [
                {"raw": tc["raw_path"], "gt": tc["gt_path"]} for tc in train_configs
            ],
            "validate": [
                {"raw": vc["raw_path"], "gt": vc["gt_path"]} for vc in val_configs
            ],
        }

        input_arrays = {
            "raw_4nm": {"shape": (16, 16, 16), "scale": (4.0, 4.0, 4.0)},
            "raw_8nm": {"shape": (8, 8, 8), "scale": (8.0, 8.0, 8.0)},
        }

        datasplit = CellMapDataSplit(
            dataset_dict=dataset_dict,
            classes=["class_0", "class_1"],
            input_arrays=input_arrays,
            target_arrays={"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            force_has_data=True,
        )

        assert datasplit.input_arrays is not None

    def test_spatial_transforms_configuration(self, datasplit_paths):
        """Test spatial transforms configuration."""
        train_configs, val_configs = datasplit_paths

        dataset_dict = {
            "train": [
                {"raw": tc["raw_path"], "gt": tc["gt_path"]} for tc in train_configs
            ],
            "validate": [
                {"raw": vc["raw_path"], "gt": vc["gt_path"]} for vc in val_configs
            ],
        }

        spatial_transforms = {
            "mirror": {"axes": {"x": 0.5}},
            "rotate": {"axes": {"z": [-30, 30]}},
        }

        datasplit = CellMapDataSplit(
            dataset_dict=dataset_dict,
            classes=["class_0", "class_1"],
            input_arrays={"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            target_arrays={"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            spatial_transforms=spatial_transforms,
            force_has_data=True,
        )

        assert datasplit is not None

    def test_only_train_split(self, datasplit_paths):
        """Test with only training data."""
        train_configs, _ = datasplit_paths

        dataset_dict = {
            "train": [
                {"raw": tc["raw_path"], "gt": tc["gt_path"]} for tc in train_configs
            ],
        }

        datasplit = CellMapDataSplit(
            dataset_dict=dataset_dict,
            classes=["class_0", "class_1"],
            input_arrays={"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            target_arrays={"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            force_has_data=True,
        )

        assert datasplit is not None

    def test_only_validation_split(self, datasplit_paths):
        """Test with only validation data."""
        _, val_configs = datasplit_paths

        dataset_dict = {
            "validate": [
                {"raw": vc["raw_path"], "gt": vc["gt_path"]} for vc in val_configs
            ],
        }

        datasplit = CellMapDataSplit(
            dataset_dict=dataset_dict,
            classes=["class_0", "class_1"],
            input_arrays={"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            target_arrays={"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            force_has_data=True,
        )

        assert datasplit is not None


class TestMultiDatasetIntegration:
    """Integration tests for multi-dataset scenarios."""

    def test_multi_dataset_with_loader(self, tmp_path):
        """Test MultiDataset with DataLoader."""
        from cellmap_data import CellMapDataLoader

        # Create multiple datasets
        datasets = []
        for i in range(2):
            config = create_test_dataset(
                tmp_path / f"dataset_{i}",
                raw_shape=(24, 24, 24),
                num_classes=2,
                seed=42 + i,
            )

            dataset = CellMapDataset(
                raw_path=config["raw_path"],
                target_path=config["gt_path"],
                classes=config["classes"],
                input_arrays={"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
                target_arrays={"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            )
            datasets.append(dataset)

        # Create MultiDataset
        multi_dataset = CellMapMultiDataset(
            classes=["class_0", "class_1"],
            input_arrays={"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            target_arrays={"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            datasets=datasets,
        )

        # Create loader
        loader = CellMapDataLoader(multi_dataset, batch_size=2, num_workers=0)

        assert loader is not None

    def test_datasplit_with_loaders(self, tmp_path):
        """Test DataSplit with separate train/val loaders."""

        # Create datasets
        train_config = create_test_dataset(
            tmp_path / "train",
            raw_shape=(24, 24, 24),
            num_classes=2,
        )
        val_config = create_test_dataset(
            tmp_path / "val",
            raw_shape=(24, 24, 24),
            num_classes=2,
        )

        dataset_dict = {
            "train": [{"raw": train_config["raw_path"], "gt": train_config["gt_path"]}],
            "validate": [{"raw": val_config["raw_path"], "gt": val_config["gt_path"]}],
        }

        datasplit = CellMapDataSplit(
            dataset_dict=dataset_dict,
            classes=["class_0", "class_1"],
            input_arrays={"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            target_arrays={"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
        )

        # DataSplit should be created successfully
        assert datasplit is not None

    def test_different_resolution_datasets(self, tmp_path):
        """Test combining datasets with different resolutions."""
        # Create datasets with different scales
        config1 = create_test_dataset(
            tmp_path / "dataset_4nm",
            raw_shape=(32, 32, 32),
            raw_scale=(4.0, 4.0, 4.0),
            num_classes=2,
        )

        config2 = create_test_dataset(
            tmp_path / "dataset_8nm",
            raw_shape=(32, 32, 32),
            raw_scale=(8.0, 8.0, 8.0),
            num_classes=2,
        )

        datasets = []
        for config in [config1, config2]:
            dataset = CellMapDataset(
                raw_path=config["raw_path"],
                target_path=config["gt_path"],
                classes=config["classes"],
                input_arrays={"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
                target_arrays={"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            )
            datasets.append(dataset)

        # Create MultiDataset
        multi_dataset = CellMapMultiDataset(
            classes=["class_0", "class_1"],
            input_arrays={"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            target_arrays={"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            datasets=datasets,
        )

        assert len(multi_dataset.datasets) == 2


class TestCellMapMultiDatasetProperties:
    """Tests for CellMapMultiDataset properties and methods not yet covered."""

    @pytest.fixture
    def multi_dataset(self, tmp_path):
        """Build a CellMapMultiDataset from two real datasets."""
        datasets = []
        for i in range(2):
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

        return CellMapMultiDataset(
            classes=["class_0", "class_1"],
            input_arrays={"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            target_arrays={"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            datasets=datasets,
        )

    def test_has_data_true(self, multi_dataset):
        assert multi_dataset.has_data is True

    def test_class_counts_structure(self, multi_dataset):
        counts = multi_dataset.class_counts
        assert "totals" in counts
        assert "class_0" in counts["totals"]
        assert "class_1" in counts["totals"]

    def test_class_weights_keys(self, multi_dataset):
        weights = multi_dataset.class_weights
        assert "class_0" in weights
        assert "class_1" in weights
        for w in weights.values():
            assert w >= 0

    def test_dataset_weights_keys(self, multi_dataset):
        dw = multi_dataset.dataset_weights
        # Should have one entry per dataset
        assert len(dw) == len(multi_dataset.datasets)
        for w in dw.values():
            assert w >= 0

    def test_sample_weights_length(self, multi_dataset):
        sw = multi_dataset.sample_weights
        assert len(sw) == len(multi_dataset)

    def test_validation_indices_nonempty(self, multi_dataset):
        indices = multi_dataset.validation_indices
        assert isinstance(indices, list)
        assert len(indices) > 0
        assert all(0 <= i < len(multi_dataset) for i in indices)

    def test_verify_true(self, multi_dataset):
        assert multi_dataset.verify() is True

    def test_get_weighted_sampler(self, multi_dataset):
        sampler = multi_dataset.get_weighted_sampler(batch_size=4)
        assert sampler is not None

    def test_get_random_subset_indices(self, multi_dataset):
        indices = multi_dataset.get_random_subset_indices(4, weighted=False)
        assert len(indices) == 4

    def test_get_random_subset_indices_weighted(self, multi_dataset):
        indices = multi_dataset.get_random_subset_indices(4, weighted=True)
        assert len(indices) == 4

    def test_get_subset_random_sampler(self, multi_dataset):
        sampler = multi_dataset.get_subset_random_sampler(4)
        assert sampler is not None

    def test_get_indices(self, multi_dataset):
        indices = multi_dataset.get_indices({"x": 8, "y": 8, "z": 8})
        assert isinstance(indices, list)
        assert len(indices) > 0

    def test_set_raw_value_transforms(self, multi_dataset):
        new_transforms = T.Compose([T.ToDtype(torch.float, scale=True)])
        multi_dataset.set_raw_value_transforms(new_transforms)

    def test_set_target_value_transforms(self, multi_dataset):
        new_transforms = T.Compose([T.ToDtype(torch.float)])
        multi_dataset.set_target_value_transforms(new_transforms)

    def test_set_spatial_transforms(self, multi_dataset):
        transforms = {"mirror": {"axes": {"x": 0.5}}}
        multi_dataset.set_spatial_transforms(transforms)

    def test_repr(self, multi_dataset):
        r = repr(multi_dataset)
        assert "CellMapMultiDataset" in r

    def test_empty_class_method(self):
        empty = CellMapMultiDataset.empty()
        assert empty is not None
        assert empty.has_data is False
        assert empty.classes == []
        assert empty.validation_indices == []

    def test_verify_empty_returns_false(self):
        empty = CellMapMultiDataset.empty()
        assert empty.verify() is False

    def test_no_classes_dataset_weights(self, tmp_path):
        """Dataset weights with no classes should give equal weights."""
        config = create_test_dataset(tmp_path / "ds", raw_shape=(32, 32, 32))
        ds = CellMapDataset(
            raw_path=config["raw_path"],
            target_path=config["gt_path"],
            classes=config["classes"],
            input_arrays={"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            target_arrays={"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
        )
        multi = CellMapMultiDataset(
            classes=[],
            input_arrays={"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            target_arrays={},
            datasets=[ds],
        )
        dw = multi.dataset_weights
        assert list(dw.values())[0] == 1.0


class TestCellMapDataSplitExtended:
    """Extended tests for CellMapDataSplit."""

    @pytest.fixture
    def train_val_configs(self, tmp_path):
        train = []
        for i in range(2):
            train.append(
                create_test_dataset(
                    tmp_path / f"train_{i}",
                    raw_shape=(32, 32, 32),
                    num_classes=2,
                    seed=i,
                )
            )
        val = [
            create_test_dataset(
                tmp_path / "val_0",
                raw_shape=(32, 32, 32),
                num_classes=2,
                seed=99,
            )
        ]
        return train, val

    @pytest.fixture
    def datasplit(self, train_val_configs):
        train, val = train_val_configs
        dataset_dict = {
            "train": [{"raw": c["raw_path"], "gt": c["gt_path"]} for c in train],
            "validate": [{"raw": c["raw_path"], "gt": c["gt_path"]} for c in val],
        }
        return CellMapDataSplit(
            dataset_dict=dataset_dict,
            classes=["class_0", "class_1"],
            input_arrays={"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            target_arrays={"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            force_has_data=True,
        )

    def test_from_csv(self, tmp_path, train_val_configs):
        """Test CellMapDataSplit.from_csv loads the dataset_dict correctly."""
        train, val = train_val_configs

        csv_path = str(tmp_path / "splits.csv")
        rows = []
        for c in train:
            raw_dir, raw_file = os.path.split(c["raw_path"])
            gt_dir, gt_file = os.path.split(c["gt_path"])
            rows.append(["train", raw_dir, raw_file, gt_dir, gt_file])
        for c in val:
            raw_dir, raw_file = os.path.split(c["raw_path"])
            gt_dir, gt_file = os.path.split(c["gt_path"])
            rows.append(["validate", raw_dir, raw_file, gt_dir, gt_file])

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        # Use from_csv via the constructor
        split = CellMapDataSplit(
            csv_path=csv_path,
            classes=["class_0", "class_1"],
            input_arrays={"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            target_arrays={"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            force_has_data=True,
        )
        assert len(split.train_datasets) == 2
        assert len(split.validation_datasets) == 1

    def test_from_csv_no_gt(self, tmp_path, train_val_configs):
        """Test CSV rows without gt columns."""
        train, _ = train_val_configs

        csv_path = str(tmp_path / "splits_no_gt.csv")
        rows = []
        for c in train:
            raw_dir, raw_file = os.path.split(c["raw_path"])
            rows.append(["train", raw_dir, raw_file])

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        # Direct call to from_csv
        split = CellMapDataSplit(
            classes=["class_0", "class_1"],
            input_arrays={"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            target_arrays={"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            force_has_data=True,
            datasets={"train": []},
        )
        # Read CSV manually using the method
        result = split.from_csv(csv_path)
        assert "train" in result
        assert len(result["train"]) == 2
        for entry in result["train"]:
            assert entry["gt"] == ""

    def test_train_datasets_combined_property(self, datasplit):
        combined = datasplit.train_datasets_combined
        assert combined is not None
        assert len(combined) > 0

    def test_validation_datasets_combined_property(self, datasplit):
        combined = datasplit.validation_datasets_combined
        assert combined is not None

    def test_class_counts_property(self, datasplit):
        counts = datasplit.class_counts
        assert "train" in counts
        assert "validate" in counts

    def test_repr(self, datasplit):
        r = repr(datasplit)
        assert "CellMapDataSplit" in r

    def test_no_source_raises(self):
        """Providing no data source should raise ValueError."""
        with pytest.raises(ValueError, match="One of"):
            CellMapDataSplit(
                classes=["class_0"],
                input_arrays={"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            )

    def test_set_raw_value_transforms(self, datasplit):
        new_transform = T.Compose([T.ToDtype(torch.float, scale=True)])
        datasplit.set_raw_value_transforms(
            train_transforms=new_transform, val_transforms=new_transform
        )

    def test_set_target_value_transforms(self, datasplit):
        new_transform = T.Compose([T.ToDtype(torch.float)])
        datasplit.set_target_value_transforms(new_transform)

    def test_set_spatial_transforms(self, datasplit):
        train_transforms = {"mirror": {"axes": {"x": 0.5}}}
        datasplit.set_spatial_transforms(train_transforms=train_transforms)

    def test_set_raw_value_transforms_after_combined(self, datasplit):
        """Test set_raw_value_transforms after train_datasets_combined is cached."""
        _ = datasplit.train_datasets_combined
        new_transform = T.Compose([T.ToDtype(torch.float, scale=True)])
        datasplit.set_raw_value_transforms(train_transforms=new_transform)

    def test_set_target_value_transforms_after_combined(self, datasplit):
        """Test set_target_value_transforms after combined datasets are cached."""
        _ = datasplit.train_datasets_combined
        _ = datasplit.validation_datasets_combined
        new_transform = T.Compose([T.ToDtype(torch.float)])
        datasplit.set_target_value_transforms(new_transform)

    def test_set_spatial_transforms_after_combined(self, datasplit):
        """Test set_spatial_transforms after train_datasets_combined is cached."""
        _ = datasplit.train_datasets_combined
        _ = datasplit.validation_datasets_combined
        transforms = {"mirror": {"axes": {"x": 0.5}}}
        datasplit.set_spatial_transforms(
            train_transforms=transforms, val_transforms=transforms
        )

    def test_to_device(self, datasplit):
        datasplit.to("cpu")
        assert datasplit.device == "cpu"

    def test_to_device_after_combined(self, datasplit):
        _ = datasplit.train_datasets_combined
        _ = datasplit.validation_datasets_combined
        datasplit.to("cpu")

    def test_pad_string_train(self, train_val_configs):
        train, val = train_val_configs
        dataset_dict = {
            "train": [{"raw": c["raw_path"], "gt": c["gt_path"]} for c in train],
            "validate": [{"raw": c["raw_path"], "gt": c["gt_path"]} for c in val],
        }
        split = CellMapDataSplit(
            dataset_dict=dataset_dict,
            classes=["class_0", "class_1"],
            input_arrays={"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            target_arrays={"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            pad="train",
            force_has_data=True,
        )
        assert split.pad_training is True
        assert split.pad_validation is False

    def test_pad_string_validate(self, train_val_configs):
        train, val = train_val_configs
        dataset_dict = {
            "train": [{"raw": c["raw_path"], "gt": c["gt_path"]} for c in train],
            "validate": [{"raw": c["raw_path"], "gt": c["gt_path"]} for c in val],
        }
        split = CellMapDataSplit(
            dataset_dict=dataset_dict,
            classes=["class_0", "class_1"],
            input_arrays={"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            target_arrays={"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            pad="validate",
            force_has_data=True,
        )
        assert split.pad_training is False
        assert split.pad_validation is True

    def test_initialization_with_datasets_no_validate(self, tmp_path):
        """Test providing datasets dict without validate key."""
        config = create_test_dataset(tmp_path / "ds", raw_shape=(32, 32, 32))
        ds = CellMapDataset(
            raw_path=config["raw_path"],
            target_path=config["gt_path"],
            classes=config["classes"],
            input_arrays={"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            target_arrays={"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
        )
        split = CellMapDataSplit(
            classes=config["classes"],
            input_arrays={"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            target_arrays={"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            datasets={"train": [ds]},
            force_has_data=True,
        )
        assert split.validation_datasets == []

    def test_validation_blocks_property(self, datasplit):
        blocks = datasplit.validation_blocks
        assert blocks is not None
