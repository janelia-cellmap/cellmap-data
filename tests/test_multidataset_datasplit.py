"""
Tests for CellMapMultiDataset and CellMapDataSplit classes.

Tests combining multiple datasets and train/validation splits.
"""

import pytest

from cellmap_data import (
    CellMapDataset,
    CellMapDataSplit,
    CellMapMultiDataset,
)

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
        multi_dataset = CellMapMultiDataset(
            classes=["class_0"],
            input_arrays={"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            target_arrays={"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            datasets=[],
        )

        assert len(multi_dataset.datasets) == 0

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

        multi_dataset = CellMapMultiDataset(
            classes=["class_0", "class_1"],
            input_arrays={"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            target_arrays={"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            datasets=multiple_datasets,
            spatial_transforms=spatial_transforms,
        )

        assert multi_dataset.spatial_transforms is not None


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
