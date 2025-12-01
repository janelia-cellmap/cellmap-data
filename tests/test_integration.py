"""
Integration tests for complete workflows.

Tests end-to-end workflows combining multiple components.
"""

import torch
import torchvision.transforms.v2 as T

from cellmap_data import (
    CellMapDataLoader,
    CellMapDataset,
    CellMapDataSplit,
    CellMapMultiDataset,
)
from cellmap_data.transforms import Binarize, GaussianNoise, Normalize

from .test_helpers import create_test_dataset


class TestTrainingWorkflow:
    """Integration tests for complete training workflows."""

    def test_basic_training_setup(self, tmp_path):
        """Test basic training pipeline setup."""
        # Create dataset
        config = create_test_dataset(
            tmp_path,
            raw_shape=(64, 64, 64),
            num_classes=3,
            raw_scale=(8.0, 8.0, 8.0),
        )

        # Configure arrays
        input_arrays = {"raw": {"shape": (32, 32, 32), "scale": (8.0, 8.0, 8.0)}}
        target_arrays = {"gt": {"shape": (32, 32, 32), "scale": (8.0, 8.0, 8.0)}}

        # Configure transforms
        spatial_transforms = {
            "mirror": {"axes": {"x": 0.5, "y": 0.5}},
            "rotate": {"axes": {"z": [-45, 45]}},
        }

        raw_transforms = T.Compose(
            [
                Normalize(scale=1.0 / 255.0),
                GaussianNoise(std=0.05),
            ]
        )

        target_transforms = T.Compose(
            [
                Binarize(threshold=0.5),
            ]
        )

        # Create dataset
        dataset = CellMapDataset(
            raw_path=config["raw_path"],
            target_path=config["gt_path"],
            classes=config["classes"],
            input_arrays=input_arrays,
            target_arrays=target_arrays,
            spatial_transforms=spatial_transforms,
            raw_value_transforms=raw_transforms,
            target_value_transforms=target_transforms,
            is_train=True,
            force_has_data=True,
        )

        # Create loader
        loader = CellMapDataLoader(
            dataset,
            batch_size=4,
            num_workers=0,
            weighted_sampler=True,
        )

        assert dataset is not None
        assert loader is not None

    def test_train_validation_split_workflow(self, tmp_path):
        """Test complete train/validation split workflow."""
        # Create training and validation datasets
        train_config = create_test_dataset(
            tmp_path / "train",
            raw_shape=(64, 64, 64),
            num_classes=2,
            seed=42,
        )

        val_config = create_test_dataset(
            tmp_path / "val",
            raw_shape=(64, 64, 64),
            num_classes=2,
            seed=100,
        )

        # Configure dataset split
        dataset_dict = {
            "train": [{"raw": train_config["raw_path"], "gt": train_config["gt_path"]}],
            "validate": [{"raw": val_config["raw_path"], "gt": val_config["gt_path"]}],
        }

        input_arrays = {"raw": {"shape": (32, 32, 32), "scale": (8.0, 8.0, 8.0)}}
        target_arrays = {"gt": {"shape": (32, 32, 32), "scale": (8.0, 8.0, 8.0)}}

        # Training transforms
        spatial_transforms = {
            "mirror": {"axes": {"x": 0.5}},
        }

        datasplit = CellMapDataSplit(
            dataset_dict=dataset_dict,
            classes=["class_0", "class_1"],
            input_arrays=input_arrays,
            target_arrays=target_arrays,
            spatial_transforms=spatial_transforms,
            pad=True,
        )

        assert datasplit is not None

    def test_multi_dataset_training(self, tmp_path):
        """Test training with multiple datasets."""
        # Create multiple datasets
        configs = []
        datasets = []

        for i in range(3):
            config = create_test_dataset(
                tmp_path / f"dataset_{i}",
                raw_shape=(48, 48, 48),
                num_classes=2,
                seed=42 + i,
            )
            configs.append(config)

            dataset = CellMapDataset(
                raw_path=config["raw_path"],
                target_path=config["gt_path"],
                classes=config["classes"],
                input_arrays={"raw": {"shape": (16, 16, 16), "scale": (8.0, 8.0, 8.0)}},
                target_arrays={"gt": {"shape": (16, 16, 16), "scale": (8.0, 8.0, 8.0)}},
                is_train=True,
                force_has_data=True,
            )
            datasets.append(dataset)

        # Combine into multi-dataset
        multi_dataset = CellMapMultiDataset(
            classes=["class_0", "class_1"],
            input_arrays={"raw": {"shape": (16, 16, 16), "scale": (8.0, 8.0, 8.0)}},
            target_arrays={"gt": {"shape": (16, 16, 16), "scale": (8.0, 8.0, 8.0)}},
            datasets=datasets,
        )

        # Create loader
        loader = CellMapDataLoader(
            multi_dataset,
            batch_size=4,
            num_workers=0,
            weighted_sampler=True,
        )

        assert len(multi_dataset.datasets) == 3
        assert loader is not None

    def test_multiscale_training_setup(self, tmp_path):
        """Test training with multiscale inputs."""
        config = create_test_dataset(
            tmp_path,
            raw_shape=(64, 64, 64),
            num_classes=2,
        )

        # Multiple scales
        input_arrays = {
            "raw_4nm": {"shape": (32, 32, 32), "scale": (4.0, 4.0, 4.0)},
            "raw_8nm": {"shape": (16, 16, 16), "scale": (8.0, 8.0, 8.0)},
        }

        target_arrays = {"gt": {"shape": (16, 16, 16), "scale": (8.0, 8.0, 8.0)}}

        dataset = CellMapDataset(
            raw_path=config["raw_path"],
            target_path=config["gt_path"],
            classes=config["classes"],
            input_arrays=input_arrays,
            target_arrays=target_arrays,
        )

        loader = CellMapDataLoader(dataset, batch_size=2, num_workers=0)

        assert "raw_4nm" in dataset.input_arrays
        assert "raw_8nm" in dataset.input_arrays
        assert loader is not None


class TestTransformPipeline:
    """Integration tests for transform pipelines."""

    def test_complete_augmentation_pipeline(self, tmp_path):
        """Test complete augmentation pipeline."""
        from cellmap_data.transforms import (
            Binarize,
            GaussianNoise,
            NaNtoNum,
            Normalize,
            RandomContrast,
            RandomGamma,
        )

        config = create_test_dataset(
            tmp_path,
            raw_shape=(48, 48, 48),
            num_classes=2,
        )

        # Complex transform pipeline
        raw_transforms = T.Compose(
            [
                NaNtoNum({"nan": 0.0}),
                Normalize(scale=1.0 / 255.0),
                GaussianNoise(std=0.05),
                RandomContrast(contrast_range=(0.8, 1.2)),
                RandomGamma(gamma_range=(0.8, 1.2)),
            ]
        )

        target_transforms = T.Compose(
            [
                Binarize(threshold=0.5),
                T.ToDtype(torch.float32),
            ]
        )

        # Spatial transforms must come first
        spatial_transforms = {
            "mirror": {"axes": {"x": 0.5, "y": 0.5, "z": 0.2}},
            "rotate": {"axes": {"z": [-180, 180]}},
            "transpose": {"axes": ["x", "y"]},
        }

        dataset = CellMapDataset(
            raw_path=config["raw_path"],
            target_path=config["gt_path"],
            classes=config["classes"],
            input_arrays={"raw": {"shape": (16, 16, 16), "scale": (8.0, 8.0, 8.0)}},
            target_arrays={"gt": {"shape": (16, 16, 16), "scale": (8.0, 8.0, 8.0)}},
            spatial_transforms=spatial_transforms,
            raw_value_transforms=raw_transforms,
            target_value_transforms=target_transforms,
            is_train=True,
            force_has_data=True,
        )

        loader = CellMapDataLoader(dataset, batch_size=2, num_workers=0)

        assert dataset.spatial_transforms is not None
        assert dataset.raw_value_transforms is not None
        assert loader is not None

    def test_per_target_transforms(self, tmp_path):
        """Test different transforms per target array."""
        config = create_test_dataset(
            tmp_path,
            raw_shape=(48, 48, 48),
            num_classes=2,
        )

        # Different transforms for different targets
        target_transforms = {
            "labels": T.Compose([Binarize(threshold=0.5)]),
            "distances": T.Compose([Normalize(scale=1.0 / 100.0)]),
        }

        target_arrays = {
            "labels": {"shape": (16, 16, 16), "scale": (8.0, 8.0, 8.0)},
            "distances": {"shape": (16, 16, 16), "scale": (8.0, 8.0, 8.0)},
        }

        dataset = CellMapDataset(
            raw_path=config["raw_path"],
            target_path=config["gt_path"],
            classes=config["classes"],
            input_arrays={"raw": {"shape": (16, 16, 16), "scale": (8.0, 8.0, 8.0)}},
            target_arrays=target_arrays,
            target_value_transforms=target_transforms,
        )

        assert dataset.target_value_transforms is not None


class TestDataLoaderOptimization:
    """Integration tests for data loader optimizations."""

    def test_memory_optimization_settings(self, tmp_path):
        """Test memory-optimized loader configuration."""
        config = create_test_dataset(
            tmp_path,
            raw_shape=(64, 64, 64),
            num_classes=2,
        )

        dataset = CellMapDataset(
            raw_path=config["raw_path"],
            target_path=config["gt_path"],
            classes=config["classes"],
            input_arrays={"raw": {"shape": (32, 32, 32), "scale": (8.0, 8.0, 8.0)}},
            target_arrays={"gt": {"shape": (32, 32, 32), "scale": (8.0, 8.0, 8.0)}},
        )

        # Optimized loader settings
        loader = CellMapDataLoader(
            dataset,
            batch_size=8,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )

        assert loader is not None

    def test_weighted_sampling_integration(self, tmp_path):
        """Test weighted sampling for class balance."""
        config = create_test_dataset(
            tmp_path,
            raw_shape=(64, 64, 64),
            num_classes=3,
            label_pattern="regions",  # Creates imbalanced classes
        )

        dataset = CellMapDataset(
            raw_path=config["raw_path"],
            target_path=config["gt_path"],
            classes=config["classes"],
            input_arrays={"raw": {"shape": (16, 16, 16), "scale": (8.0, 8.0, 8.0)}},
            target_arrays={"gt": {"shape": (16, 16, 16), "scale": (8.0, 8.0, 8.0)}},
            is_train=True,
            force_has_data=True,
        )

        # Use weighted sampler to balance classes
        loader = CellMapDataLoader(
            dataset,
            batch_size=4,
            num_workers=0,
            weighted_sampler=True,
        )

        assert loader is not None

    def test_iterations_per_epoch_large_dataset(self, tmp_path):
        """Test limited iterations for large datasets."""
        config = create_test_dataset(
            tmp_path,
            raw_shape=(128, 128, 128),  # Larger dataset
            num_classes=2,
        )

        dataset = CellMapDataset(
            raw_path=config["raw_path"],
            target_path=config["gt_path"],
            classes=config["classes"],
            input_arrays={"raw": {"shape": (32, 32, 32), "scale": (8.0, 8.0, 8.0)}},
            target_arrays={"gt": {"shape": (32, 32, 32), "scale": (8.0, 8.0, 8.0)}},
        )

        # Limit iterations per epoch
        loader = CellMapDataLoader(
            dataset,
            batch_size=4,
            num_workers=0,
            iterations_per_epoch=50,  # Only 50 batches per epoch
        )

        assert loader is not None


class TestEdgeCases:
    """Integration tests for edge cases and special scenarios."""

    def test_small_dataset(self, tmp_path):
        """Test with very small dataset."""
        config = create_test_dataset(
            tmp_path,
            raw_shape=(16, 16, 16),  # Small
            num_classes=2,
        )

        dataset = CellMapDataset(
            raw_path=config["raw_path"],
            target_path=config["gt_path"],
            classes=config["classes"],
            input_arrays={"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            target_arrays={"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}},
            pad=True,  # Need padding for small dataset
        )

        loader = CellMapDataLoader(dataset, batch_size=1, num_workers=0)

        assert dataset.pad is True
        assert loader is not None

    def test_single_class(self, tmp_path):
        """Test with single class."""
        config = create_test_dataset(
            tmp_path,
            raw_shape=(32, 32, 32),
            num_classes=1,
        )

        dataset = CellMapDataset(
            raw_path=config["raw_path"],
            target_path=config["gt_path"],
            classes=config["classes"],
            input_arrays={"raw": {"shape": (16, 16, 16), "scale": (8.0, 8.0, 8.0)}},
            target_arrays={"gt": {"shape": (16, 16, 16), "scale": (8.0, 8.0, 8.0)}},
        )

        loader = CellMapDataLoader(dataset, batch_size=2, num_workers=0)

        assert len(dataset.classes) == 1
        assert loader is not None

    def test_anisotropic_data(self, tmp_path):
        """Test with anisotropic voxel sizes."""
        config = create_test_dataset(
            tmp_path,
            raw_shape=(32, 64, 64),
            raw_scale=(16.0, 4.0, 4.0),  # Anisotropic
            num_classes=2,
        )

        dataset = CellMapDataset(
            raw_path=config["raw_path"],
            target_path=config["gt_path"],
            classes=config["classes"],
            input_arrays={"raw": {"shape": (16, 32, 32), "scale": (16.0, 4.0, 4.0)}},
            target_arrays={"gt": {"shape": (16, 32, 32), "scale": (16.0, 4.0, 4.0)}},
        )

        loader = CellMapDataLoader(dataset, batch_size=2, num_workers=0)

        assert dataset.input_arrays["raw"]["scale"] == (16.0, 4.0, 4.0)
        assert loader is not None
