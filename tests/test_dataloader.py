"""
Tests for CellMapDataLoader class.

Tests data loading, batching, and optimization features using real data.
"""

import pytest
import torch
import numpy as np
from pathlib import Path

from cellmap_data import CellMapDataLoader, CellMapDataset
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
        
        return CellMapDataLoader(dataset, batch_size=2, num_workers=0)
    
    def test_length(self, simple_loader):
        """Test that loader has a length."""
        # Loader may or may not implement __len__
        # depending on configuration
        try:
            length = len(simple_loader)
            assert length >= 0
        except TypeError:
            # Some configurations may not support len
            pass
    
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
        from cellmap_data.transforms import Normalize, Binarize
        import torchvision.transforms.v2 as T
        
        config = create_test_dataset(
            tmp_path,
            raw_shape=(32, 32, 32),
            num_classes=2,
        )
        
        input_arrays = {"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}}
        target_arrays = {"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}}
        
        raw_transforms = T.Compose([Normalize(scale=1.0 / 255.0)])
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
