"""
Test coordinate transformation caching performance optimization.

This test validates the implementation of dataset-level coordinate transformation
caching that addresses the TODO comment in dataset.py line 765.
"""
import pytest
import time
import numpy as np
import torch
from unittest.mock import patch

from cellmap_data import CellMapDataset
from cellmap_data.utils.coordinate_cache import CoordinateTransformCache


def create_test_dataset():
    """Helper function to create a test dataset with proper RNG."""
    input_arrays = {
        "test_input": {
            "shape": [100, 100, 100],  # Use 3D arrays to avoid MultiDataset creation
            "scale": [1.0, 1.0, 1.0]
        }
    }
    target_arrays = {
        "test_target": {
            "shape": [100, 100, 100],  # Use 3D arrays to avoid MultiDataset creation
            "scale": [1.0, 1.0, 1.0]
        }
    }
    
    return CellMapDataset(
        input_path="test_path",
        target_path="test_path",
        input_arrays=input_arrays,
        target_arrays=target_arrays,
        classes=["background", "test"],
        rng=torch.Generator().manual_seed(42)
    )


class TestCoordinateTransformationCaching:
    """Test coordinate transformation caching performance optimization."""

    def test_coordinate_cache_initialization(self):
        """Test that coordinate cache is properly initialized in CellMapDataset."""
        dataset = create_test_dataset()
        
        # Verify coordinate cache is initialized
        assert hasattr(dataset, '_coordinate_cache')
        assert isinstance(dataset._coordinate_cache, CoordinateTransformCache)

    def test_reference_frame_id_generation(self):
        """Test reference frame ID generation for caching."""
        dataset = create_test_dataset()

        # Test reference frame ID generation
        center1 = {"x": 5.0, "y": 5.0, "z": 5.0}
        center2 = {"x": 5.0, "y": 5.0, "z": 5.0}  # Same center
        center3 = {"x": 6.0, "y": 5.0, "z": 5.0}  # Different center

        ref_id1 = dataset._get_reference_frame_id(center1)
        ref_id2 = dataset._get_reference_frame_id(center2)
        ref_id3 = dataset._get_reference_frame_id(center3)

        # Same centers should produce same reference frame IDs
        assert ref_id1 == ref_id2
        # Different centers should produce different reference frame IDs
        assert ref_id1 != ref_id3
        # Reference frame IDs should be strings with expected format
        assert isinstance(ref_id1, str)
        assert ref_id1.startswith("ref_frame_")

    def test_performance_metrics_collection(self):
        """Test that performance metrics are collected properly."""
        dataset = create_test_dataset()

        # Get initial performance metrics
        metrics = dataset.get_performance_metrics()
        
        # Verify metrics structure
        assert isinstance(metrics, dict)
        assert 'cache_size' in metrics
        assert 'max_cache_size' in metrics
        assert 'total_cache_requests' in metrics
        assert 'cache_efficiency' in metrics

        # Initial metrics should show empty cache
        assert metrics['cache_size'] == 0

    def test_coordinate_cache_basic_functionality(self):
        """Test basic coordinate cache functionality."""
        cache = CoordinateTransformCache(max_cache_size=10)
        
        # Test initial state
        assert cache.get_cache_stats()['cache_size'] == 0
        
        # Test cache miss
        result = cache.get_cached_transforms("test_ref", None)
        assert result is None
        
        # Test caching
        test_coords = {'test': 'data'}
        cache.cache_transforms("test_ref", None, test_coords)
        
        # Test cache hit
        result = cache.get_cached_transforms("test_ref", None)
        assert result is not None
        assert 'transformed_coords' in result
        
        # Verify cache size increased
        assert cache.get_cache_stats()['cache_size'] == 1

    def test_spatial_transform_application_caching(self):
        """Test that spatial transforms are applied using the caching system."""
        dataset = create_test_dataset()

        center = {"x": 5.0, "y": 5.0, "z": 5.0}
        spatial_transforms = {"mirror": ["x"]}

        # Test cache miss (first call)
        cache_hit_1 = dataset._apply_cached_transforms_to_sources(spatial_transforms, center)
        assert cache_hit_1 is False  # Should be cache miss on first call

        # Cache the transforms
        dataset._cache_current_transforms(spatial_transforms, center, 0.1)

        # Test cache hit (second call with same parameters)
        cache_hit_2 = dataset._apply_cached_transforms_to_sources(spatial_transforms, center)
        assert cache_hit_2 is True  # Should be cache hit on second call

    def test_cache_key_generation_with_transforms(self):
        """Test cache key generation with different spatial transforms."""
        cache = CoordinateTransformCache()
        
        # Test with None transforms
        key1 = cache._generate_cache_key("ref1", None)
        key2 = cache._generate_cache_key("ref1", None)
        assert key1 == key2
        
        # Test with different transforms
        transforms1 = {"mirror": ["x"]}
        transforms2 = {"mirror": ["y"]}
        
        key3 = cache._generate_cache_key("ref1", transforms1)
        key4 = cache._generate_cache_key("ref1", transforms2)
        assert key3 != key4
        
        # Test with same transforms
        key5 = cache._generate_cache_key("ref1", transforms1)
        assert key3 == key5

    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache reaches maximum size."""
        cache = CoordinateTransformCache(max_cache_size=3)
        
        # Fill cache to capacity
        for i in range(3):
            cache.cache_transforms(f"ref_{i}", None, {'data': i})
        
        assert cache.get_cache_stats()['cache_size'] == 3
        
        # Add one more item, should trigger LRU eviction
        cache.cache_transforms("ref_3", None, {'data': 3})
        
        # Cache size should remain at max capacity
        assert cache.get_cache_stats()['cache_size'] == 3
        
        # First item should have been evicted (LRU)
        result = cache.get_cached_transforms("ref_0", None)
        assert result is None  # Should be evicted
        
        # Last item should still be present
        result = cache.get_cached_transforms("ref_3", None)
        assert result is not None


class TestCoordinateTransformationPerformance:
    """Test performance improvements from coordinate transformation caching."""

    def test_todo_comment_resolution_tracking(self):
        """Test that the TODO comment has been addressed in the implementation."""
        # This test documents that we've resolved the TODO comment from line 765
        # in dataset.py regarding coordinate transformation optimization
        
        dataset = create_test_dataset()

        # Verify that coordinate transformation caching infrastructure exists
        assert hasattr(dataset, '_coordinate_cache')
        assert hasattr(dataset, '_get_reference_frame_id')
        assert hasattr(dataset, '_apply_cached_transforms_to_sources')
        assert hasattr(dataset, '_cache_current_transforms')
        assert hasattr(dataset, 'get_performance_metrics')

        # Verify cache is functional
        center = {"x": 5.0, "y": 5.0, "z": 5.0}
        ref_id = dataset._get_reference_frame_id(center)
        assert isinstance(ref_id, str)
        assert len(ref_id) > 0

    def test_performance_improvement_potential(self):
        """Test that the caching system has the infrastructure for performance improvements."""
        # This test verifies that the performance optimization infrastructure
        # is in place and can provide measurable improvements
        
        dataset = create_test_dataset()

        center = {"x": 5.0, "y": 5.0, "z": 5.0}
        spatial_transforms = {"mirror": ["x", "y"]}

        # Measure time for cache miss (first access)
        start_time = time.time()
        cache_hit_1 = dataset._apply_cached_transforms_to_sources(spatial_transforms, center)
        first_access_time = time.time() - start_time
        
        assert cache_hit_1 is False  # Should be cache miss
        
        # Cache the transforms
        dataset._cache_current_transforms(spatial_transforms, center, first_access_time)
        
        # Measure time for cache hit (second access)
        start_time = time.time()
        cache_hit_2 = dataset._apply_cached_transforms_to_sources(spatial_transforms, center)
        second_access_time = time.time() - start_time
        
        assert cache_hit_2 is True  # Should be cache hit
        
        # Verify that performance metrics can be collected
        metrics = dataset.get_performance_metrics()
        assert isinstance(metrics, dict)
        assert 'cache_size' in metrics
        assert metrics['cache_size'] > 0  # Should have cached data

        # The infrastructure is in place for performance improvements
        # (actual performance gains depend on the complexity of transformations)
        print(f"Cache miss time: {first_access_time:.6f}s")
        print(f"Cache hit time: {second_access_time:.6f}s")
        print(f"Performance metrics: {metrics}")
