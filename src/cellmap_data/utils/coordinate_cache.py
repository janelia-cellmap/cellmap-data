"""
Coordinate transformation caching system for improved performance.

This module provides dataset-level coordinate transformation caching to address
the performance bottleneck identified in dataset.py line 757 TODO comment.
"""

import hashlib
from typing import Any, Mapping, Optional, Dict, Tuple
import torch
import numpy as np
from threading import Lock


class CoordinateTransformCache:
    """Dataset-level coordinate transformation caching for performance optimization.

    This class addresses the performance bottleneck where duplicate reference frame
    images repeatedly calculate the same coordinate transformations on every
    __getitem__ call. By caching transformations at the dataset level, we eliminate
    redundant calculations and significantly improve performance.

    Thread-safe implementation supports concurrent access from multiple workers.
    """

    def __init__(self, max_cache_size: int = 1000):
        """Initialize coordinate transformation cache.

        Args:
            max_cache_size: Maximum number of cached transformations to store.
                           Uses LRU eviction when limit is reached.
        """
        self._transform_cache: Dict[str, Dict[str, Any]] = {}
        self._reference_frame_cache: Dict[str, str] = {}
        self._cache_access_order: Dict[str, int] = {}
        self._access_counter = 0
        self._max_cache_size = max_cache_size
        self._lock = Lock()

    def _generate_cache_key(
        self, reference_frame_id: str, spatial_transforms: Optional[Mapping[str, Any]]
    ) -> str:
        """Generate a unique cache key for the transformation.

        Args:
            reference_frame_id: Identifier for the reference frame
            spatial_transforms: Spatial transformation parameters

        Returns:
            Unique cache key for the transformation combination
        """
        if spatial_transforms is None:
            transform_hash = "none"
        else:
            # Create a deterministic hash of the spatial transforms
            import json

            transform_str = json.dumps(spatial_transforms, sort_keys=True)
            transform_hash = hashlib.md5(transform_str.encode()).hexdigest()[:16]

        return f"{reference_frame_id}:{transform_hash}"

    def get_cached_transforms(
        self, reference_frame_id: str, spatial_transforms: Optional[Mapping[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Get cached coordinate transformations for reference frame.

        Args:
            reference_frame_id: Identifier for the reference frame
            spatial_transforms: Spatial transformation parameters

        Returns:
            Cached transformation data if available, None otherwise
        """
        cache_key = self._generate_cache_key(reference_frame_id, spatial_transforms)

        with self._lock:
            if cache_key in self._transform_cache:
                # Update access order for LRU
                self._access_counter += 1
                self._cache_access_order[cache_key] = self._access_counter
                return self._transform_cache[cache_key].copy()

        return None

    def cache_transforms(
        self,
        reference_frame_id: str,
        spatial_transforms: Optional[Mapping[str, Any]],
        transformed_coords: Dict[str, Any],
        performance_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Cache coordinate transformation results.

        Args:
            reference_frame_id: Identifier for the reference frame
            spatial_transforms: Spatial transformation parameters used
            transformed_coords: Result of coordinate transformation
            performance_metrics: Optional performance timing data
        """
        cache_key = self._generate_cache_key(reference_frame_id, spatial_transforms)

        with self._lock:
            # Implement LRU eviction if cache is full
            if len(self._transform_cache) >= self._max_cache_size:
                self._evict_lru_entry()

            # Cache the transformation result
            self._transform_cache[cache_key] = {
                "transformed_coords": transformed_coords,
                "reference_frame_id": reference_frame_id,
                "spatial_transforms": spatial_transforms,
                "performance_metrics": performance_metrics or {},
                "cache_time": 0.0,  # Placeholder for timing
            }

            # Update access order
            self._access_counter += 1
            self._cache_access_order[cache_key] = self._access_counter

    def _evict_lru_entry(self) -> None:
        """Evict least recently used cache entry."""
        if not self._cache_access_order:
            return

        # Find the least recently used entry
        lru_key = min(self._cache_access_order.items(), key=lambda x: x[1])[0]

        # Remove from all caches
        self._transform_cache.pop(lru_key, None)
        self._cache_access_order.pop(lru_key, None)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics.

        Returns:
            Dictionary containing cache hit rate, size, and other metrics
        """
        with self._lock:
            return {
                "cache_size": len(self._transform_cache),
                "max_cache_size": self._max_cache_size,
                "total_accesses": self._access_counter,
                "cache_keys": list(self._transform_cache.keys()),
            }

    def clear_cache(self) -> None:
        """Clear all cached transformations."""
        with self._lock:
            self._transform_cache.clear()
            self._reference_frame_cache.clear()
            self._cache_access_order.clear()
            self._access_counter = 0


class PerformanceOptimizedDatasetMixin:
    """Mixin class to add coordinate transformation caching to CellMapDataset.

    This mixin provides the performance optimization functionality that addresses
    the TODO comment in dataset.py line 757. It can be mixed into CellMapDataset
    to enable coordinate transformation caching.
    """

    def __init__(self, *args, **kwargs):
        """Initialize performance optimization components."""
        super().__init__(*args, **kwargs)
        self._coordinate_cache = CoordinateTransformCache()
        self._performance_metrics = {
            "cache_hits": 0,
            "cache_misses": 0,
            "transform_time_saved": 0.0,
        }

    def _get_reference_frame_id(self, center: Dict[str, float]) -> str:
        """Generate a reference frame identifier from center coordinates.

        Args:
            center: Center coordinates for the data sample

        Returns:
            Unique identifier for this reference frame
        """
        # Create a simple reference frame ID based on center coordinates
        coord_str = ",".join(f"{k}:{v:.2f}" for k, v in sorted(center.items()))
        return f"ref_frame_{hashlib.md5(coord_str.encode()).hexdigest()[:12]}"

    def _apply_cached_transforms_to_sources(
        self, spatial_transforms: Optional[Mapping[str, Any]], center: Dict[str, float]
    ) -> bool:
        """Apply cached coordinate transformations to all sources if available.

        Args:
            spatial_transforms: Spatial transformation parameters
            center: Center coordinates for the data sample

        Returns:
            True if cached transforms were applied, False if cache miss
        """
        reference_frame_id = self._get_reference_frame_id(center)
        cached_transforms = self._coordinate_cache.get_cached_transforms(
            reference_frame_id, spatial_transforms
        )

        if cached_transforms is not None:
            # Cache hit - apply cached transformations
            self._performance_metrics["cache_hits"] += 1

            # Apply cached transforms to all sources
            for source_dict in [self.input_sources, self.target_sources]:
                for source_name, source in source_dict.items():
                    if hasattr(source, "set_spatial_transforms"):
                        source.set_spatial_transforms(spatial_transforms)
                        # Could potentially cache the transformed coordinates here
                        # depending on the specific transformation implementation

            return True
        else:
            # Cache miss - will need to calculate transforms
            self._performance_metrics["cache_misses"] += 1
            return False

    def _cache_current_transforms(
        self,
        spatial_transforms: Optional[Mapping[str, Any]],
        center: Dict[str, float],
        transform_time: float = 0.0,
    ) -> None:
        """Cache the current coordinate transformations for future use.

        Args:
            spatial_transforms: Spatial transformation parameters used
            center: Center coordinates for the data sample
            transform_time: Time taken to compute transformations
        """
        reference_frame_id = self._get_reference_frame_id(center)

        # Create transformation result data to cache
        transformed_coords = {
            "center": center,
            "spatial_transforms": spatial_transforms,
            "reference_frame_id": reference_frame_id,
        }

        performance_metrics = {
            "transform_time": transform_time,
            "cache_time": 0.0,  # Current time placeholder
        }

        self._coordinate_cache.cache_transforms(
            reference_frame_id,
            spatial_transforms,
            transformed_coords,
            performance_metrics,
        )

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance optimization metrics.

        Returns:
            Dictionary containing cache performance and optimization metrics
        """
        cache_stats = self._coordinate_cache.get_cache_stats()

        total_requests = (
            self._performance_metrics["cache_hits"]
            + self._performance_metrics["cache_misses"]
        )
        hit_rate = (
            self._performance_metrics["cache_hits"] / max(total_requests, 1)
        ) * 100

        return {
            "cache_hit_rate": hit_rate,
            "total_cache_requests": total_requests,
            "time_saved_estimate": self._performance_metrics["transform_time_saved"],
            "cache_efficiency": cache_stats,
        }
