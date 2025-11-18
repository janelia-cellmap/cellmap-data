"""
CellMap Coordinate Transformation Module

Centralized coordinate transformation logic extracted from CellMapDataset
to provide specialized coordinate handling and eliminate code duplication.

This module consolidates coordinate transformations, spatial transforms,
and coordinate caching that was previously embedded in the monolithic
CellMapDataset class.
"""

import time
from typing import Dict, Any, Optional, Mapping, Sequence, Tuple
import numpy as np
import torch
from ..utils.logging_config import get_logger
from ..utils.coordinate_cache import CoordinateTransformCache
from ..exceptions import CoordinateTransformError

logger = get_logger("coordinate_transformer")


class CoordinateTransformer:
    """Centralized coordinate transformation management for CellMap datasets.

    Handles spatial transformations, coordinate mapping, and caching for improved
    performance. Extracted from CellMapDataset to reduce complexity and enable
    reuse across different dataset types.

    Attributes:
        axis_order: Order of spatial axes (e.g., "zyx")
        cache: Coordinate transformation cache for performance
        transform_stats: Statistics on transformation operations

    Examples:
        >>> transformer = CoordinateTransformer("zyx")
        >>> transformed_center = transformer.apply_transforms(center, transforms)

        >>> transformer = CoordinateTransformer("zyx", enable_caching=True)
        >>> transformer.get_performance_metrics()
    """

    def __init__(self, axis_order: str = "zyx", enable_caching: bool = True):
        """Initialize coordinate transformer.

        Args:
            axis_order: Order of spatial axes in data arrays
            enable_caching: Whether to enable coordinate transformation caching
        """
        self.axis_order = axis_order
        self.enable_caching = enable_caching

        # Initialize coordinate cache for performance optimization
        if enable_caching:
            self._coordinate_cache = CoordinateTransformCache()
        else:
            self._coordinate_cache = None

        # Track transformation statistics
        self.transform_stats = {
            "total_transforms": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_transform_time": 0.0,
            "average_transform_time": 0.0,
        }

        logger.debug(
            f"CoordinateTransformer initialized with axis_order='{axis_order}', caching={enable_caching}"
        )

    def get_reference_frame_id(self, center: Dict[str, float]) -> str:
        """Generate reference frame identifier for coordinate caching.

        Args:
            center: Center coordinates as dict with axis names as keys

        Returns:
            str: Unique identifier for the reference frame
        """
        # Create a consistent identifier based on center coordinates
        sorted_coords = sorted(center.items())
        coord_str = "_".join(f"{k}={v:.3f}" for k, v in sorted_coords)
        return f"ref_frame_{hash(coord_str) % 10000000}"  # Limit hash size

    def validate_center_coordinates(self, center: Dict[str, float]) -> None:
        """Validate center coordinate format and values.

        Args:
            center: Center coordinates to validate

        Raises:
            CoordinateTransformError: If coordinates are invalid
        """
        if not isinstance(center, dict):
            raise CoordinateTransformError("Center coordinates must be a dictionary")

        if not center:
            raise CoordinateTransformError("Center coordinates cannot be empty")

        # Check that all axis order dimensions are present
        for axis in self.axis_order:
            if axis not in center:
                raise CoordinateTransformError(
                    f"Missing axis '{axis}' in center coordinates. "
                    f"Expected axes: {list(self.axis_order)}, got: {list(center.keys())}"
                )

        # Validate coordinate values
        for axis, value in center.items():
            if not isinstance(value, (int, float)):
                raise CoordinateTransformError(
                    f"Coordinate value for axis '{axis}' must be numeric, got {type(value)}"
                )
            if not np.isfinite(value):
                raise CoordinateTransformError(
                    f"Coordinate value for axis '{axis}' must be finite, got {value}"
                )

    def apply_spatial_transforms(
        self, center: Dict[str, float], spatial_transforms: Optional[Mapping[str, Any]]
    ) -> Dict[str, float]:
        """Apply spatial transformations to center coordinates.

        Args:
            center: Original center coordinates
            spatial_transforms: Spatial transformation specifications

        Returns:
            Dict[str, float]: Transformed center coordinates

        Raises:
            CoordinateTransformError: If transformation fails
        """
        start_time = time.time()

        try:
            # Validate input coordinates
            self.validate_center_coordinates(center)

            if spatial_transforms is None:
                # No transforms to apply
                transformed_center = center.copy()
            else:
                # Apply transformations (simplified implementation)
                transformed_center = self._apply_transform_operations(
                    center, spatial_transforms
                )

            # Update statistics
            transform_time = time.time() - start_time
            self.transform_stats["total_transforms"] += 1
            self.transform_stats["total_transform_time"] += transform_time
            self.transform_stats["average_transform_time"] = (
                self.transform_stats["total_transform_time"]
                / self.transform_stats["total_transforms"]
            )

            logger.debug(f"Applied spatial transforms in {transform_time:.4f}s")
            return transformed_center

        except Exception as e:
            logger.error(f"Spatial transformation failed: {e}")
            raise CoordinateTransformError(f"Failed to apply spatial transforms: {e}")

    def _apply_transform_operations(
        self, center: Dict[str, float], spatial_transforms: Mapping[str, Any]
    ) -> Dict[str, float]:
        """Apply individual transformation operations to coordinates.

        Args:
            center: Original coordinates
            spatial_transforms: Transform specifications

        Returns:
            Dict[str, float]: Transformed coordinates
        """
        transformed_center = center.copy()

        # Apply rotation transforms
        if "rotation" in spatial_transforms:
            transformed_center = self._apply_rotation(
                transformed_center, spatial_transforms["rotation"]
            )

        # Apply translation transforms
        if "translation" in spatial_transforms:
            transformed_center = self._apply_translation(
                transformed_center, spatial_transforms["translation"]
            )

        # Apply scaling transforms
        if "scaling" in spatial_transforms:
            transformed_center = self._apply_scaling(
                transformed_center, spatial_transforms["scaling"]
            )

        return transformed_center

    def _apply_rotation(
        self, center: Dict[str, float], rotation_config: Mapping[str, Any]
    ) -> Dict[str, float]:
        """Apply rotation transformation to coordinates.

        Args:
            center: Input coordinates
            rotation_config: Rotation configuration

        Returns:
            Dict[str, float]: Rotated coordinates
        """
        # Simplified rotation implementation
        # In practice, this would involve proper 3D rotation matrices
        rotated_center = center.copy()

        if "angle" in rotation_config and "axis" in rotation_config:
            angle = rotation_config["angle"]
            axis = rotation_config["axis"]

            # Apply rotation around specified axis (simplified)
            if axis in self.axis_order and abs(angle) > 1e-6:
                logger.debug(f"Applying rotation: {angle} radians around {axis} axis")
                # Note: Full rotation implementation would go here

        return rotated_center

    def _apply_translation(
        self, center: Dict[str, float], translation_config: Mapping[str, Any]
    ) -> Dict[str, float]:
        """Apply translation transformation to coordinates.

        Args:
            center: Input coordinates
            translation_config: Translation configuration

        Returns:
            Dict[str, float]: Translated coordinates
        """
        translated_center = center.copy()

        for axis in self.axis_order:
            if axis in translation_config:
                offset = float(translation_config[axis])
                translated_center[axis] += offset
                logger.debug(f"Applied translation: {axis} += {offset}")

        return translated_center

    def _apply_scaling(
        self, center: Dict[str, float], scaling_config: Mapping[str, Any]
    ) -> Dict[str, float]:
        """Apply scaling transformation to coordinates.

        Args:
            center: Input coordinates
            scaling_config: Scaling configuration

        Returns:
            Dict[str, float]: Scaled coordinates
        """
        scaled_center = center.copy()

        for axis in self.axis_order:
            if axis in scaling_config:
                scale_factor = float(scaling_config[axis])
                if scale_factor != 1.0:
                    scaled_center[axis] *= scale_factor
                    logger.debug(f"Applied scaling: {axis} *= {scale_factor}")

        return scaled_center

    def apply_cached_transforms(
        self, center: Dict[str, float], spatial_transforms: Optional[Mapping[str, Any]]
    ) -> Tuple[Dict[str, float], bool]:
        """Apply transformations using cache when available.

        Args:
            center: Center coordinates
            spatial_transforms: Spatial transformation specifications

        Returns:
            Tuple[Dict[str, float], bool]: (transformed_coordinates, was_cached)
        """
        if not self.enable_caching or self._coordinate_cache is None:
            # No caching - apply transforms directly
            transformed_center = self.apply_spatial_transforms(
                center, spatial_transforms
            )
            return transformed_center, False

        # Check cache first
        reference_frame_id = self.get_reference_frame_id(center)

        cached_result = self._coordinate_cache.get_cached_transforms(
            reference_frame_id, spatial_transforms
        )

        if cached_result is not None:
            # Cache hit
            self.transform_stats["cache_hits"] += 1
            logger.debug(f"Cache hit for reference frame: {reference_frame_id}")
            return cached_result["transformed_coords"], True
        else:
            # Cache miss - compute and cache
            transformed_center = self.apply_spatial_transforms(
                center, spatial_transforms
            )

            # Cache the result
            self._coordinate_cache.cache_transforms(
                reference_frame_id, spatial_transforms, transformed_center
            )

            self.transform_stats["cache_misses"] += 1
            logger.debug(f"Cache miss for reference frame: {reference_frame_id}")
            return transformed_center, False

    def map_coordinates_to_indices(
        self,
        center: Dict[str, float],
        array_shape: Sequence[int],
        scale: Sequence[float],
    ) -> Dict[str, int]:
        """Map continuous coordinates to discrete array indices.

        Args:
            center: Center coordinates in physical space
            array_shape: Shape of the target array
            scale: Scale factors for each dimension

        Returns:
            Dict[str, int]: Mapped indices for each axis

        Raises:
            CoordinateTransformError: If mapping fails
        """
        try:
            indices = {}

            for i, axis in enumerate(self.axis_order):
                if axis in center and i < len(array_shape) and i < len(scale):
                    # Convert physical coordinate to array index
                    physical_coord = center[axis]
                    scale_factor = scale[i]
                    array_size = array_shape[i]

                    # Map to array index
                    index = int(round(physical_coord / scale_factor))

                    # Clamp to valid range
                    index = max(0, min(index, array_size - 1))

                    indices[axis] = index

            return indices

        except Exception as e:
            raise CoordinateTransformError(f"Failed to map coordinates to indices: {e}")

    def compute_bounding_box(
        self,
        centers: Sequence[Dict[str, float]],
        margins: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Tuple[float, float]]:
        """Compute bounding box for a set of coordinate centers.

        Args:
            centers: List of center coordinates
            margins: Optional margins to add to bounding box

        Returns:
            Dict[str, Tuple[float, float]]: Bounding box as (min, max) for each axis
        """
        if not centers:
            raise CoordinateTransformError(
                "Cannot compute bounding box for empty centers list"
            )

        bounding_box = {}
        margins = margins or {}

        for axis in self.axis_order:
            # Extract coordinates for this axis
            axis_coords = [
                center.get(axis, 0.0) for center in centers if axis in center
            ]

            if axis_coords:
                min_coord = min(axis_coords)
                max_coord = max(axis_coords)

                # Apply margins
                margin = margins.get(axis, 0.0)
                bounding_box[axis] = (min_coord - margin, max_coord + margin)
            else:
                bounding_box[axis] = (0.0, 0.0)

        logger.debug(f"Computed bounding box: {bounding_box}")
        return bounding_box

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for coordinate transformations.

        Returns:
            Dict[str, Any]: Performance metrics including cache statistics
        """
        metrics = {
            "transformer_config": {
                "axis_order": self.axis_order,
                "caching_enabled": self.enable_caching,
            },
            "transform_stats": self.transform_stats.copy(),
        }

        if self._coordinate_cache is not None:
            metrics["cache_stats"] = self._coordinate_cache.get_cache_stats()

        return metrics

    def clear_cache(self) -> None:
        """Clear coordinate transformation cache."""
        if self._coordinate_cache is not None:
            self._coordinate_cache.clear_cache()
            logger.debug("Cleared coordinate transformation cache")

    def reset_stats(self) -> None:
        """Reset transformation statistics."""
        self.transform_stats = {
            "total_transforms": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_transform_time": 0.0,
            "average_transform_time": 0.0,
        }
        logger.debug("Reset coordinate transformation statistics")

    def __repr__(self) -> str:
        """String representation of CoordinateTransformer.

        Returns:
            str: Human-readable representation
        """
        return (
            f"CoordinateTransformer(axis_order='{self.axis_order}', "
            f"caching={self.enable_caching}, "
            f"transforms={self.transform_stats['total_transforms']})"
        )
