"""
Coordinate Transformation Module for CellMap Dataset Architecture.

Specialized coordinate transformation, spatial transform generation, and
performance-optimized coordinate operations. Extracted from monolithic
CellMapDataset to improve maintainability and enable reuse.
"""

import time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union
import numpy as np
import torch

from ..utils.logging_config import get_logger
from ..utils.coordinate_cache import CoordinateTransformCache
from ..exceptions import CoordinateTransformError

logger = get_logger("coordinate_manager")


class CoordinateTransformer:
    """Coordinate transformation and spatial transform management.

    Handles all coordinate-related operations including transformation generation,
    coordinate mapping, bounding box calculations, and performance optimization
    through caching. Designed for high-performance data loading scenarios.

    Attributes:
        axis_order: Order of spatial axes ('zyx', 'xyz', etc.)
        rng: Random number generator for reproducible transformations
        cache: Coordinate transformation cache for performance

    Examples:
        >>> transformer = CoordinateTransformer(axis_order="zyx")
        >>> center = {"z": 100, "y": 200, "x": 300}
        >>> transforms = transformer.generate_spatial_transforms(
        ...     {"rotate": {"axes": {"z": [-180, 180]}}}
        ... )
        >>> transforms
        {'rotate': {'z': 45.0}}
    """

    def __init__(
        self,
        axis_order: str = "zyx",
        rng: Optional[torch.Generator] = None,
        enable_caching: bool = True,
    ):
        """Initialize coordinate transformer.

        Args:
            axis_order: Order of spatial axes in data arrays
            rng: Random number generator for reproducible transformations
            enable_caching: Whether to enable coordinate transformation caching

        Raises:
            ValueError: If axis_order is invalid
        """
        if len(axis_order) != 3 or set(axis_order.lower()) != {"x", "y", "z"}:
            raise ValueError("axis_order must contain exactly 'x', 'y', 'z' characters")

        self.axis_order = axis_order
        self._rng = rng if rng is not None else torch.Generator()
        self._enable_caching = enable_caching

        # Initialize coordinate transformation cache for performance
        if self._enable_caching:
            self._coordinate_cache = CoordinateTransformCache()
        else:
            self._coordinate_cache = None

        # Performance tracking
        self._transform_stats = {
            "total_transforms": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_transform_time": 0.0,
        }

        logger.debug(
            f"CoordinateTransformer initialized with axis_order='{axis_order}', caching={enable_caching}"
        )

    def generate_spatial_transforms(
        self, spatial_transform_config: Optional[Mapping[str, Any]] = None
    ) -> Optional[Mapping[str, Any]]:
        """Generate spatial transformations based on configuration.

        Args:
            spatial_transform_config: Configuration dictionary for spatial transforms

        Returns:
            Generated spatial transformation parameters or None if no config

        Raises:
            ValueError: If spatial transform configuration is invalid
        """
        if spatial_transform_config is None:
            return None

        start_time = time.time()
        spatial_transforms = {}

        try:
            for transform, params in spatial_transform_config.items():
                if transform == "mirror":
                    spatial_transforms[transform] = self._generate_mirror_transform(
                        params
                    )
                elif transform == "transpose":
                    spatial_transforms[transform] = self._generate_transpose_transform(
                        params
                    )
                elif transform == "rotate":
                    spatial_transforms[transform] = self._generate_rotate_transform(
                        params
                    )
                else:
                    raise ValueError(f"Unknown spatial transform: {transform}")

            # Update performance stats
            self._transform_stats["total_transforms"] += 1
            self._transform_stats["total_transform_time"] += time.time() - start_time

            logger.debug(f"Generated spatial transforms: {spatial_transforms}")
            return spatial_transforms

        except Exception as e:
            logger.error(f"Failed to generate spatial transforms: {e}")
            raise ValueError(f"Invalid spatial transform configuration: {e}") from e

    def _generate_mirror_transform(self, params: Mapping[str, Any]) -> list[str]:
        """Generate mirror transform parameters.

        Args:
            params: Mirror transform parameters with axes and probabilities

        Returns:
            List of axes to mirror
        """
        mirrored_axes = []
        for axis, prob in params["axes"].items():
            if torch.rand(1, generator=self._rng).item() < prob:
                mirrored_axes.append(axis)
        return mirrored_axes

    def _generate_transpose_transform(
        self, params: Mapping[str, Any]
    ) -> Dict[str, int]:
        """Generate transpose transform parameters.

        Args:
            params: Transpose transform parameters with axes to shuffle

        Returns:
            Dictionary mapping axes to new positions
        """
        # Create axis mapping
        axes = {axis: i for i, axis in enumerate(self.axis_order)}

        # Get axes to shuffle
        shuffled_axes = [axes[a] for a in params["axes"]]

        # Randomly permute the selected axes
        shuffled_indices = torch.randperm(len(shuffled_axes), generator=self._rng)
        shuffled_axes = [shuffled_axes[i] for i in shuffled_indices]

        # Map shuffled axes back to axis names
        shuffled_mapping = {
            axis: shuffled_axes[i] for i, axis in enumerate(params["axes"])
        }

        # Update full axis mapping
        axes.update(shuffled_mapping)
        return axes

    def _generate_rotate_transform(self, params: Mapping[str, Any]) -> Dict[str, float]:
        """Generate rotation transform parameters.

        Args:
            params: Rotation parameters with axes and angle ranges

        Returns:
            Dictionary mapping axes to rotation angles
        """
        rotation_angles = {}
        for axis, limits in params["axes"].items():
            angle = torch.rand(1, generator=self._rng).item()
            angle = angle * (limits[1] - limits[0]) + limits[0]
            rotation_angles[axis] = angle
        return rotation_angles

    def convert_index_to_center(
        self, idx: int, sampling_box_shape: Mapping[str, Union[int, float]]
    ) -> Dict[str, float]:
        """Convert linear index to center coordinates.

        Args:
            idx: Linear index to convert
            sampling_box_shape: Shape of sampling box for each axis

        Returns:
            Dictionary mapping axis names to center coordinates

        Raises:
            CoordinateTransformError: If coordinate conversion fails
        """
        try:
            # Convert to numpy for unravel_index
            shape_tuple = tuple(
                int(sampling_box_shape[axis]) for axis in self.axis_order
            )

            # Use numpy's unravel_index for efficient conversion
            coords = np.unravel_index(idx, shape_tuple)

            # Map back to axis names and convert to float
            center = {axis: float(coords[i]) for i, axis in enumerate(self.axis_order)}

            logger.debug(f"Converted index {idx} to center coordinates: {center}")
            return center

        except (ValueError, IndexError) as e:
            shape_info = {c: sampling_box_shape[c] for c in self.axis_order}
            dataset_length = int(np.prod(list(sampling_box_shape.values())))
            raise CoordinateTransformError(
                f"Failed to convert index {idx} to coordinates. "
                f"Sampling box shape: {shape_info}, dataset length: {dataset_length}. "
                f"Original error: {e}"
            ) from e
        except Exception as e:
            raise CoordinateTransformError(
                f"Unexpected error in coordinate transformation for index {idx}: {e}"
            ) from e

    def convert_center_to_index(
        self,
        center: Mapping[str, Union[int, float]],
        sampling_box_shape: Mapping[str, Union[int, float]],
    ) -> int:
        """Convert center coordinates to linear index.

        Args:
            center: Dictionary mapping axis names to coordinates
            sampling_box_shape: Shape of sampling box for each axis

        Returns:
            Linear index corresponding to center coordinates

        Raises:
            CoordinateTransformError: If coordinate conversion fails
        """
        try:
            # Extract coordinates in axis order
            coords = tuple(int(center[axis]) for axis in self.axis_order)

            # Extract shape in axis order
            shape_tuple = tuple(
                int(sampling_box_shape[axis]) for axis in self.axis_order
            )

            # Use numpy's ravel_multi_index for efficient conversion
            idx = np.ravel_multi_index(coords, shape_tuple)

            logger.debug(f"Converted center {center} to index: {idx}")
            return int(idx)

        except (ValueError, IndexError) as e:
            shape_info = {c: sampling_box_shape[c] for c in self.axis_order}
            raise CoordinateTransformError(
                f"Failed to convert center {center} to index. "
                f"Sampling box shape: {shape_info}. "
                f"Original error: {e}"
            ) from e
        except Exception as e:
            raise CoordinateTransformError(
                f"Unexpected error in coordinate transformation for center {center}: {e}"
            ) from e

    def calculate_bounding_box(
        self, array_configs: Mapping[str, Mapping[str, Sequence[Union[int, float]]]]
    ) -> Dict[str, Dict[str, Union[int, float]]]:
        """Calculate combined bounding box from array configurations.

        Args:
            array_configs: Dictionary of array configurations with shape and scale

        Returns:
            Dictionary with bounding box coordinates for each axis
        """
        if not array_configs:
            return {axis: {"min": 0, "max": 0} for axis in self.axis_order}

        # Initialize bounding box
        bounding_box = {}
        for axis in self.axis_order:
            bounding_box[axis] = {"min": float("inf"), "max": float("-inf")}

        # Process each array configuration
        for array_name, config in array_configs.items():
            shape = config["shape"]
            scale = config["scale"]

            # Calculate array bounds
            for i, axis in enumerate(self.axis_order):
                array_size = shape[i] * scale[i]
                bounding_box[axis]["min"] = min(bounding_box[axis]["min"], 0)
                bounding_box[axis]["max"] = max(bounding_box[axis]["max"], array_size)

        # Convert inf values to 0 if no valid arrays
        for axis in self.axis_order:
            if bounding_box[axis]["min"] == float("inf"):
                bounding_box[axis]["min"] = 0
            if bounding_box[axis]["max"] == float("-inf"):
                bounding_box[axis]["max"] = 0

        logger.debug(f"Calculated bounding box: {bounding_box}")
        return bounding_box

    def calculate_sampling_box(
        self,
        bounding_box: Mapping[str, Mapping[str, Union[int, float]]],
        array_shapes: Mapping[str, Sequence[Union[int, float]]],
    ) -> Dict[str, Union[int, float]]:
        """Calculate sampling box for valid center sampling.

        Args:
            bounding_box: Combined bounding box of all arrays
            array_shapes: Dictionary of array shapes

        Returns:
            Dictionary with sampling box dimensions for each axis
        """
        if not array_shapes:
            return {axis: 1 for axis in self.axis_order}

        # Find maximum array shape for each axis
        max_shapes = {axis: 0.0 for axis in self.axis_order}
        for shape in array_shapes.values():
            for i, axis in enumerate(self.axis_order):
                max_shapes[axis] = max(max_shapes[axis], float(shape[i]))

        # Calculate sampling box
        sampling_box = {}
        for axis in self.axis_order:
            box_size = bounding_box[axis]["max"] - bounding_box[axis]["min"]
            half_array = max_shapes[axis] / 2
            sampling_size = max(1, box_size - 2 * half_array)
            sampling_box[axis] = sampling_size

        logger.debug(f"Calculated sampling box: {sampling_box}")
        return sampling_box

    def get_reference_frame_id(self, center: Mapping[str, float]) -> str:
        """Generate reference frame ID for caching.

        Args:
            center: Center coordinates

        Returns:
            Reference frame identifier for caching
        """
        # Create a stable ID based on center coordinates
        coord_str = "_".join(
            f"{axis}:{int(center[axis])}" for axis in sorted(center.keys())
        )
        return f"frame_{coord_str}"

    def apply_cached_transforms(
        self,
        reference_frame_id: str,
        spatial_transforms: Optional[Mapping[str, Any]],
        apply_transform_func: Callable,
    ) -> bool:
        """Apply cached transformations if available.

        Args:
            reference_frame_id: Reference frame identifier
            spatial_transforms: Current spatial transformation parameters
            apply_transform_func: Function to apply cached transforms

        Returns:
            True if cached transforms were applied, False otherwise
        """
        if not self._enable_caching or self._coordinate_cache is None:
            return False

        try:
            cached_entry = self._coordinate_cache.get_cached_transforms(
                reference_frame_id, spatial_transforms
            )

            if cached_entry is not None:
                # Apply cached transforms
                apply_transform_func(cached_entry["transforms"])

                # Update performance stats
                self._transform_stats["cache_hits"] += 1

                logger.debug(
                    f"Applied cached transforms for frame {reference_frame_id}"
                )
                return True
            else:
                self._transform_stats["cache_misses"] += 1
                return False

        except Exception as e:
            logger.warning(f"Failed to apply cached transforms: {e}")
            return False

    def cache_transforms(
        self,
        reference_frame_id: str,
        spatial_transforms: Optional[Mapping[str, Any]],
        computed_transforms: Any,
        transform_time: float = 0.0,
    ) -> None:
        """Cache computed transformations for future use.

        Args:
            reference_frame_id: Reference frame identifier
            spatial_transforms: Spatial transformation parameters
            computed_transforms: Computed transformation results
            transform_time: Time taken to compute transforms
        """
        if not self._enable_caching or self._coordinate_cache is None:
            return

        try:
            self._coordinate_cache.cache_transforms(
                reference_frame_id,
                spatial_transforms,
                computed_transforms,
                {"transform_time": transform_time},
            )
            logger.debug(f"Cached transforms for frame {reference_frame_id}")
        except Exception as e:
            logger.warning(f"Failed to cache transforms: {e}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get coordinate transformation performance metrics.

        Returns:
            Dictionary with performance statistics
        """
        total_requests = (
            self._transform_stats["cache_hits"] + self._transform_stats["cache_misses"]
        )
        cache_hit_rate = (
            self._transform_stats["cache_hits"] / total_requests
            if total_requests > 0
            else 0.0
        )

        metrics = {
            "total_transforms": self._transform_stats["total_transforms"],
            "total_transform_time": self._transform_stats["total_transform_time"],
            "average_transform_time": (
                self._transform_stats["total_transform_time"]
                / max(1, self._transform_stats["total_transforms"])
            ),
            "cache_hits": self._transform_stats["cache_hits"],
            "cache_misses": self._transform_stats["cache_misses"],
            "cache_hit_rate": cache_hit_rate,
            "caching_enabled": self._enable_caching,
        }

        if self._coordinate_cache is not None:
            try:
                cache_metrics = self._coordinate_cache.get_cache_stats()
                metrics.update({f"cache_{k}": v for k, v in cache_metrics.items()})
            except AttributeError:
                # Handle case where cache doesn't have get_cache_stats method
                metrics["cache_enabled"] = True

        return metrics

    def clear_cache(self) -> None:
        """Clear coordinate transformation cache."""
        if self._coordinate_cache is not None:
            self._coordinate_cache.clear_cache()
            logger.debug("Coordinate transformation cache cleared")

    def __str__(self) -> str:
        """String representation of coordinate transformer."""
        return (
            f"CoordinateTransformer(axis_order='{self.axis_order}', "
            f"caching={self._enable_caching})"
        )

    def __repr__(self) -> str:
        """Detailed representation of coordinate transformer."""
        return (
            f"CoordinateTransformer(axis_order='{self.axis_order}', "
            f"caching={self._enable_caching}, "
            f"stats={self._transform_stats})"
        )
