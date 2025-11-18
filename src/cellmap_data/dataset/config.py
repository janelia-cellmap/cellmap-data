"""
Dataset Configuration Module for CellMap Dataset Architecture.

Centralized configuration management, parameter validation, and deprecation
handling. Extracted from monolithic CellMapDataset to improve maintainability
and provide consistent configuration patterns.
"""

import warnings
from typing import Any, Callable, Mapping, Optional, Sequence, Union
import torch
import tensorstore
from pydantic import BaseModel, Field, validator

from ..utils.error_handling import ErrorMessages, validate_parameter_required
from ..utils.logging_config import get_logger

logger = get_logger("dataset_config")


class ArrayConfigSpec(BaseModel):
    """Configuration specification for array metadata.
    
    Defines the structure and validation for array configuration
    dictionaries used throughout the dataset system.
    """
    shape: Sequence[Union[int, float]] = Field(..., min_items=3, max_items=3)
    scale: Sequence[Union[int, float]] = Field(..., min_items=3, max_items=3)
    
    @validator('shape')
    def validate_shape(cls, v):
        """Validate array shape specification."""
        if not all(isinstance(x, (int, float)) and x > 0 for x in v):
            raise ValueError("Array shape must contain positive numbers")
        return v
        
    @validator('scale')
    def validate_scale(cls, v):
        """Validate array scale specification."""
        if not all(isinstance(x, (int, float)) and x > 0 for x in v):
            raise ValueError("Array scale must contain positive numbers")
        return v


class DatasetConfig:
    """Configuration manager for CellMapDataset with validation and migration.
    
    Handles parameter validation, deprecation warnings, type conversion,
    and configuration integrity checking. Provides a centralized point
    for all dataset configuration management.
    
    Attributes:
        input_path: Validated input data path
        target_path: Validated target data path
        classes: List of segmentation classes
        input_arrays: Validated input array specifications
        target_arrays: Validated target array specifications
        spatial_transforms: Spatial transformation configurations
        raw_value_transforms: Value transforms for input data
        target_value_transforms: Value transforms for target data
        class_relationships: Class relationship mappings
        is_train: Training mode flag
        axis_order: Spatial axes ordering
        context: TensorStore context
        rng: Random number generator
        force_has_data: Force data availability flag
        empty_value: Fill value for empty regions
        pad: Padding enabled flag
        device: Device specification
        max_workers: Maximum worker threads
        
    Examples:
        >>> config = DatasetConfig(
        ...     input_path="/data/raw",
        ...     target_path="/data/labels",
        ...     classes=["class1", "class2"],
        ...     input_arrays={"raw": {"shape": [64, 64, 64], "scale": [4, 4, 4]}},
        ...     target_arrays={"labels": {"shape": [64, 64, 64], "scale": [4, 4, 4]}}
        ... )
        >>> config.validate()
        >>> # Configuration ready for use
    """
    
    def __init__(
        self,
        input_path: Optional[str] = None,
        target_path: Optional[str] = None,
        classes: Optional[Sequence[str]] = None,
        input_arrays: Optional[Mapping[str, Mapping[str, Sequence[Union[int, float]]]]] = None,
        target_arrays: Optional[Mapping[str, Mapping[str, Sequence[Union[int, float]]]]] = None,
        spatial_transforms: Optional[Mapping[str, Mapping]] = None,
        raw_value_transforms: Optional[Callable] = None,
        target_value_transforms: Optional[
            Union[Callable, Sequence[Callable], Mapping[str, Callable]]
        ] = None,
        class_relationships: Optional[Mapping[str, Sequence[str]]] = None,
        is_train: bool = False,
        axis_order: str = "zyx",
        context: Optional[tensorstore.Context] = None,
        rng: Optional[torch.Generator] = None,
        force_has_data: bool = False,
        empty_value: Union[float, int] = torch.nan,
        pad: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        max_workers: Optional[int] = None,
        # Deprecated parameters for backward compatibility
        raw_path: Optional[str] = None,
        class_relation_dict: Optional[Mapping[str, Sequence[str]]] = None,
    ):
        """Initialize dataset configuration with validation.
        
        Args:
            input_path: Path to input/raw data files
            target_path: Path to target/ground truth data files
            classes: List of segmentation classes
            input_arrays: Input array specifications
            target_arrays: Target array specifications
            spatial_transforms: Spatial transformation configurations
            raw_value_transforms: Value transforms for input data
            target_value_transforms: Value transforms for target data
            class_relationships: Class relationship mappings
            is_train: Whether dataset is in training mode
            axis_order: Order of spatial axes
            context: TensorStore context
            rng: Random number generator
            force_has_data: Force data availability
            empty_value: Fill value for empty regions
            pad: Whether to enable padding
            device: Device specification
            max_workers: Maximum worker threads
            raw_path: **Deprecated** - use input_path instead
            class_relation_dict: **Deprecated** - use class_relationships instead
            
        Raises:
            ValueError: If configuration is invalid or conflicting parameters provided
        """
        # Handle deprecated parameters with warnings
        self._handle_deprecated_parameters(
            input_path, raw_path, class_relationships, class_relation_dict
        )
        
        # Store normalized parameters
        self.input_path = input_path if input_path is not None else raw_path
        self.target_path = target_path
        self.classes = list(classes) if classes is not None else []
        self.raw_only = classes is None
        self.input_arrays = dict(input_arrays) if input_arrays is not None else {}
        self.target_arrays = dict(target_arrays) if target_arrays is not None else {}
        self.spatial_transforms = dict(spatial_transforms) if spatial_transforms is not None else {}
        self.raw_value_transforms = raw_value_transforms
        self.target_value_transforms = target_value_transforms
        self.class_relationships = (
            dict(class_relationships) if class_relationships is not None 
            else dict(class_relation_dict) if class_relation_dict is not None 
            else {}
        )
        self.is_train = is_train
        self.axis_order = axis_order
        self.context = context
        self.rng = rng
        self.force_has_data = force_has_data
        self.empty_value = empty_value
        self.pad = pad
        self.device = device
        self.max_workers = max_workers
        
        # Validate configuration
        self.validate()
        
        logger.debug("DatasetConfig initialized with validation complete")
        
    def _handle_deprecated_parameters(
        self,
        input_path: Optional[str],
        raw_path: Optional[str],
        class_relationships: Optional[Mapping[str, Sequence[str]]],
        class_relation_dict: Optional[Mapping[str, Sequence[str]]],
    ) -> None:
        """Handle deprecated parameter migration with warnings.
        
        Args:
            input_path: New parameter name
            raw_path: Deprecated parameter name
            class_relationships: New parameter name
            class_relation_dict: Deprecated parameter name
            
        Raises:
            ValueError: If both new and deprecated parameters provided
        """
        # Handle raw_path -> input_path migration
        if raw_path is not None:
            if input_path is not None:
                raise ValueError(
                    f"{ErrorMessages.format_conflicting_parameters('input_path', 'raw_path')}. "
                    "Please use 'input_path' only. 'raw_path' will be removed in a future version."
                )
            warnings.warn(
                "Parameter 'raw_path' is deprecated and will be removed in a future version. "
                "Please use 'input_path' instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            
        # Handle class_relation_dict -> class_relationships migration
        if class_relation_dict is not None:
            if class_relationships is not None:
                raise ValueError(
                    "Cannot specify both 'class_relationships' and deprecated 'class_relation_dict'. "
                    "Please use 'class_relationships' only. 'class_relation_dict' will be removed in a future version."
                )
            warnings.warn(
                "Parameter 'class_relation_dict' is deprecated and will be removed in a future version. "
                "Please use 'class_relationships' instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            
    def validate(self) -> None:
        """Validate configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate required parameters
        validate_parameter_required("input_path", self.input_path)
        validate_parameter_required("target_path", self.target_path)
        validate_parameter_required("input_arrays", self.input_arrays)
        
        # Validate array configurations
        self._validate_array_configs()
        
        # Validate axis order
        self._validate_axis_order()
        
        # Validate class configuration
        self._validate_classes()
        
        # Validate transforms
        self._validate_transforms()
        
        logger.debug("Configuration validation passed")
        
    def _validate_array_configs(self) -> None:
        """Validate array configuration dictionaries."""
        for array_name, array_config in self.input_arrays.items():
            try:
                ArrayConfigSpec(**array_config)
            except Exception as e:
                raise ValueError(f"Invalid input array config for '{array_name}': {e}") from e
                
        for array_name, array_config in self.target_arrays.items():
            try:
                ArrayConfigSpec(**array_config)
            except Exception as e:
                raise ValueError(f"Invalid target array config for '{array_name}': {e}") from e
                
    def _validate_axis_order(self) -> None:
        """Validate axis order specification."""
        if not isinstance(self.axis_order, str):
            raise ValueError("axis_order must be a string")
        if len(self.axis_order) != 3:
            raise ValueError("axis_order must contain exactly 3 characters")
        if set(self.axis_order.lower()) != {"x", "y", "z"}:
            raise ValueError("axis_order must contain exactly 'x', 'y', 'z' characters")
            
    def _validate_classes(self) -> None:
        """Validate class configuration."""
        if not self.raw_only and not self.classes:
            raise ValueError("Classes must be specified for non-raw-only mode")
        if self.classes and len(set(self.classes)) != len(self.classes):
            raise ValueError("Class names must be unique")
            
    def _validate_transforms(self) -> None:
        """Validate transform configurations."""
        # Validate spatial transforms structure
        if self.spatial_transforms:
            for transform_name, transform_config in self.spatial_transforms.items():
                if not isinstance(transform_config, dict):
                    raise ValueError(f"Spatial transform '{transform_name}' must be a dictionary")
                    
        # Validate target value transforms structure
        if self.target_value_transforms is not None:
            if not (
                callable(self.target_value_transforms) or
                isinstance(self.target_value_transforms, (list, tuple)) or
                isinstance(self.target_value_transforms, dict)
            ):
                raise ValueError("target_value_transforms must be callable, sequence, or mapping")
                
    def get_array_names(self) -> dict[str, list[str]]:
        """Get dictionary of array names by type.
        
        Returns:
            Dictionary with 'input' and 'target' keys containing array names
        """
        return {
            "input": list(self.input_arrays.keys()),
            "target": list(self.target_arrays.keys()),
        }
        
    def get_array_shapes(self) -> dict[str, dict[str, Sequence[Union[int, float]]]]:
        """Get dictionary of array shapes by name.
        
        Returns:
            Dictionary mapping array names to their shape specifications
        """
        shapes = {}
        for name, config in self.input_arrays.items():
            shapes[name] = config["shape"]
        for name, config in self.target_arrays.items():
            shapes[name] = config["shape"]
        return shapes
        
    def get_array_scales(self) -> dict[str, dict[str, Sequence[Union[int, float]]]]:
        """Get dictionary of array scales by name.
        
        Returns:
            Dictionary mapping array names to their scale specifications
        """
        scales = {}
        for name, config in self.input_arrays.items():
            scales[name] = config["scale"]
        for name, config in self.target_arrays.items():
            scales[name] = config["scale"]
        return scales
        
    def is_2d_array_config(self) -> bool:
        """Check if configuration specifies 2D arrays.
        
        Returns:
            True if any array has 2D shape specification
        """
        from ..utils import is_array_2D
        return is_array_2D(self.input_arrays) or is_array_2D(self.target_arrays)
        
    def copy(self) -> "DatasetConfig":
        """Create a copy of the configuration.
        
        Returns:
            New DatasetConfig instance with copied parameters
        """
        return DatasetConfig(
            input_path=self.input_path,
            target_path=self.target_path,
            classes=self.classes.copy() if self.classes else None,
            input_arrays=dict(self.input_arrays),
            target_arrays=dict(self.target_arrays),
            spatial_transforms=dict(self.spatial_transforms),
            raw_value_transforms=self.raw_value_transforms,
            target_value_transforms=self.target_value_transforms,
            class_relationships=dict(self.class_relationships),
            is_train=self.is_train,
            axis_order=self.axis_order,
            context=self.context,
            rng=self.rng,
            force_has_data=self.force_has_data,
            empty_value=self.empty_value,
            pad=self.pad,
            device=self.device,
            max_workers=self.max_workers,
        )
        
    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        return {
            "input_path": self.input_path,
            "target_path": self.target_path,
            "classes": self.classes,
            "raw_only": self.raw_only,
            "input_arrays": self.input_arrays,
            "target_arrays": self.target_arrays,
            "spatial_transforms": self.spatial_transforms,
            "class_relationships": self.class_relationships,
            "is_train": self.is_train,
            "axis_order": self.axis_order,
            "force_has_data": self.force_has_data,
            "empty_value": self.empty_value,
            "pad": self.pad,
            "device": str(self.device) if self.device else None,
            "max_workers": self.max_workers,
        }
        
    def __str__(self) -> str:
        """String representation of configuration."""
        return (f"DatasetConfig(input_path='{self.input_path}', "
                f"target_path='{self.target_path}', classes={len(self.classes)}, "
                f"input_arrays={len(self.input_arrays)}, target_arrays={len(self.target_arrays)})")
        
    def __repr__(self) -> str:
        """Detailed representation of configuration."""
        return f"DatasetConfig({self.to_dict()})"
