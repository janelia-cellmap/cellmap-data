"""
CellMap Dataset Validation Module

Centralized validation logic extracted from CellMapDataset to provide
consistent validation across dataset operations and improve maintainability.

This module consolidates parameter validation, data integrity checks, and
configuration validation that was previously embedded in the monolithic
CellMapDataset class.
"""

import os
from pathlib import Path
from typing import Dict, Any, Mapping, Sequence, Optional, Union, Callable
import torch
import numpy as np

from ..utils.error_handling import ErrorMessages, validate_parameter_required
from ..utils.logging_config import get_logger
from ..exceptions import ValidationError

logger = get_logger("dataset_validator")


class DatasetValidator:
    """Centralized validation for CellMap dataset operations.
    
    Provides comprehensive validation of dataset parameters, array configurations,
    data sources, and integrity checks extracted from CellMapDataset for better
    separation of concerns and maintainability.
    
    Attributes:
        strict_mode: Whether to enforce strict validation rules
        validation_cache: Cache for expensive validation operations
        
    Examples:
        >>> validator = DatasetValidator()
        >>> validator.validate_dataset_config(config)
        
        >>> validator = DatasetValidator(strict_mode=True)
        >>> validator.validate_data_sources(input_path, target_path)
    """
    
    def __init__(self, strict_mode: bool = False):
        """Initialize dataset validator.
        
        Args:
            strict_mode: Whether to enforce strict validation rules.
                        If True, performs more comprehensive but slower validation.
        """
        self.strict_mode = strict_mode
        self.validation_cache: Dict[str, Any] = {}
        
        logger.debug(f"DatasetValidator initialized with strict_mode={strict_mode}")
        
    def validate_required_parameters(
        self,
        input_path: Optional[str] = None,
        target_path: Optional[str] = None,
        input_arrays: Optional[Mapping[str, Mapping[str, Sequence[int | float]]]] = None,
    ) -> None:
        """Validate required dataset parameters.
        
        Args:
            input_path: Path to input data
            target_path: Path to target data  
            input_arrays: Input array specifications
            
        Raises:
            ValueError: If required parameters are missing
        """
        validate_parameter_required("input_path", input_path)
        validate_parameter_required("target_path", target_path)
        validate_parameter_required("input_arrays", input_arrays)
        
        logger.debug("Required parameters validation passed")
        
    def validate_deprecated_parameters(
        self,
        raw_path: Optional[str] = None,
        input_path: Optional[str] = None,
        class_relation_dict: Optional[Mapping[str, Sequence[str]]] = None,
        class_relationships: Optional[Mapping[str, Sequence[str]]] = None,
    ) -> tuple[Optional[str], Optional[Mapping[str, Sequence[str]]]]:
        """Handle deprecated parameter validation and migration.
        
        Args:
            raw_path: Deprecated input path parameter
            input_path: Current input path parameter
            class_relation_dict: Deprecated class relationships parameter
            class_relationships: Current class relationships parameter
            
        Returns:
            tuple: Migrated (input_path, class_relationships) values
            
        Raises:
            ValueError: If both deprecated and current parameters are provided
        """
        import warnings
        
        # Handle raw_path deprecation
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
            input_path = raw_path
            
        # Handle class_relation_dict deprecation
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
            class_relationships = class_relation_dict
            
        return input_path, class_relationships
        
    def validate_array_config(
        self,
        arrays: Mapping[str, Mapping[str, Sequence[int | float]]],
        array_type: str = "input"
    ) -> None:
        """Validate array configuration structure and values.
        
        Args:
            arrays: Array specifications to validate
            array_type: Type of arrays ('input' or 'target') for error messages
            
        Raises:
            ValidationError: If array configuration is invalid
        """
        if not arrays:
            logger.debug(f"Empty {array_type} arrays configuration - skipping validation")
            return
            
        for array_name, array_info in arrays.items():
            if not isinstance(array_info, dict):
                raise ValidationError(
                    f"Invalid {array_type} array '{array_name}': configuration must be a dictionary"
                )
                
            # Validate required array fields
            if "shape" not in array_info:
                raise ValidationError(
                    f"Missing 'shape' in {array_type} array '{array_name}' configuration"
                )
                
            if "scale" not in array_info:
                raise ValidationError(
                    f"Missing 'scale' in {array_type} array '{array_name}' configuration"
                )
                
            # Validate shape format
            shape = array_info["shape"]
            if not isinstance(shape, (list, tuple, np.ndarray)) or len(shape) == 0:
                raise ValidationError(
                    f"Invalid 'shape' in {array_type} array '{array_name}': must be non-empty sequence"
                )
                
            # Validate scale format
            scale = array_info["scale"]
            if not isinstance(scale, (list, tuple, np.ndarray)) or len(scale) == 0:
                raise ValidationError(
                    f"Invalid 'scale' in {array_type} array '{array_name}': must be non-empty sequence"
                )
                
            # Validate shape-scale compatibility
            if len(shape) != len(scale):
                raise ValidationError(
                    f"Shape-scale mismatch in {array_type} array '{array_name}': "
                    f"shape has {len(shape)} dimensions, scale has {len(scale)} dimensions"
                )
                
        logger.debug(f"Array configuration validation passed for {len(arrays)} {array_type} arrays")
        
    def validate_data_sources(
        self,
        input_path: str,
        target_path: Optional[str] = None,
        check_accessibility: Optional[bool] = None
    ) -> None:
        """Validate data source paths and accessibility.
        
        Args:
            input_path: Path to input data
            target_path: Path to target data (optional)
            check_accessibility: Whether to check file accessibility.
                               If None, uses strict_mode setting.
                               
        Raises:
            ValidationError: If data sources are invalid or inaccessible
        """
        if check_accessibility is None:
            check_accessibility = self.strict_mode
            
        # Validate input path
        if not input_path or not isinstance(input_path, str):
            raise ValidationError("Input path must be a non-empty string")
            
        if check_accessibility:
            input_path_obj = Path(input_path)
            if not input_path_obj.exists():
                raise ValidationError(f"Input path does not exist: {input_path}")
            if not input_path_obj.is_dir():
                raise ValidationError(f"Input path is not a directory: {input_path}")
                
        # Validate target path if provided
        if target_path is not None:
            if not isinstance(target_path, str):
                raise ValidationError("Target path must be a string")
                
            if check_accessibility and target_path:
                target_path_obj = Path(target_path)
                if not target_path_obj.exists():
                    raise ValidationError(f"Target path does not exist: {target_path}")
                    
        logger.debug("Data source validation passed")
        
    def validate_transforms(
        self,
        spatial_transforms: Optional[Mapping[str, Mapping]] = None,
        raw_value_transforms: Optional[Callable] = None,
        target_value_transforms: Optional[Union[Callable, Sequence[Callable], Mapping[str, Callable]]] = None,
    ) -> None:
        """Validate transformation configurations.
        
        Args:
            spatial_transforms: Spatial transformation configurations
            raw_value_transforms: Raw value transform function
            target_value_transforms: Target value transform function(s)
            
        Raises:
            ValidationError: If transform configurations are invalid
        """
        # Validate spatial transforms
        if spatial_transforms is not None:
            if not isinstance(spatial_transforms, dict):
                raise ValidationError("Spatial transforms must be a dictionary")
                
            for transform_name, transform_config in spatial_transforms.items():
                if not isinstance(transform_config, dict):
                    raise ValidationError(
                        f"Spatial transform '{transform_name}' configuration must be a dictionary"
                    )
                    
        # Validate raw value transforms
        if raw_value_transforms is not None:
            if not callable(raw_value_transforms):
                raise ValidationError("Raw value transforms must be callable")
                
        # Validate target value transforms
        if target_value_transforms is not None:
            if callable(target_value_transforms):
                # Single function - valid
                pass
            elif isinstance(target_value_transforms, (list, tuple)):
                # Sequence of functions
                for i, transform in enumerate(target_value_transforms):
                    if not callable(transform):
                        raise ValidationError(
                            f"Target value transform at index {i} must be callable"
                        )
            elif isinstance(target_value_transforms, dict):
                # Dictionary of functions
                for name, transform in target_value_transforms.items():
                    if not callable(transform):
                        raise ValidationError(
                            f"Target value transform '{name}' must be callable"
                        )
            else:
                raise ValidationError(
                    "Target value transforms must be callable, sequence of callables, or dict of callables"
                )
                
        logger.debug("Transform validation passed")
        
    def validate_class_config(
        self,
        classes: Optional[Sequence[str]] = None,
        class_relationships: Optional[Mapping[str, Sequence[str]]] = None,
    ) -> None:
        """Validate class configuration and relationships.
        
        Args:
            classes: List of class names
            class_relationships: Class relationship mappings
            
        Raises:
            ValidationError: If class configuration is invalid
        """
        # Validate classes
        if classes is not None:
            if not isinstance(classes, (list, tuple)):
                raise ValidationError("Classes must be a sequence")
                
            if len(classes) == 0:
                raise ValidationError("Classes sequence cannot be empty")
                
            for i, class_name in enumerate(classes):
                if not isinstance(class_name, str):
                    raise ValidationError(f"Class at index {i} must be a string")
                if not class_name.strip():
                    raise ValidationError(f"Class at index {i} cannot be empty")
                    
            # Check for duplicates
            if len(set(classes)) != len(classes):
                raise ValidationError("Duplicate class names are not allowed")
                
        # Validate class relationships
        if class_relationships is not None and classes is not None:
            if not isinstance(class_relationships, dict):
                raise ValidationError("Class relationships must be a dictionary")
                
            class_set = set(classes)
            for source_class, related_classes in class_relationships.items():
                if source_class not in class_set:
                    raise ValidationError(
                        f"Class relationship source '{source_class}' not found in classes list"
                    )
                    
                if not isinstance(related_classes, (list, tuple)):
                    raise ValidationError(
                        f"Class relationships for '{source_class}' must be a sequence"
                    )
                    
                for related_class in related_classes:
                    if not isinstance(related_class, str):
                        raise ValidationError(
                            f"Related class '{related_class}' for '{source_class}' must be a string"
                        )
                    if related_class not in class_set:
                        raise ValidationError(
                            f"Related class '{related_class}' for '{source_class}' not found in classes list"
                        )
                        
        logger.debug("Class configuration validation passed")
        
    def validate_device_config(
        self,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """Validate device configuration.
        
        Args:
            device: Device specification to validate
            
        Raises:
            ValidationError: If device configuration is invalid
        """
        if device is not None:
            if isinstance(device, str):
                valid_devices = ["cpu", "cuda", "mps"]
                if device not in valid_devices and not device.startswith("cuda:"):
                    raise ValidationError(
                        f"Invalid device string '{device}'. Must be one of {valid_devices} or 'cuda:N'"
                    )
            elif isinstance(device, torch.device):
                # torch.device is valid
                pass
            else:
                raise ValidationError("Device must be a string or torch.device")
                
        logger.debug("Device configuration validation passed")
        
    def validate_performance_config(
        self,
        max_workers: Optional[int] = None,
        axis_order: str = "zyx",
    ) -> None:
        """Validate performance-related configuration.
        
        Args:
            max_workers: Maximum worker threads
            axis_order: Axis order specification
            
        Raises:
            ValidationError: If performance configuration is invalid
        """
        # Validate max_workers
        if max_workers is not None:
            if not isinstance(max_workers, int):
                raise ValidationError("max_workers must be an integer")
            if max_workers <= 0:
                raise ValidationError("max_workers must be positive")
            if max_workers > 64:  # Reasonable upper limit
                logger.warning(f"max_workers={max_workers} is very high, consider reducing")
                
        # Validate axis_order
        if not isinstance(axis_order, str):
            raise ValidationError("axis_order must be a string")
        if len(axis_order) < 2:
            raise ValidationError("axis_order must have at least 2 dimensions")
        valid_axes = set("xyzctuvw")  # Common axis names
        if not all(c in valid_axes for c in axis_order.lower()):
            logger.warning(f"axis_order '{axis_order}' contains non-standard axis names")
            
        logger.debug("Performance configuration validation passed")
        
    def validate_dataset_config(
        self,
        input_path: Optional[str] = None,
        target_path: Optional[str] = None,
        classes: Optional[Sequence[str]] = None,
        input_arrays: Optional[Mapping[str, Mapping[str, Sequence[int | float]]]] = None,
        target_arrays: Optional[Mapping[str, Mapping[str, Sequence[int | float]]]] = None,
        spatial_transforms: Optional[Mapping[str, Mapping]] = None,
        raw_value_transforms: Optional[Callable] = None,
        target_value_transforms: Optional[Union[Callable, Sequence[Callable], Mapping[str, Callable]]] = None,
        class_relationships: Optional[Mapping[str, Sequence[str]]] = None,
        device: Optional[Union[str, torch.device]] = None,
        max_workers: Optional[int] = None,
        axis_order: str = "zyx",
        # Deprecated parameters
        raw_path: Optional[str] = None,
        class_relation_dict: Optional[Mapping[str, Sequence[str]]] = None,
    ) -> Dict[str, Any]:
        """Comprehensive dataset configuration validation.
        
        Args:
            All dataset configuration parameters
            
        Returns:
            Dict[str, Any]: Validated and normalized configuration
            
        Raises:
            ValidationError: If any validation fails
        """
        logger.debug("Starting comprehensive dataset configuration validation")
        
        # Handle deprecated parameters
        input_path, class_relationships = self.validate_deprecated_parameters(
            raw_path, input_path, class_relation_dict, class_relationships
        )
        
        # Validate required parameters
        self.validate_required_parameters(input_path, target_path, input_arrays)
        
        # Validate array configurations
        if input_arrays:
            self.validate_array_config(input_arrays, "input")
        if target_arrays:
            self.validate_array_config(target_arrays, "target")
            
        # Validate data sources
        if input_path:
            self.validate_data_sources(input_path, target_path)
        
        # Validate transforms
        self.validate_transforms(spatial_transforms, raw_value_transforms, target_value_transforms)
        
        # Validate class configuration
        self.validate_class_config(classes, class_relationships)
        
        # Validate device configuration
        self.validate_device_config(device)
        
        # Validate performance configuration
        self.validate_performance_config(max_workers, axis_order)
        
        # Return normalized configuration
        normalized_config = {
            "input_path": input_path,
            "target_path": target_path,
            "classes": classes,
            "input_arrays": input_arrays,
            "target_arrays": target_arrays,
            "spatial_transforms": spatial_transforms,
            "raw_value_transforms": raw_value_transforms,
            "target_value_transforms": target_value_transforms,
            "class_relationships": class_relationships,
            "device": device,
            "max_workers": max_workers,
            "axis_order": axis_order,
        }
        
        logger.debug("Dataset configuration validation completed successfully")
        return normalized_config
        
    def check_data_integrity(
        self,
        input_path: str,
        target_path: Optional[str] = None,
        input_arrays: Optional[Mapping[str, Mapping[str, Sequence[int | float]]]] = None,
        target_arrays: Optional[Mapping[str, Mapping[str, Sequence[int | float]]]] = None,
    ) -> Dict[str, Any]:
        """Check data integrity and provide diagnostic information.
        
        Args:
            input_path: Path to input data
            target_path: Path to target data
            input_arrays: Input array specifications
            target_arrays: Target array specifications
            
        Returns:
            Dict[str, Any]: Data integrity report
        """
        if not self.strict_mode:
            logger.debug("Skipping data integrity check (not in strict mode)")
            return {"integrity_check": "skipped", "reason": "not in strict mode"}
            
        logger.debug("Starting data integrity check")
        
        integrity_report = {
            "input_path_exists": os.path.exists(input_path) if input_path else False,
            "target_path_exists": os.path.exists(target_path) if target_path else False,
            "input_arrays_count": len(input_arrays) if input_arrays else 0,
            "target_arrays_count": len(target_arrays) if target_arrays else 0,
            "checks_performed": [],
            "warnings": [],
            "errors": [],
        }
        
        # Add specific integrity checks here if needed
        integrity_report["checks_performed"].append("path_existence")
        
        logger.debug("Data integrity check completed")
        return integrity_report
        
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation operations performed.
        
        Returns:
            Dict[str, Any]: Validation summary including cache stats, mode info
        """
        return {
            "strict_mode": self.strict_mode,
            "cache_size": len(self.validation_cache),
            "validator_class": self.__class__.__name__,
        }
