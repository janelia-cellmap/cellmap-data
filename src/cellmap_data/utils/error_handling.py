"""
Standardized error message templates and utilities for CellMap-Data.

This module provides consistent error message formatting and common error patterns
to improve debugging and user experience across the CellMap-Data library.
"""

import warnings
from typing import Any, Optional


class ErrorMessages:
    """Standardized error message templates for consistent error reporting."""

    # Parameter validation messages
    PARAMETER_REQUIRED = "Parameter '{parameter}' is required but not provided"
    PARAMETER_CONFLICT = "Cannot specify both '{param1}' and '{param2}' parameters"
    PARAMETER_DEPRECATED = "Parameter '{parameter}' is deprecated and will be removed in a future version. Please use '{replacement}' instead."
    PARAMETER_INVALID_TYPE = (
        "Parameter '{parameter}' must be of type {expected_type}, got {actual_type}"
    )
    PARAMETER_INVALID_VALUE = (
        "Parameter '{parameter}' has invalid value '{value}'. Expected: {expected}"
    )

    # File and path validation messages
    FILE_NOT_FOUND = "File not found: {path}"
    FILE_EXISTS = "File already exists at {path}. Set overwrite=True to overwrite"
    PATH_INVALID = "Invalid path: {path}"
    PATH_NOT_ACCESSIBLE = "Path is not accessible: {path}"

    # Data validation messages
    DATA_SHAPE_MISMATCH = (
        "Data shape mismatch: expected {expected_shape}, got {actual_shape}"
    )
    DATA_TYPE_UNSUPPORTED = "Unsupported data type: {data_type}"
    DATA_EMPTY = "Dataset is empty or contains no valid data"
    DATA_CORRUPTED = "Data appears to be corrupted: {details}"

    # Array and coordinate messages
    COORDINATE_OUT_OF_BOUNDS = "Coordinate {coordinate} is out of bounds for axis {axis}. Valid range: [{min_val}, {max_val}]"
    ARRAY_SIZE_MISMATCH = (
        "Array size mismatch for '{array_name}': expected {expected}, got {actual}"
    )
    INDEX_INVALID = "Invalid index {index} for array of length {length}"

    # Configuration and setup messages
    CONFIG_INVALID = "Invalid configuration: {details}"
    SETUP_INCOMPLETE = "Setup is incomplete: {missing_component}"
    DEVICE_UNSUPPORTED = "Device '{device}' is not supported"

    # Warning messages
    FALLBACK_DRIVER = "Falling back to {fallback_driver} driver due to error: {error}"
    PERFORMANCE_WARNING = "Performance warning: {details}"
    COMPATIBILITY_WARNING = "Compatibility warning: {details}"


class StandardWarnings:
    """Utilities for issuing standardized warnings."""

    @staticmethod
    def parameter_deprecated(
        parameter: str, replacement: str, stacklevel: int = 2
    ) -> None:
        """Issue a deprecation warning for a parameter."""
        warnings.warn(
            ErrorMessages.PARAMETER_DEPRECATED.format(
                parameter=parameter, replacement=replacement
            ),
            DeprecationWarning,
            stacklevel=stacklevel,
        )

    @staticmethod
    def fallback_driver(fallback_driver: str, error: str, stacklevel: int = 2) -> None:
        """Issue a warning about falling back to a different driver."""
        warnings.warn(
            ErrorMessages.FALLBACK_DRIVER.format(
                fallback_driver=fallback_driver, error=error
            ),
            UserWarning,
            stacklevel=stacklevel,
        )

    @staticmethod
    def performance_warning(details: str, stacklevel: int = 2) -> None:
        """Issue a performance-related warning."""
        warnings.warn(
            ErrorMessages.PERFORMANCE_WARNING.format(details=details),
            UserWarning,
            stacklevel=stacklevel,
        )


class ValidationError(ValueError):
    """Enhanced ValueError with standardized error message formatting."""

    def __init__(self, template: str, **kwargs):
        """Initialize with a template and format arguments."""
        message = template.format(**kwargs)
        super().__init__(message)


def validate_parameter_conflict(
    param1: str, value1: Any, param2: str, value2: Any
) -> None:
    """Validate that conflicting parameters are not both specified."""
    if value1 is not None and value2 is not None:
        raise ValidationError(
            ErrorMessages.PARAMETER_CONFLICT, param1=param1, param2=param2
        )


def validate_parameter_required(parameter: str, value: Any) -> None:
    """Validate that a required parameter is provided."""
    if value is None:
        raise ValidationError(ErrorMessages.PARAMETER_REQUIRED, parameter=parameter)


def validate_parameter_type(parameter: str, value: Any, expected_type: type) -> None:
    """Validate that a parameter has the expected type."""
    if value is not None and not isinstance(value, expected_type):
        raise ValidationError(
            ErrorMessages.PARAMETER_INVALID_TYPE,
            parameter=parameter,
            expected_type=expected_type.__name__,
            actual_type=type(value).__name__,
        )


def create_bounds_error(
    coordinate: float, axis: str, min_val: float, max_val: float
) -> str:
    """Create a standardized coordinate bounds error message."""
    return ErrorMessages.COORDINATE_OUT_OF_BOUNDS.format(
        coordinate=coordinate, axis=axis, min_val=min_val, max_val=max_val
    )


def create_shape_mismatch_error(expected_shape: tuple, actual_shape: tuple) -> str:
    """Create a standardized shape mismatch error message."""
    return ErrorMessages.DATA_SHAPE_MISMATCH.format(
        expected_shape=expected_shape, actual_shape=actual_shape
    )
