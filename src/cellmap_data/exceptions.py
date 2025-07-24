"""
Exception classes for CellMap-Data operations.
This module provides a hierarchy of custom exceptions for better error handling.
"""


class CellMapDataError(Exception):
    """Base exception for CellMap-Data operations."""

    pass


class DataLoadingError(CellMapDataError):
    """Errors during data loading operations."""

    pass


class ValidationError(CellMapDataError):
    """Data validation errors."""

    pass


class ConfigurationError(CellMapDataError):
    """Configuration and parameter errors."""

    pass


class IndexError(CellMapDataError):
    """Indexing and coordinate transformation errors."""

    pass


class CoordinateTransformError(IndexError):
    """Errors in coordinate transformation operations."""

    pass
