"""
Centralized logging configuration for CellMap-Data.

This module provides consistent logging setup and standardized patterns
across the entire CellMap-Data library.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union, Dict, Any


class CellMapLogger:
    """Centralized logging configuration and utilities for CellMap-Data."""

    # Default configuration
    DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DEFAULT_LEVEL = logging.INFO

    # Library-specific logger name
    LIBRARY_NAME = "cellmap_data"

    # Standard loggers for different components
    COMPONENT_LOGGERS = {
        "dataset": f"{LIBRARY_NAME}.dataset",
        "dataset_writer": f"{LIBRARY_NAME}.dataset_writer",
        "datasplit": f"{LIBRARY_NAME}.datasplit",
        "image": f"{LIBRARY_NAME}.image",
        "image_writer": f"{LIBRARY_NAME}.image_writer",
        "multidataset": f"{LIBRARY_NAME}.multidataset",
        "dataloader": f"{LIBRARY_NAME}.dataloader",
        "transforms": f"{LIBRARY_NAME}.transforms",
        "utils": f"{LIBRARY_NAME}.utils",
        "validation": f"{LIBRARY_NAME}.validation",
    }

    _configured = False
    _loggers: Dict[str, logging.Logger] = {}

    @classmethod
    def configure_logging(
        cls,
        level: Union[int, str] = DEFAULT_LEVEL,
        format_string: str = DEFAULT_FORMAT,
        filename: Optional[Union[str, Path]] = None,
        console: bool = True,
    ) -> None:
        """
        Configure logging for the entire CellMap-Data library.

        Parameters
        ----------
        level : int or str, default=logging.INFO
            Logging level to use
        format_string : str, default=DEFAULT_FORMAT
            Format string for log messages
        filename : str or Path, optional
            If provided, log to this file in addition to console
        console : bool, default=True
            Whether to log to console
        """
        if cls._configured:
            return

        # Create formatter
        formatter = logging.Formatter(format_string)

        # Configure root library logger
        root_logger = logging.getLogger(cls.LIBRARY_NAME)
        root_logger.setLevel(level)

        # Clear any existing handlers
        root_logger.handlers.clear()

        # Add console handler if requested
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)

        # Add file handler if requested
        if filename:
            file_handler = logging.FileHandler(filename)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

        # Prevent propagation to avoid duplicate messages
        root_logger.propagate = False

        cls._configured = True

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get a configured logger for a specific component.

        Parameters
        ----------
        name : str
            Logger name or component name (e.g., 'dataset', 'image', etc.)

        Returns
        -------
        logging.Logger
            Configured logger instance
        """
        # Auto-configure if not already done
        if not cls._configured:
            cls.configure_logging()

        # Map component names to full logger names
        if name in cls.COMPONENT_LOGGERS:
            logger_name = cls.COMPONENT_LOGGERS[name]
        elif name.startswith(cls.LIBRARY_NAME):
            logger_name = name
        else:
            logger_name = f"{cls.LIBRARY_NAME}.{name}"

        # Return cached logger or create new one
        if logger_name not in cls._loggers:
            cls._loggers[logger_name] = logging.getLogger(logger_name)

        return cls._loggers[logger_name]

    @classmethod
    def set_level(cls, level: Union[int, str]) -> None:
        """
        Set logging level for all CellMap-Data loggers.

        Parameters
        ----------
        level : int or str
            New logging level
        """
        root_logger = logging.getLogger(cls.LIBRARY_NAME)
        root_logger.setLevel(level)

        # Update all existing component loggers
        for logger in cls._loggers.values():
            logger.setLevel(level)

    @classmethod
    def reset_configuration(cls) -> None:
        """Reset logging configuration to allow reconfiguration."""
        cls._configured = False
        cls._loggers.clear()

        # Clear handlers from root logger
        root_logger = logging.getLogger(cls.LIBRARY_NAME)
        # Properly close file handlers before clearing
        for handler in root_logger.handlers[:]:
            if hasattr(handler, "close"):
                handler.close()
        root_logger.handlers.clear()


# Convenience functions for common usage patterns
def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger for a component.

    This is the primary function that should be used throughout the library.

    Parameters
    ----------
    name : str
        Component name (e.g., 'dataset', 'image') or full module name

    Returns
    -------
    logging.Logger
        Configured logger instance

    Examples
    --------
    >>> from cellmap_data.utils.logging_config import get_logger
    >>> logger = get_logger('dataset')
    >>> logger.info("Dataset loading started")
    """
    return CellMapLogger.get_logger(name)


def configure_logging(
    level: Union[int, str] = logging.INFO,
    format_string: str = CellMapLogger.DEFAULT_FORMAT,
    filename: Optional[Union[str, Path]] = None,
    console: bool = True,
) -> None:
    """
    Configure logging for the CellMap-Data library.

    This should typically be called once at the start of a script or application
    that uses CellMap-Data.

    Parameters
    ----------
    level : int or str, default=logging.INFO
        Logging level
    format_string : str, optional
        Custom format string for log messages
    filename : str or Path, optional
        Log file path
    console : bool, default=True
        Whether to log to console

    Examples
    --------
    >>> from cellmap_data.utils.logging_config import configure_logging
    >>> configure_logging(level=logging.DEBUG, filename='cellmap.log')
    """
    CellMapLogger.configure_logging(level, format_string, filename, console)


def set_log_level(level: Union[int, str]) -> None:
    """
    Set logging level for all CellMap-Data loggers.

    Parameters
    ----------
    level : int or str
        New logging level (e.g., logging.DEBUG, 'INFO', etc.)

    Examples
    --------
    >>> from cellmap_data.utils.logging_config import set_log_level
    >>> set_log_level(logging.DEBUG)
    """
    CellMapLogger.set_level(level)


# Standard logging patterns for common use cases
class LoggingMixin:
    """
    Mixin class that provides standardized logging functionality.

    Classes that inherit from this mixin will automatically get a configured
    logger instance as self.logger.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Create logger based on class module and name
        module_name = cls.__module__.replace("cellmap_data.", "")
        cls._logger_name = f"{module_name}.{cls.__name__}"

    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        if not hasattr(self, "_logger"):
            self._logger = get_logger(self._logger_name)
        return self._logger


# Contextual logging utilities
def log_method_entry(logger: logging.Logger, method_name: str, **kwargs) -> None:
    """Log method entry with parameters."""
    if kwargs:
        param_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        logger.debug(f"Entering {method_name} with parameters: {param_str}")
    else:
        logger.debug(f"Entering {method_name}")


def log_method_exit(logger: logging.Logger, method_name: str, result=None) -> None:
    """Log method exit with optional result."""
    if result is not None:
        logger.debug(f"Exiting {method_name} with result: {result}")
    else:
        logger.debug(f"Exiting {method_name}")


def log_performance(logger: logging.Logger, operation: str, duration: float) -> None:
    """Log performance metrics."""
    logger.info(f"Performance: {operation} completed in {duration:.3f}s")
