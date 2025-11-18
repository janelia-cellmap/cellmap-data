"""
CellMap Dataset Module.

Modular dataset architecture with separated concerns for improved maintainability,
testability, and performance. This module provides the core dataset functionality
with extracted specialized components.
"""

from .core import CellMapDataset
from .config import DatasetConfig
from .device_manager import DeviceManager
from .coordinate_manager import CoordinateTransformer
from .data_loader import DataLoader
from .array_manager import ArrayManager
from .validator import DatasetValidator

__all__ = [
    "CellMapDataset",
    "DatasetConfig",
    "DeviceManager",
    "CoordinateTransformer",
    "DataLoader",
    "ArrayManager",
    "DatasetValidator",
]
