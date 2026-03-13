"""Utility for loading CellMap data for machine learning training."""

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    from importlib_metadata import PackageNotFoundError, version

try:
    __version__ = version("cellmap-data")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Jeff Rhoades"
__email__ = "rhoadesj@hhmi.org"

from .dataloader import CellMapDataLoader
from .dataset import CellMapDataset
from .dataset_writer import CellMapDatasetWriter
from .datasplit import CellMapDataSplit
from .empty_image import EmptyImage
from .image import CellMapImage
from .image_writer import ImageWriter
from .multidataset import CellMapMultiDataset
from .sampler import ClassBalancedSampler

__all__ = [
    "CellMapDataLoader",
    "CellMapDataset",
    "CellMapDatasetWriter",
    "CellMapDataSplit",
    "CellMapImage",
    "ImageWriter",
    "CellMapMultiDataset",
    "EmptyImage",
    "ClassBalancedSampler",
]
