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

from .base_dataset import CellMapBaseDataset
from .base_image import CellMapImageBase
from .dataloader import CellMapDataLoader
from .dataset import CellMapDataset
from .dataset_writer import CellMapDatasetWriter
from .datasplit import CellMapDataSplit
from .empty_image import EmptyImage
from .image import CellMapImage
from .image_writer import ImageWriter
from .multidataset import CellMapMultiDataset
from .mutable_sampler import MutableSubsetRandomSampler
from .subdataset import CellMapSubset

__all__ = [
    "CellMapBaseDataset",
    "CellMapImageBase",
    "CellMapDataLoader",
    "CellMapDataset",
    "CellMapDatasetWriter",
    "CellMapDataSplit",
    "CellMapImage",
    "ImageWriter",
    "CellMapMultiDataset",
    "CellMapSubset",
    "EmptyImage",
    "MutableSubsetRandomSampler",
]
