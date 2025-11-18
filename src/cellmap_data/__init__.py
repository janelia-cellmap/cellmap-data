"""
CellMap Data Loading Module.

Utility for loading CellMap data for machine learning training,
utilizing PyTorch, TensorStore, XArray, and PyDantic.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cellmap_data")
except PackageNotFoundError:
    __version__ = "0.1.1"

__author__ = "Jeff Rhoades"
__email__ = "rhoadesj@hhmi.org"

from . import transforms, utils
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
    "CellMapMultiDataset",
    "CellMapDataLoader",
    "CellMapDataSplit",
    "CellMapDataset",
    "CellMapDatasetWriter",
    "CellMapImage",
    "EmptyImage",
    "ImageWriter",
    "CellMapSubset",
    "MutableSubsetRandomSampler",
    "transforms",
    "utils",
]
