"""
CellMap Data Loading Module.

Utility for loading CellMap data for machine learning training,
utilizing PyTorch, TensorStore, and PyDantic.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cellmap.data")
except PackageNotFoundError:
    __version__ = "0.0.0"

__author__ = "Jeff Rhoades"
__email__ = "rhoadesj@hhmi.org"

from .multidataset import CellMapMultiDataset
from .dataloader import CellMapDataLoader
from .datasplit import CellMapDataSplit
from .dataset import CellMapDataset
from .image import CellMapImage
from .subdataset import CellMapSubset
