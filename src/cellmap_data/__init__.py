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

from .multidataset import CellMapMultiDataset
from .dataloader import CellMapDataLoader
from .datasplit import CellMapDataSplit
from .utils.logging_config import configure_logging, get_logger, set_log_level
from .dataset import CellMapDataset
from .dataset_writer import CellMapDatasetWriter
from .image import CellMapImage
from .empty_image import EmptyImage
from .image_writer import ImageWriter
from .subdataset import CellMapSubset
from .mutable_sampler import MutableSubsetRandomSampler
from . import transforms
from . import utils
from .exceptions import (
    CellMapDataError,
    DataLoadingError,
    ValidationError,
    ConfigurationError,
    IndexError,
    CoordinateTransformError,
)
