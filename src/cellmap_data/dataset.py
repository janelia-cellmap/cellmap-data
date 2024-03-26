# %%
import csv
from typing import Callable, Dict, Iterable, Optional
from torch.utils.data import Dataset
import tensorstore as tswift
from fibsem_tools.io.core import read, read_xarray
from .image import CellMapImage


# %%
class CellMapDataset(Dataset):
    """This subclasses PyTorch Dataset to load CellMap data for training. It maintains the same API as the Dataset class. Importantly, it maintains information about and handles for the sources for raw and groundtruth data. This information includes the path to the data, the classes for segmentation, and the arrays to input to the network and use as targets for the network. The dataset constructs the sources for the raw and groundtruth data, and retrieves the data from the sources. The dataset also provides methods to get the number of pixels for each class in the ground truth data, normalized by the resolution. Additionally, random crops of the data can be generated for training, because the CellMapDataset maintains information about the extents of its source arrays. This object additionally combines images for different classes into a single output array, which is useful for training segmentation networks."""

    raw_path: str
    gt_path: str
    classes: Iterable[str]
    input_arrays: dict[str, dict[str, Iterable[int | float]]]
    target_arrays: dict[str, dict[str, Iterable[int | float]]]
    input_sources: dict[str, CellMapImage]
    target_sources: dict[str, dict[str, CellMapImage]]

    def __init__(
        self,
        raw_path: str,
        gt_path: str,
        classes: Iterable[str],
        input_arrays: dict[str, dict[str, Iterable[int | float]]],
        target_arrays: dict[str, dict[str, Iterable[int | float]]],
    ):
        """Initializes the CellMapDataset class.

        Args:
            raw_path (str): The path to the raw data.
            gt_path (str): The path to the ground truth data.
            classes (Iterable[str]): A list of classes for segmentation training. Class order will be preserved in the output arrays. Classes not contained in the dataset will be filled in with zeros.
            input_arrays (dict[str, dict[str, Iterable[int | float]]]): A dictionary containing the arrays of the dataset to input to the network. The dictionary should have the following structure:
                {
                    "array_name": {
                        "shape": typle[int],
                        "scale": Iterable[float],
                    },
                    ...
                }
            target_arrays (dict[str, dict[str, Iterable[int | float]]]): A dictionary containing the arrays of the dataset to use as targets for the network. The dictionary should have the following structure:
                {
                    "array_name": {
                        "shape": typle[int],
                        "scale": Iterable[float],
                    },
                    ...
                }
        """
        self.raw_path = raw_path
        self.gt_path = gt_path
        self.classes = classes
        self.input_arrays = input_arrays
        self.target_arrays = target_arrays
        self._bounding_box = None
        self._class_counts = None
        self.construct()

    def __len__(self): ...

    def __getitem__(self, idx):
        """Returns a random crop of the input and target data as PyTorch tensors."""
        ...

    def __iter__(self): ...

    def construct(self):
        """Constructs the input and target sources for the dataset."""
        self.input_sources = {}
        for array_name, array_info in self.input_arrays.items():
            self.input_sources[array_name] = CellMapImage(
                self.raw_path,
                "raw",
                array_info["scale"],
                array_info["shape"],
            )
        self.target_sources = {}
        for array_name, array_info in self.target_arrays.items():
            self.target_sources[array_name] = {}
            for label in self.classes:
                self.target_sources[array_name][label] = CellMapImage(
                    self.gt_path,
                    label,
                    array_info["scale"],
                    array_info["shape"],
                )

    @property
    def bounding_box(self):
        """Returns the bounding box of the dataset."""
        if self._bounding_box is None:
            bounding_box = {c: [0, 2**32] for c in "xyz"}
            for source in [self.input_sources.values(), self.target_sources.values()]:
                for c, (start, stop) in source.bounding_box.items():
                    bounding_box[c][0] = max(bounding_box[c][0], start)
                    bounding_box[c][1] = min(bounding_box[c][1], stop)
            self._bounding_box = bounding_box
        return self._bounding_box

    @property
    def class_counts(self) -> Dict[str, Dict[str, int]]:
        """Returns the number of pixels for each class in the ground truth data, normalized by the resolution."""
        if self._class_counts is None:
            class_counts = {}
            for array_name, sources in self.target_sources.items():
                class_counts[array_name] = {}
                for label, source in sources.items():
                    class_counts[array_name][label] = source.class_counts
            self._class_counts = class_counts
        return self._class_counts


# Example input arrays:
# {'0_input': {'shape': (90, 90, 90), 'scale': (32, 32, 32)},
#  '1_input': {'shape': (114, 114, 114), 'scale': (8, 8, 8)}}

# Example output arrays:
# {'0_output': {'shape': (28, 28, 28), 'scale': (32, 32, 32)},
#  '1_output': {'shape': (32, 32, 32), 'scale': (8, 8, 8)}}
