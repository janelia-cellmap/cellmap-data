import csv
from torchvision.transforms.v2 import RandomApply
from typing import Callable, Dict, Iterable, Optional, Sequence

from .multidataset import CellMapMultiDataset
from .dataset import CellMapDataset


class CellMapDataSplit:
    """
    This subclasses PyTorch Dataset to split data into training and validation sets. It maintains the same API as the Dataset class. It retrieves raw and groundtruth data from CellMapDataset objects.
    """

    input_arrays: dict[str, dict[str, Sequence[int | float]]]
    target_arrays: dict[str, dict[str, Sequence[int | float]]]
    classes: Sequence[str]
    to_target: Callable
    datasets: dict[str, Iterable[CellMapDataset]]
    train_datasets: Iterable[CellMapDataset]
    validate_datasets: Iterable[CellMapDataset]
    train_datasets_combined: CellMapMultiDataset
    validate_datasets_combined: CellMapMultiDataset
    transforms: RandomApply | None

    def __init__(
        self,
        input_arrays: dict[str, dict[str, Sequence[int | float]]],
        target_arrays: dict[str, dict[str, Sequence[int | float]]],
        classes: Sequence[str],
        to_target: Callable,
        datasets: Optional[Dict[str, Iterable[CellMapDataset]]] = None,
        dataset_dict: Optional[Dict[str, Dict[str, str]]] = None,
        csv_path: Optional[str] = None,
        transforms: Optional[Sequence[Callable]] = None,
    ):
        """Initializes the CellMapDatasets class.

        Args:
            input_arrays (dict[str, dict[str, Sequence[int | float]]]): A dictionary containing the arrays of the dataset to input to the network. The dictionary should have the following structure:
                {
                    "array_name": {
                        "shape": typle[int],
                        "scale": Sequence[float],
                    },
                    ...
                }
            target_arrays (dict[str, dict[str, Sequence[int | float]]]): A dictionary containing the arrays of the dataset to use as targets for the network. The dictionary should have the following structure:
                {
                    "array_name": {
                        "shape": typle[int],
                        "scale": Sequence[float],
                    },
                    ...
                }
            classes (Sequence[str]): A list of classes for segmentation training. Class order will be preserved in the output arrays. Classes not contained in the dataset will be filled in with zeros.
            to_target (Callable): A function to convert the ground truth data to target arrays. The function should have the following structure:
                def to_target(gt: torch.Tensor, classes: Sequence[str]) -> dict[str, torch.Tensor]:
            datasets (Optional[Dict[str, CellMapDataset]], optional): A dictionary containing the dataset objects. The dictionary should have the following structure:
                {
                    "train": Iterable[CellMapDataset],
                    "validate": Iterable[CellMapDataset],
                }. Defaults to None.
            dataset_dict (Optional[Dict[str, Dict[str, str]]], optional): A dictionary containing the dataset data. The dictionary should have the following structure:
                {
                    "train" | "validate": {
                        "raw": str (path to raw data),
                        "gt": str (path to ground truth data),
                    },
                    ...
                }. Defaults to None.
            csv_path (Optional[str], optional): A path to a csv file containing the dataset data. Defaults to None. Each row in the csv file should have the following structure:
                train | validate, raw path, gt path
            transforms (Optional[Iterable[dict[str, any]]], optional): A list of transforms to apply to the data. Each augmentation should be a dictionary containing the following structure:
        """
        self.input_arrays = input_arrays
        self.target_arrays = target_arrays
        self.classes = classes
        if datasets is not None:
            self.train_datasets = datasets["train"]
            self.validate_datasets = datasets["validate"]
            self.dataset_dict = {}
        elif dataset_dict is not None:
            self.dataset_dict = dataset_dict
            self.construct(dataset_dict)
        elif csv_path is not None:
            self.from_csv(csv_path)
        self.to_target = to_target
        self.transforms = RandomApply(transforms) if transforms is not None else None

    def from_csv(self, csv_path):
        # Load file data from csv file
        dataset_dict = {}
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] not in dataset_dict:
                    dataset_dict[row[0]] = {"raw": [], "gt": []}
                dataset_dict[row[0]]["raw"].append(row[1])
                dataset_dict[row[0]]["gt"].append(row[2])

        self.dataset_dict = dataset_dict
        self.construct(dataset_dict)

    def construct(self, dataset_dict):
        self.train_datasets = []
        self.validate_datasets = []
        for raw, gt in zip(dataset_dict["train"]["raw"], dataset_dict["train"]["gt"]):
            self.train_datasets.append(
                CellMapDataset(
                    raw,
                    gt,
                    self.classes,
                    self.input_arrays,
                    self.target_arrays,
                    self.to_target,
                    self.transforms,
                )
            )
        for raw, gt in zip(
            dataset_dict["validate"]["raw"], dataset_dict["validate"]["gt"]
        ):
            self.validate_datasets.append(
                CellMapDataset(
                    raw,
                    gt,
                    self.classes,
                    self.input_arrays,
                    self.target_arrays,
                    self.to_target,
                )
            )
        self.train_datasets_combined = CellMapMultiDataset(
            self.classes,
            self.input_arrays,
            self.target_arrays,
            [ds for ds in self.train_datasets if ds.has_data],
        )
        self.validate_datasets_combined = CellMapMultiDataset(
            self.classes,
            self.input_arrays,
            self.target_arrays,
            [ds for ds in self.validate_datasets if ds.has_data],
        )


# Example input arrays:
# {'0_input': {'shape': (90, 90, 90), 'scale': (32, 32, 32)},
#  '1_input': {'shape': (114, 114, 114), 'scale': (8, 8, 8)}}

# Example output arrays:
# {'0_output': {'shape': (28, 28, 28), 'scale': (32, 32, 32)},
#  '1_output': {'shape': (32, 32, 32), 'scale': (8, 8, 8)}}
