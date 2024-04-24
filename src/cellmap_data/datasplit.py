import csv
import os
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence
import tensorstore
from .dataset import CellMapDataset
from .multidataset import CellMapMultiDataset


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
    spatial_transforms: Optional[dict[str, any]] = None
    raw_value_transforms: Optional[Callable] = None
    gt_value_transforms: Optional[
        Callable | Sequence[Callable] | dict[str, Callable]
    ] = None
    force_has_data: bool = False
    context: Optional[tensorstore.Context] = None  # type: ignore

    def __init__(
        self,
        input_arrays: dict[str, dict[str, Sequence[int | float]]],
        target_arrays: dict[str, dict[str, Sequence[int | float]]],
        classes: Sequence[str],
        datasets: Optional[Dict[str, Iterable[CellMapDataset]]] = None,
        dataset_dict: Optional[Mapping[str, Sequence[Dict[str, str]]]] = None,
        csv_path: Optional[str] = None,
        spatial_transforms: Optional[dict[str, any]] = None,
        raw_value_transforms: Optional[Callable] = None,
        gt_value_transforms: Optional[
            Callable | Sequence[Callable] | dict[str, Callable]
        ] = None,
        force_has_data: bool = False,
        context: Optional[tensorstore.Context] = None,  # type: ignore
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
            dataset_dict (Optional[Dict[str, Sequence[Dict[str, str]]]], optional): A dictionary containing the dataset data. The dictionary should have the following structure:
                {
                    "train" | "validate": [{
                        "raw": str (path to raw data),
                        "gt": str (path to ground truth data),
                    }],
                    ...
                }. Defaults to None.
            csv_path (Optional[str], optional): A path to a csv file containing the dataset data. Defaults to None. Each row in the csv file should have the following structure:
                train | validate, raw path, gt path
            spatial_transforms (Optional[Sequence[dict[str, any]]], optional): A sequence of dictionaries containing the spatial transformations to apply to the data. The dictionary should have the following structure:
                {transform_name: {transform_args}}
                Defaults to None.
            raw_value_transforms (Optional[Callable], optional): A function to apply to the raw data. Defaults to None. Example is to normalize the raw data.
            gt_value_transforms (Optional[Callable | Sequence[Callable] | dict[str, Callable]], optional): A function to convert the ground truth data to target arrays. Defaults to None. Example is to convert the ground truth data to a signed distance transform. May be a single function, a list of functions, or a dictionary of functions for each class. In the case of a list of functions, it is assumed that the functions correspond to each class in the classes list in order.
            force_has_data (bool, optional): Whether to force the dataset to have data. Defaults to False.
            context (Optional[tensorstore.Context], optional): The context for the image data. Defaults to None.
        """
        self.input_arrays = input_arrays
        self.target_arrays = target_arrays
        self.classes = classes
        self.force_has_data = force_has_data
        if datasets is not None:
            self.train_datasets = datasets["train"]
            self.validate_datasets = datasets["validate"]
            self.dataset_dict = {}
        elif dataset_dict is not None:
            self.dataset_dict = dataset_dict
            self.construct(dataset_dict)
        elif csv_path is not None:
            self.from_csv(csv_path)
        self.spatial_transforms = spatial_transforms
        self.raw_value_transforms = raw_value_transforms
        self.gt_value_transforms = gt_value_transforms
        self.context = context
        self.construct(self.dataset_dict)

    def from_csv(self, csv_path):
        # Load file data from csv file
        dataset_dict = {}
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] not in dataset_dict:
                    dataset_dict[row[0]] = []
                dataset_dict[row[0]].append(
                    {
                        "raw": os.path.join(row[1], row[2]),
                        "gt": os.path.join(row[3], row[4]),
                    }
                )

        self.dataset_dict = dataset_dict
        self.construct(dataset_dict)

    def construct(self, dataset_dict):
        self._class_counts = None
        self.train_datasets = []
        self.validate_datasets = []
        for data_paths in dataset_dict["train"]:
            self.train_datasets.append(
                CellMapDataset(
                    data_paths["raw"],
                    data_paths["gt"],
                    self.classes,
                    self.input_arrays,
                    self.target_arrays,
                    self.spatial_transforms,
                    self.raw_value_transforms,
                    self.gt_value_transforms,
                    is_train=True,
                    context=self.context,
                    force_has_data=self.force_has_data,
                )
            )

        self.train_datasets_combined = CellMapMultiDataset(
            self.classes,
            self.input_arrays,
            self.target_arrays,
            [ds for ds in self.train_datasets if self.force_has_data or ds.has_data],
        )

        # TODO: probably want larger arrays for validation

        if "validate" in dataset_dict:
            for data_paths in dataset_dict["validate"]:
                self.validate_datasets.append(
                    CellMapDataset(
                        data_paths["raw"],
                        data_paths["gt"],
                        self.classes,
                        self.input_arrays,
                        self.target_arrays,
                        gt_value_transforms=self.gt_value_transforms,
                        is_train=False,
                        context=self.context,
                        force_has_data=self.force_has_data,
                    )
                )
            self.validate_datasets_combined = CellMapMultiDataset(
                self.classes,
                self.input_arrays,
                self.target_arrays,
                [
                    ds
                    for ds in self.validate_datasets
                    if self.force_has_data or ds.has_data
                ],
            )

    @property
    def class_counts(self):
        if self._class_counts is None:
            self._class_counts = {
                "train": self.train_datasets_combined.class_counts,
                "validate": self.validate_datasets_combined.class_counts,
            }
        return self._class_counts


# Example input arrays:
# {'0_input': {'shape': (90, 90, 90), 'scale': (32, 32, 32)},
#  '1_input': {'shape': (114, 114, 114), 'scale': (8, 8, 8)}}

# Example output arrays:
# {'0_output': {'shape': (28, 28, 28), 'scale': (32, 32, 32)},
#  '1_output': {'shape': (32, 32, 32), 'scale': (8, 8, 8)}}
