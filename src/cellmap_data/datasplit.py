import csv
import os
from typing import Any, Callable, Dict, Mapping, Optional, Sequence
import tensorstore
import torch
from .dataset import CellMapDataset
from .multidataset import CellMapMultiDataset
from .subdataset import CellMapSubset


class CellMapDataSplit:
    """
    This subclasses PyTorch Dataset to split data into training and validation sets. It maintains the same API as the Dataset class. It retrieves raw and groundtruth data from CellMapDataset objects.
    """

    input_arrays: dict[str, dict[str, Sequence[int | float]]]
    target_arrays: dict[str, dict[str, Sequence[int | float]]]
    classes: Sequence[str]
    to_target: Callable
    datasets: dict[str, Sequence[CellMapDataset]]
    train_datasets: Sequence[CellMapDataset]
    validation_datasets: Sequence[CellMapDataset]
    spatial_transforms: Optional[dict[str, Any]] = None
    train_raw_value_transforms: Optional[Callable] = None
    target_value_transforms: Optional[
        Callable | Sequence[Callable] | dict[str, Callable]
    ] = None
    force_has_data: bool = False
    context: Optional[tensorstore.Context] = None  # type: ignore

    def __init__(
        self,
        input_arrays: dict[str, dict[str, Sequence[int | float]]],
        target_arrays: dict[str, dict[str, Sequence[int | float]]],
        classes: Sequence[str],
        empty_value: int = 0,
        datasets: Optional[Dict[str, Sequence[CellMapDataset]]] = None,
        dataset_dict: Optional[Mapping[str, Sequence[Dict[str, str]]]] = None,
        csv_path: Optional[str] = None,
        spatial_transforms: Optional[dict[str, Any]] = None,
        train_raw_value_transforms: Optional[Callable] = None,
        val_raw_value_transforms: Optional[Callable] = None,
        target_value_transforms: Optional[
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
            classes (Sequence[str]): A list of classes for segmentation training. Class order will be preserved in the output arrays.
            empty_value (int, optional): The value to use for classes without ground truth. Defaults to 0.
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
            spatial_transforms (Optional[Sequence[dict[str, Any]]], optional): A sequence of dictionaries containing the spatial transformations to apply to the data. The dictionary should have the following structure:
                {transform_name: {transform_args}}
                Defaults to None.
            raw_value_transforms (Optional[Callable], optional): A function to apply to the raw data. Defaults to None. Example is to normalize the raw data.
            target_value_transforms (Optional[Callable | Sequence[Callable] | dict[str, Callable]], optional): A function to convert the ground truth data to target arrays. Defaults to None. Example is to convert the ground truth data to a signed distance transform. May be a single function, a list of functions, or a dictionary of functions for each class. In the case of a list of functions, it is assumed that the functions correspond to each class in the classes list in order.
            force_has_data (bool, optional): Whether to force the dataset to have data. Defaults to False.
            context (Optional[tensorstore.Context], optional): The context for the image data. Defaults to None.
        """
        self.input_arrays = input_arrays
        self.target_arrays = target_arrays
        self.classes = classes
        self.empty_value = empty_value
        self.force_has_data = force_has_data
        if datasets is not None:
            self.datasets = datasets
            self.train_datasets = datasets["train"]
            if "validate" in datasets:
                self.validation_datasets = datasets["validate"]
            else:
                self.validation_datasets = []
            self.dataset_dict = None
        elif dataset_dict is not None:
            self.dataset_dict = dataset_dict
        elif csv_path is not None:
            self.dataset_dict = self.from_csv(csv_path)
        self.spatial_transforms = spatial_transforms
        self.train_raw_value_transforms = train_raw_value_transforms
        self.val_raw_value_transforms = val_raw_value_transforms
        self.target_value_transforms = target_value_transforms
        self.context = context
        if self.dataset_dict is not None:
            self.construct(self.dataset_dict)
        self.verify_datasets()
        assert len(self.train_datasets) > 0, "No valid training datasets found."

    def __repr__(self):
        return f"CellMapDataSplit(\n\tInput arrays: {self.input_arrays}\n\tTarget arrays:{self.target_arrays}\n\tClasses: {self.classes}\n\tDataset dict: {self.dataset_dict}\n\tSpatial transforms: {self.spatial_transforms}\n\tRaw value transforms: {self.train_raw_value_transforms}\n\tGT value transforms: {self.target_value_transforms}\n\tForce has data: {self.force_has_data}\n\tContext: {self.context})"

    @property
    def train_datasets_combined(self):
        if not hasattr(self, "_train_datasets_combined"):
            self._train_datasets_combined = CellMapMultiDataset(
                self.classes,
                self.input_arrays,
                self.target_arrays,
                [
                    ds
                    for ds in self.train_datasets
                    if self.force_has_data or ds.has_data
                ],
            )
        return self._train_datasets_combined

    @property
    def validation_datasets_combined(self):
        assert len(self.validation_datasets) > 0, "Validation datasets not loaded."
        if not hasattr(self, "_validation_datasets_combined"):
            self._validation_datasets_combined = CellMapMultiDataset(
                self.classes,
                self.input_arrays,
                self.target_arrays,
                [
                    ds
                    for ds in self.validation_datasets
                    if self.force_has_data or ds.has_data
                ],
            )
        return self._validation_datasets_combined

    @property
    def validation_blocks(self):
        if not hasattr(self, "_validation_blocks"):
            self._validation_blocks = CellMapSubset(
                self.validation_datasets_combined,
                self.validation_datasets_combined.validation_indices,
            )
        return self._validation_blocks

    @property
    def class_counts(self):
        if not hasattr(self, "_class_counts"):
            self._class_counts = {
                "train": self.train_datasets_combined.class_counts,
                "validate": self.validation_datasets_combined.class_counts,
            }
        return self._class_counts

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
                        "gt": os.path.join(row[3], row[4]) if len(row) > 3 else "",
                    }
                )

        return dataset_dict

    def construct(self, dataset_dict):
        self.train_datasets = []
        self.validation_datasets = []
        self.datasets = {}
        for data_paths in dataset_dict["train"]:
            try:
                self.train_datasets.append(
                    CellMapDataset(
                        data_paths["raw"],
                        data_paths["gt"],
                        self.classes,
                        self.input_arrays,
                        self.target_arrays,
                        self.spatial_transforms,
                        self.train_raw_value_transforms,
                        self.target_value_transforms,
                        is_train=True,
                        context=self.context,
                        force_has_data=self.force_has_data,
                        empty_value=self.empty_value,
                    )
                )
            except ValueError as e:
                print(f"Error loading dataset: {e}")

        self.datasets["train"] = self.train_datasets

        # TODO: probably want larger arrays for validation

        if "validate" in dataset_dict:
            for data_paths in dataset_dict["validate"]:
                try:
                    self.validation_datasets.append(
                        CellMapDataset(
                            data_paths["raw"],
                            data_paths["gt"],
                            self.classes,
                            self.input_arrays,
                            self.target_arrays,
                            raw_value_transforms=self.val_raw_value_transforms,
                            target_value_transforms=self.target_value_transforms,
                            is_train=False,
                            context=self.context,
                            force_has_data=self.force_has_data,
                            empty_value=self.empty_value,
                        )
                    )
                except ValueError as e:
                    print(f"Error loading dataset: {e}")

            self.datasets["validate"] = self.validation_datasets

    def verify_datasets(self):
        if self.force_has_data:
            return
        verified_datasets = []
        for ds in self.train_datasets:
            if ds.verify():
                verified_datasets.append(ds)
        self.train_datasets = verified_datasets

        verified_datasets = []
        for ds in self.validation_datasets:
            if ds.verify():
                verified_datasets.append(ds)
        self.validation_datasets = verified_datasets

    def set_raw_value_transforms(
        self, train_transforms: Callable, val_transforms: Callable
    ):
        """Sets the raw value transforms for each dataset in the training multi-dataset."""
        for dataset in self.train_datasets:
            dataset.set_raw_value_transforms(train_transforms)
        if hasattr(self, "_train_datasets_combined"):
            self._train_datasets_combined.set_raw_value_transforms(train_transforms)
        for dataset in self.validation_datasets:
            dataset.set_raw_value_transforms(val_transforms)
        if hasattr(self, "_validation_datasets_combined"):
            self._validation_datasets_combined.set_raw_value_transforms(val_transforms)

    def set_target_value_transforms(self, transforms: Callable):
        """Sets the target value transforms for each dataset in the multi-datasets."""
        for dataset in self.train_datasets:
            dataset.set_target_value_transforms(transforms)
        if hasattr(self, "_train_datasets_combined"):
            self._train_datasets_combined.set_target_value_transforms(transforms)

        for dataset in self.validation_datasets:
            dataset.set_target_value_transforms(transforms)
        if hasattr(self, "_validation_datasets_combined"):
            self._validation_datasets_combined.set_target_value_transforms(transforms)
        if hasattr(self, "_validation_blocks"):
            self._validation_blocks.set_target_value_transforms(transforms)


# Example input arrays:
# {'0_input': {'shape': (90, 90, 90), 'scale': (32, 32, 32)},
#  '1_input': {'shape': (114, 114, 114), 'scale': (8, 8, 8)}}

# Example output arrays:
# {'0_output': {'shape': (28, 28, 28), 'scale': (32, 32, 32)},
#  '1_output': {'shape': (32, 32, 32), 'scale': (8, 8, 8)}}
