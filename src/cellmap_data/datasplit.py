import csv
import os
from typing import Any, Callable, Mapping, Optional, Sequence
import tensorstore
import torch
import torchvision.transforms.v2 as T
from tqdm import tqdm
from .transforms import NaNtoNum, Normalize, Binarize
from .dataset import CellMapDataset
from .multidataset import CellMapMultiDataset
from .subdataset import CellMapSubset
import logging

logger = logging.getLogger(__name__)


class CellMapDataSplit:
    """
    A class to split the data into training and validation datasets.

    Attributes:
        input_arrays (dict[str, dict[str, Sequence[int | float]]]): A dictionary containing the arrays of the dataset to input to the network. The dictionary should have the following structure::
            {
                "array_name": {
                    "shape": tuple[int],
                    "scale": Sequence[float],
                },
                ...
            }

        target_arrays (dict[str, dict[str, Sequence[int | float]]]): A dictionary containing the arrays of the dataset to use as targets for the network. The dictionary should have the same structure as input_arrays.
        classes (Sequence[str]): A list of classes for segmentation training. Class order will be preserved in the output arrays.
        empty_value (int | float): The value to use for empty data. Defaults to torch.nan.
        pad (bool | str): Whether to pad the data. If a string, it should be either "train" or "validate". Defaults to False.
        datasets (Optional[Mapping[str, Sequence[CellMapDataset]]]): A dictionary containing the dataset objects. The dictionary should have the following structure:
            {
                "train": Iterable[CellMapDataset],
                "validate": Iterable[CellMapDataset],
            }. Defaults to None.
        dataset_dict (Optional[Mapping[str, Sequence[Mapping[str, str]]]): A dictionary containing the dataset data. Defaults to None. The dictionary should have the following structure::

            {
                "train" | "validate": [{
                    "raw": str (path to raw data),
                    "gt": str (path to ground truth data),
                }],
                ...
            }

        csv_path (Optional[str]): A path to a csv file containing the dataset data. Defaults to None. Each row in the csv file should have the following structure:
            train | validate, raw path, gt path
        spatial_transforms (Optional[Sequence[dict[str, Any]]]): A sequence of dictionaries containing the spatial transformations to apply to the data. Defaults to None. The dictionary should have the following structure::

            {transform_name: {transform_args}}

        train_raw_value_transforms (Optional[Callable]): A function to apply to the raw data in training datasets. Example is to add gaussian noise to the raw data. Defailts to Normalize, ToDtype, and NaNtoNum.
        val_raw_value_transforms (Optional[Callable]): A function to apply to the raw data in validation datasets. Example is to normalize the raw data. Defaults to Normalize, ToDtype, and NaNtoNum.
        target_value_transforms (Optional[Callable | Sequence[Callable] | Mapping[str, Callable]]): A function to convert the ground truth data to target arrays. Example is to convert the ground truth data to a signed distance transform. May be a single function, a list of functions, or a dictionary of functions for each class. In the case of a list of functions, it is assumed that the functions correspond to each class in the classes list in order. Defaults to ToDtype and Binarize.
        class_relation_dict (Optional[Mapping[str, Sequence[str]]]): A dictionary containing the class relations. The dictionary should have the following structure::

            {
                "class_name": [mutually_exclusive_class_name, ...],
                ...
            }

        force_has_data (bool): Whether to force the datasets to have data even if no ground truth data is found. Defaults to False. Useful for training with only raw data.
        context (Optional[tensorstore.Context]): The TensorStore context for the image data. Defaults to None.
        device (Optional[str | torch.device]): Device to use for the dataloaders. Defaults to None.

    Note:
        The csv_path, dataset_dict, and datasets arguments are mutually exclusive, but one must be supplied.

    Methods:
        __repr__(): Returns the string representation of the class.
        from_csv(csv_path: str): Loads the dataset data from a csv file.
        construct(dataset_dict: Mapping[str, Sequence[Mapping[str, str]]]): Constructs the datasets from the dataset dictionary.
        verify_datasets(): Verifies that the datasets have data, and removes ones that don't from 'self.train_datasets' and 'self.validation_datasets'.
        set_raw_value_transforms(train_transforms: Optional[Callable] = None, val_transforms: Optional[Callable] = None): Sets the raw value transforms for each dataset in the training/validation multi-datasets.
        set_target_value_transforms(transforms: Callable): Sets the target value transforms for each dataset in both training and validation multi-datasets.
        set_spatial_transforms(spatial_transforms: dict[str, Any] | None): Sets the spatial transforms for each dataset in the training multi-dataset.
        set_arrays(arrays: Mapping[str, Mapping[str, Sequence[int | float]]], type: str = "target", usage: str = "validate"): Sets the input or target arrays for the training or validation datasets.

    Properties:
        train_datasets_combined: A multi-dataset from the combination of all training datasets.
        validation_datasets_combined: A multi-dataset from the combination of all validation datasets.
        validation_blocks: A subset of the validation datasets, tiling the validation datasets with non-overlapping blocks.
        class_counts: A dictionary containing the class counts for the training and validation datasets.

    """

    def __init__(
        self,
        input_arrays: Mapping[str, Mapping[str, Sequence[int | float]]],
        target_arrays: Optional[
            Mapping[str, Mapping[str, Sequence[int | float]]]
        ] = None,
        classes: Sequence[str] | None = None,
        empty_value: int | float = torch.nan,
        pad: bool | str = False,
        datasets: Optional[Mapping[str, Sequence[CellMapDataset]]] = None,
        dataset_dict: Optional[Mapping[str, Sequence[Mapping[str, str]]]] = None,
        csv_path: Optional[str] = None,
        spatial_transforms: Optional[Mapping[str, Any]] = None,
        train_raw_value_transforms: Optional[T.Transform] = T.Compose(
            [
                Normalize(),
                T.ToDtype(torch.float, scale=True),
                NaNtoNum({"nan": 0, "posinf": None, "neginf": None}),
            ],
        ),
        val_raw_value_transforms: Optional[T.Transform] = T.Compose(
            [
                Normalize(),
                T.ToDtype(torch.float, scale=True),
                NaNtoNum({"nan": 0, "posinf": None, "neginf": None}),
            ],
        ),
        target_value_transforms: Optional[T.Transform] = T.Compose(
            [T.ToDtype(torch.float), Binarize()]
        ),
        class_relation_dict: Optional[Mapping[str, Sequence[str]]] = None,
        force_has_data: bool = False,
        context: Optional[tensorstore.Context] = None,  # type: ignore
        device: Optional[str | torch.device] = None,
    ) -> None:
        """Initializes the CellMapDatasets class.

        Args:
            input_arrays (dict[str, dict[str, Sequence[int | float]]]): A dictionary containing the arrays of the dataset to input to the network. The dictionary should have the following structure::

                {
                    "array_name": {
                        "shape": tuple[int],
                        "scale": Sequence[float],
                    },
                    ...
                }

            target_arrays (dict[str, dict[str, Sequence[int | float]]]): A dictionary containing the arrays of the dataset to use as targets for the network. The dictionary should have the same structure as input_arrays.
            classes (Sequence[str]): A list of classes for segmentation training. Class order will be preserved in the output arrays.
            empty_value (int | float): The value to use for empty data. Defaults to torch.nan.
            pad (bool | str): Whether to pad the data. If a string, it should be either "train" or "validate". Defaults to False.
            datasets (Optional[Mapping[str, Sequence[CellMapDataset]]]): A dictionary containing the dataset objects. Defaults to None. The dictionary should have the following structure::

                {
                    "train": Iterable[CellMapDataset],
                    "validate": Iterable[CellMapDataset],
                }.

            dataset_dict (Optional[Mapping[str, Sequence[Mapping[str, str]]]): A dictionary containing the dataset data. Defaults to None. The dictionary should have the following structure::

                {
                    "train" | "validate": [{
                        "raw": str (path to raw data),
                        "gt": str (path to ground truth data),
                    }],
                    ...
                }

            csv_path (Optional[str]): A path to a csv file containing the dataset data. Defaults to None. Each row in the csv file should have the following structure:"

                train | validate, raw path, gt path

            spatial_transforms (Optional[Sequence[dict[str, Any]]]): A sequence of dictionaries containing the spatial transformations to apply to the data. Defaults to None. The dictionary should have the following structure::

                {transform_name: {transform_args}}

            train_raw_value_transforms (Optional[Callable]): A function to apply to the raw data in training datasets. Defaults to None. Example is to add gaussian noise to the raw data.
            val_raw_value_transforms (Optional[Callable]): A function to apply to the raw data in validation datasets. Defaults to None. Example is to normalize the raw data.
            target_value_transforms (Optional[Callable | Sequence[Callable] | Mapping[str, Callable]]): A function to convert the ground truth data to target arrays. Defaults to None. Example is to convert the ground truth data to a signed distance transform. May be a single function, a list of functions, or a dictionary of functions for each class. In the case of a list of functions, it is assumed that the functions correspond to each class in the classes list in order.
            class_relation_dict (Optional[Mapping[str, Sequence[str]]]): A dictionary containing the class relations. The dictionary should have the following structure::

                {
                    "class_name": [mutually_exclusive_class_name, ...],
                    ...
                }

            force_has_data (bool): Whether to force the datasets to have data even if no ground truth data is found. Defaults to False. Useful for training with only raw data.
            context (Optional[tensorstore.Context]): The TensorStore context for the image data. Defaults to None.
            device (Optional[str | torch.device]): Device to use for the dataloaders. Defaults to None.

        Note:
            The csv_path, dataset_dict, and datasets arguments are mutually exclusive, but one must be supplied.

        """

        logger.info("Initializing CellMapDataSplit...")
        self.input_arrays = input_arrays
        self.target_arrays = target_arrays
        self.classes = classes
        self.empty_value = empty_value
        self.pad = pad
        self.device = device
        if isinstance(pad, str):
            self.pad_training = pad.lower() == "train"
            self.pad_validation = pad.lower() == "validate"
        else:
            self.pad_training = pad
            self.pad_validation = pad
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
        self.class_relation_dict = class_relation_dict
        self.context = context
        if self.dataset_dict is not None:
            self.construct(self.dataset_dict)
        self.verify_datasets()
        assert len(self.train_datasets) > 0, "No valid training datasets found."
        logger.info("CellMapDataSplit initialized.")

    def __repr__(self) -> str:
        """Returns the string representation of the class."""
        return f"CellMapDataSplit(\n\tInput arrays: {self.input_arrays}\n\tTarget arrays:{self.target_arrays}\n\tClasses: {self.classes}\n\tDataset dict: {self.dataset_dict}\n\tSpatial transforms: {self.spatial_transforms}\n\tRaw value transforms: {self.train_raw_value_transforms}\n\tGT value transforms: {self.target_value_transforms}\n\tForce has data: {self.force_has_data}\n\tContext: {self.context})"

    @property
    def train_datasets_combined(self) -> CellMapMultiDataset:
        """A multi-dataset from the combination of all training datasets."""
        try:
            return self._train_datasets_combined
        except AttributeError:
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
    def validation_datasets_combined(self) -> CellMapMultiDataset:
        """A multi-dataset from the combination of all validation datasets."""
        try:
            return self._validation_datasets_combined
        except AttributeError:
            if len(self.validation_datasets) == 0:
                UserWarning("Validation datasets not loaded.")
                self._validation_datasets_combined = CellMapMultiDataset.empty()
            else:
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
    def validation_blocks(self) -> CellMapSubset:
        """A subset of the validation datasets, tiling the validation datasets with non-overlapping blocks."""
        try:
            return self._validation_blocks
        except AttributeError:
            self._validation_blocks = CellMapSubset(
                self.validation_datasets_combined,
                self.validation_datasets_combined.validation_indices,
            )
            return self._validation_blocks

    @property
    def class_counts(self) -> dict[str, dict[str, float]]:
        """A dictionary containing the class counts for the training and validation datasets."""
        try:
            return self._class_counts
        except AttributeError:
            self._class_counts = {
                "train": self.train_datasets_combined.class_counts,
                "validate": self.validation_datasets_combined.class_counts,
            }
            return self._class_counts

    def from_csv(self, csv_path) -> dict[str, Sequence[dict[str, str]]]:
        """Loads the dataset_dict data from a csv file."""
        dataset_dict = {}
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            logger.info("Reading csv file...")
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

    def construct(self, dataset_dict) -> None:
        """Constructs the datasets from the dataset dictionary."""
        self.train_datasets = []
        self.validation_datasets = []
        self.datasets = {}
        logger.info("Constructing datasets...")
        for data_paths in tqdm(dataset_dict["train"], desc="Training datasets"):
            try:
                self.train_datasets.append(
                    CellMapDataset(
                        data_paths["raw"],
                        data_paths["gt"],
                        self.classes,
                        self.input_arrays,
                        self.target_arrays,
                        self.spatial_transforms,
                        raw_value_transforms=self.train_raw_value_transforms,
                        target_value_transforms=self.target_value_transforms,
                        is_train=True,
                        context=self.context,
                        force_has_data=self.force_has_data,
                        empty_value=self.empty_value,
                        class_relation_dict=self.class_relation_dict,
                        pad=self.pad_training,
                        device=self.device,
                    )
                )
            except ValueError as e:
                logger.warning(f"Error loading dataset: {e}")

        self.datasets["train"] = self.train_datasets

        # TODO: probably want larger arrays for validation

        if "validate" in dataset_dict:
            for data_paths in tqdm(
                dataset_dict["validate"], desc="Validation datasets"
            ):
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
                            class_relation_dict=self.class_relation_dict,
                            pad=self.pad_validation,
                            device=self.device,
                        )
                    )
                except ValueError as e:
                    logger.warning(f"Error loading dataset: {e}")

            self.datasets["validate"] = self.validation_datasets

    def verify_datasets(self) -> None:
        """Verifies that the datasets have data, and removes ones that don't from ``self.train_datasets`` and ``self.validation_datasets``."""
        if self.force_has_data:
            return
        logger.info("Verifying datasets...")
        verified_datasets = []
        for ds in tqdm(self.train_datasets, desc="Training datasets"):
            if ds.verify():
                verified_datasets.append(ds)
        self.train_datasets = verified_datasets

        verified_datasets = []
        for ds in tqdm(self.validation_datasets, desc="Validation datasets"):
            if ds.verify():
                verified_datasets.append(ds)
        self.validation_datasets = verified_datasets

    def set_raw_value_transforms(
        self,
        train_transforms: Optional[Callable] = None,
        val_transforms: Optional[Callable] = None,
    ) -> None:
        """Sets the raw value transforms for each dataset in the training/validation multi-datasets."""
        if train_transforms is not None:
            for dataset in self.train_datasets:
                dataset.set_raw_value_transforms(train_transforms)
            if hasattr(self, "_train_datasets_combined"):
                self._train_datasets_combined.set_raw_value_transforms(train_transforms)
        if val_transforms is not None:
            for dataset in self.validation_datasets:
                dataset.set_raw_value_transforms(val_transforms)
            if hasattr(self, "_validation_datasets_combined"):
                self._validation_datasets_combined.set_raw_value_transforms(
                    val_transforms
                )

    def set_target_value_transforms(self, transforms: Callable) -> None:
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

    def set_spatial_transforms(
        self,
        train_transforms: Optional[dict[str, Any]] = None,
        val_transforms: Optional[dict[str, Any]] = None,
    ) -> None:
        """Sets the raw value transforms for each dataset in the training/validation multi-dataset."""
        if train_transforms is not None:
            for dataset in self.train_datasets:
                dataset.spatial_transforms = train_transforms
            if hasattr(self, "_train_datasets_combined"):
                self._train_datasets_combined.set_spatial_transforms(train_transforms)
        if val_transforms is not None:
            for dataset in self.validation_datasets:
                dataset.spatial_transforms = val_transforms
            if hasattr(self, "_validation_datasets_combined"):
                self._validation_datasets_combined.set_spatial_transforms(
                    val_transforms
                )

    def set_arrays(
        self,
        arrays: Mapping[str, Mapping[str, Sequence[int | float]]],
        type: str = "target",
        usage: str = "validate",
    ) -> None:
        """Sets the input or target arrays for the training or validation datasets."""
        reset_attrs = []
        for dataset in self.datasets[usage]:
            if type == "inputs":
                dataset.input_arrays = arrays
            elif type == "target":
                dataset.target_arrays = arrays
            else:
                raise ValueError("Type must be 'inputs' or 'target'.")
            dataset.reset_arrays(type)

        if usage == "train":
            self.train_datasets = self.datasets["train"]
            reset_attrs.append("_train_datasets_combined")
        elif usage == "validate":
            self.validation_datasets = self.datasets["validate"]
            reset_attrs.extend(["_validation_datasets_combined", "_validation_blocks"])
        for attr in reset_attrs:
            if hasattr(self, attr):
                delattr(self, attr)

    def to(self, device: str | torch.device, non_blocking: bool = True) -> None:
        """Sets the device for the dataloaders."""
        self.device = device
        for dataset in self.train_datasets:
            dataset.to(device, non_blocking=non_blocking)
        for dataset in self.validation_datasets:
            dataset.to(device, non_blocking=non_blocking)
        if hasattr(self, "_train_datasets_combined"):
            self._train_datasets_combined.to(device, non_blocking=non_blocking)
        if hasattr(self, "_validation_datasets_combined"):
            self._validation_datasets_combined.to(device, non_blocking=non_blocking)
        if hasattr(self, "_validation_blocks"):
            self._validation_blocks.to(device, non_blocking=non_blocking)
