from .validation.schemas import DataSplitConfig
from .validation.validation import ConfigValidator
import csv
import os
from typing import Any, Callable, Mapping, Optional, Sequence
import tensorstore
import torch
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader
from tqdm import tqdm
from .transforms import NaNtoNum, Normalize, Binarize
from .dataset import CellMapDataset
from .multidataset import CellMapMultiDataset
from .subdataset import CellMapSubset
from .dataset_factory import DatasetFactory
from .data_split_config_manager import DataSplitConfigManager
import logging

logger = logging.getLogger(__name__)


class CellMapDataSplit:
    """
    A class for managing data splits for CellMap datasets.
    """

    def __init__(
        self,
        input_arrays: dict[str, dict[str, Sequence[int | float]]],
        target_arrays: dict[str, dict[str, Sequence[int | float]]],
        classes: list[str],
        empty_value: int | float,
        pad: int,
        dataset_dict: Optional[dict[str, list[dict[str, Any]]]] = None,
        csv_path: Optional[str] = None,
        datasets_in: Optional[dict[str, list[CellMapDataset]]] = None,
        spatial_transforms: Optional[Callable] = None,
        train_raw_value_transforms: Optional[Callable] = None,
        val_raw_value_transforms: Optional[Callable] = None,
        target_value_transforms: Optional[Callable] = None,
        class_relation_dict: Optional[dict[str, str]] = None,
        force_has_data: bool = False,
        context: Optional[list[int]] = None,
        device: Optional[str | torch.device] = None,
    ):
        self._config_manager = DataSplitConfigManager(
            input_arrays=input_arrays,
            target_arrays=target_arrays,
            classes=classes,
            empty_value=empty_value,
            pad=pad,
            dataset_dict=dataset_dict,
            csv_path=csv_path,
            spatial_transforms=spatial_transforms,
            train_raw_value_transforms=train_raw_value_transforms,
            val_raw_value_transforms=val_raw_value_transforms,
            target_value_transforms=target_value_transforms,
            class_relation_dict=class_relation_dict,
            force_has_data=force_has_data,
            context=context,
            device=device,
        )
        self._dataset_factory = DatasetFactory()

        # Maintain attributes for backward compatibility
        for key, value in self._config_manager.config.items():
            setattr(self, key, value)
        self.input_arrays = input_arrays
        self.target_arrays = target_arrays
        self.classes = classes

        self.datasets = {}
        self._train_dataset = None
        self._validate_dataset = None
        self._class_counts: Optional[Mapping[str, Any]] = None

        if datasets_in is not None:
            self.datasets = datasets_in
        elif self._config_manager.config.get("dataset_dict") is not None:
            self.datasets = self.get_datasets_from_dict(
                self._config_manager.config["dataset_dict"]
            )
        elif self._config_manager.config.get("csv_path") is not None:
            self.datasets = self.get_datasets_from_csv(
                self._config_manager.config["csv_path"]
            )
        else:
            raise ValueError(
                "Must provide one of 'datasets_in', 'dataset_dict', or 'csv_path'."
            )

        self.train_datasets = self.datasets.get("train", [])
        self.validation_datasets = self.datasets.get("validate", [])
        self.verify_datasets()
        assert len(self.train_datasets) > 0, "No valid training datasets found."
        logger.info("CellMapDataSplit initialized.")

    def __repr__(self) -> str:
        return (
            f"CellMapDataSplit(\n"
            f"\tInput arrays: {self.input_arrays}\n"
            f"\tTarget arrays:{self.target_arrays}\n"
            f"\tClasses: {self.classes}\n"
            f"\tDataset dict: {self._config_manager.config['dataset_dict']}\n"
            f"\tSpatial transforms: {self._config_manager.config['spatial_transforms']}\n"
            f"\tTrain raw value transforms: {self._config_manager.config['train_raw_value_transforms']}\n"
            f"\tTarget value transforms: {self._config_manager.config['target_value_transforms']}\n"
            f"\tForce has data: {self._config_manager.config['force_has_data']}\n"
            f"\tContext: {self._config_manager.config['context']}\n"
            f")"
        )

    def get_datasets_from_dict(self, dataset_dict):
        """
        Gets datasets from a dictionary.
        """
        datasets = {"train": [], "validate": []}
        for key, value in dataset_dict.items():
            for item in value:
                ds = CellMapDataset(
                    item["raw"],
                    item["gt"],
                    self.classes,
                    self.input_arrays,
                    self.target_arrays,
                    is_train=key == "train",
                    context=self._config_manager.config["context"],
                    force_has_data=self._config_manager.config["force_has_data"],
                    empty_value=self._config_manager.config["empty_value"],
                    class_relation_dict=self._config_manager.config[
                        "class_relation_dict"
                    ],
                    pad=self._config_manager.config["pad"],
                    device=self._config_manager.config["device"],
                )
                datasets[key].append(ds)
        return datasets

    def get_datasets_from_csv(self, csv_path):
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

        return self.get_datasets_from_dict(dataset_dict)

    def verify_datasets(self):
        """
        Verifies the datasets in the split.
        """
        for name, datasets in self.datasets.items():
            logger.info(f"Verifying {name} datasets...")
            for ds in datasets:
                if self._config_manager.config["force_has_data"] or ds.has_data:
                    pass
                else:
                    logger.warning(f"Dataset {ds.raw_path} has no data, removing.")
                    datasets.remove(ds)

    @property
    def train_dataset(self) -> CellMapMultiDataset | CellMapSubset:
        """A multi-dataset from the combination of all training datasets."""
        if self._train_dataset is None:
            self._train_dataset = self.get_train_dataset()
        return self._train_dataset

    def get_train_dataset(self) -> CellMapMultiDataset | CellMapSubset:
        logger.info("Constructing training dataset...")
        return self.construct(self.train_datasets, "train")

    @property
    def validate_dataset(self) -> CellMapMultiDataset | CellMapSubset:
        """A multi-dataset from the combination of all validation datasets."""
        if self._validate_dataset is None:
            self._validate_dataset = self.get_validate_dataset()
        return self._validate_dataset

    def get_validate_dataset(self) -> CellMapMultiDataset | CellMapSubset:
        logger.info("Constructing validation dataset...")
        return self.construct(self.validation_datasets, "validate")

    def get_class_counts(self) -> Mapping[str, Any]:
        """
        Returns the class counts for the split.
        """
        if self._class_counts is None:
            self._class_counts = {
                "train": self.train_dataset.class_counts,
                "validate": self.validate_dataset.class_counts,
            }
        return self._class_counts

    @property
    def class_counts(self) -> Mapping[str, Any]:
        """
        Returns the class counts for the split.
        """
        return self.get_class_counts()

    def construct(
        self, datasets: list[CellMapDataset], name: str
    ) -> CellMapMultiDataset | CellMapSubset:
        """
        Constructs a dataset from a list of datasets.

        Args:
            datasets (list[CellMapDataset]): A list of datasets to construct from.
            name (str): The name of the dataset.

        Returns:
            A CellMapMultiDataset or CellMapSubset.
        """
        filtered_datasets = [
            ds
            for ds in datasets
            if self._config_manager.config["force_has_data"] or ds.has_data
        ]

        if not filtered_datasets:
            raise ValueError(f"No valid datasets found for {name}")

        if name == "train":
            return self._dataset_factory.create_multidataset(
                filtered_datasets,
                spatial_transforms=self._config_manager.config["spatial_transforms"],
                raw_value_transforms=self._config_manager.config[
                    "train_raw_value_transforms"
                ],
                target_value_transforms=self._config_manager.config[
                    "target_value_transforms"
                ],
                context=self._config_manager.config["context"],
            )
        elif name == "validate":
            return self._dataset_factory.create_multidataset(
                filtered_datasets,
                spatial_transforms=self._config_manager.config["spatial_transforms"],
                raw_value_transforms=self._config_manager.config[
                    "val_raw_value_transforms"
                ],
                target_value_transforms=self._config_manager.config[
                    "target_value_transforms"
                ],
                context=self._config_manager.config["context"],
            )
        else:
            return self._dataset_factory.create_multidataset(
                filtered_datasets,
                spatial_transforms=self._config_manager.config["spatial_transforms"],
                raw_value_transforms=self._config_manager.config[
                    "val_raw_value_transforms"
                ],
                target_value_transforms=self._config_manager.config[
                    "target_value_transforms"
                ],
                context=self._config_manager.config["context"],
            )

    def get_dataloader(self, name: str, **kwargs) -> DataLoader:
        """
        Gets a dataloader for a dataset.

        Args:
            name (str): The name of the dataset.
            **kwargs: Additional arguments for the DataLoader.

        Returns:
            A DataLoader for the dataset.
        """
        if name == "train":
            dataset = self.train_dataset
        elif name == "validate":
            dataset = self.validate_dataset
        else:
            raise ValueError(f"Unknown dataset name: {name}")
        return DataLoader(dataset, **kwargs)

    def get_train_dataloader(self, **kwargs) -> DataLoader:
        """
        Gets the dataloader for the training dataset.

        Returns:
            A DataLoader for the training dataset.
        """
        return self.get_dataloader("train", **kwargs)

    def get_validate_dataloader(self, **kwargs) -> DataLoader:
        """
        Gets the dataloader for the validation dataset.

        Returns:
            A DataLoader for the validation dataset.
        """
        return self.get_dataloader("validate", **kwargs)
