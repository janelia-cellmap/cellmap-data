from .schemas import DatasetConfig, DataLoaderConfig, DataSplitConfig
from pydantic import ValidationError
import logging
from typing import TYPE_CHECKING, Dict, Any
import torch

if TYPE_CHECKING:
    from ..multidataset import CellMapMultiDataset
from pydantic import ValidationError
import logging
from typing import TYPE_CHECKING, Dict, Any

if TYPE_CHECKING:
    from ..multidataset import CellMapMultiDataset

logger = logging.getLogger(__name__)


def validate_multidataset(multi_dataset: "CellMapMultiDataset") -> bool:
    """
    Centralized validation for CellMapMultiDataset.
    Checks that all datasets have matching classes and array keys.
    Returns True if valid, False otherwise.
    Logs errors for mismatches.
    """
    if len(multi_dataset.datasets) == 0:
        logger.warning("Multi-dataset is empty.")
        return False
    n_verified_datasets = 0
    for dataset in multi_dataset.datasets:
        if hasattr(dataset, "verify"):
            n_verified_datasets += int(dataset.verify())
        else:
            logger.warning(f"Dataset {dataset} missing verify() method.")
            continue
        try:
            assert (
                dataset.classes == multi_dataset.classes
            ), "All datasets must have the same classes."
            assert set(dataset.input_arrays.keys()) == set(
                multi_dataset.input_arrays.keys()
            ), "All datasets must have the same input arrays."
            if multi_dataset.target_arrays is not None:
                assert set(dataset.target_arrays.keys()) == set(
                    multi_dataset.target_arrays.keys()
                ), "All datasets must have the same target arrays."
        except AssertionError as e:
            logger.error(
                f"Dataset {dataset} does not match the expected structure: {e}"
            )
            return False
    return n_verified_datasets > 0


class ConfigValidator:
    """
    Validates configuration dictionaries for datasets and dataloaders
    using Pydantic schemas.
    """

    @staticmethod
    def validate_dataset_config(config: Dict[str, Any]) -> bool:
        """
        Validates the configuration for a CellMapDataset.

        Args:
            config: A dictionary containing dataset configuration.

        Returns:
            True if the configuration is valid, False otherwise.
        """
        try:
            DatasetConfig(**config)
            return True
        except ValidationError as e:
            logger.error(f"Dataset configuration validation failed: {e}")
            return False

    @staticmethod
    def validate_dataloader_config(config: Dict[str, Any]) -> bool:
        """
        Validates the configuration for a CellMapDataLoader.

        Args:
            config: A dictionary containing dataloader configuration.

        Returns:
            True if the configuration is valid, False otherwise.
        """
        try:
            DataLoaderConfig(**config)
            return True
        except ValidationError as e:
            logger.error(f"DataLoader configuration validation failed: {e}")
            return False

    @staticmethod
    def validate_datasplit_config(config: Dict[str, Any]) -> bool:
        """
        Validates the configuration for a CellMapDataSplit.

        Args:
            config: A dictionary containing datasplit configuration.

        Returns:
            True if the configuration is valid, False otherwise.
        """
        try:
            DataSplitConfig(**config)
            return True
        except ValidationError as e:
            logger.error(f"DataSplit configuration validation failed: {e}")
            return False


class SchemaValidator:
    """
    Provides schema-based validation for complex configurations.
    This could be extended with a library like Pydantic or jsonschema.
    """

    @staticmethod
    def get_dataset_schema() -> Dict[str, Any]:
        """Returns the expected schema for a dataset configuration."""
        return {
            "raw_path": str,
            "target_path": str,
            "classes": list,
            "input_arrays": dict,
            "target_arrays": (dict, type(None)),
            "spatial_transforms": (dict, type(None)),
            "raw_value_transforms": (object, type(None)),  # Should be callable
            "target_value_transforms": (object, type(None)),  # Should be callable
            "is_train": bool,
        }

    def validate_with_schema(
        self, config: Dict[str, Any], schema: Dict[str, Any]
    ) -> bool:
        """
        Validates a configuration against a schema.

        Args:
            config: The configuration dictionary to validate.
            schema: The schema to validate against.

        Returns:
            True if valid, otherwise raises an error.
        """
        for key, expected_type in schema.items():
            if key not in config:
                # Allow optional keys if type is a tuple containing None
                if isinstance(expected_type, tuple) and type(None) in expected_type:
                    continue
                raise ValueError(f"Missing required config key: '{key}'")

            if not isinstance(config[key], expected_type):
                # Special check for callables
                if expected_type is object and callable(config[key]):
                    continue
                raise TypeError(
                    f"Invalid type for '{key}'. Expected {expected_type}, got {type(config[key])}."
                )

        return True


def validate_data_structure(data: Any) -> bool:
    """
    Validates the structure of data samples or batches.

    Args:
        data: The data to validate (e.g., a sample dictionary).

    Returns:
        True if the structure is valid, otherwise raises an error.
    """
    if not isinstance(data, dict):
        raise TypeError("Data must be a dictionary.")

    if "raw" not in data or not isinstance(data["raw"], torch.Tensor):
        raise ValueError("Data must contain a 'raw' tensor.")

    if "gt" not in data or not isinstance(data["gt"], torch.Tensor):
        raise ValueError("Data must contain a 'gt' tensor.")

    return True
