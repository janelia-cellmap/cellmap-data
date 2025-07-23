# Configuration validation logic for cellmap-data
from typing import TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from .multidataset import CellMapMultiDataset

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
