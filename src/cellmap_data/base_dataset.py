"""Abstract base class for CellMap dataset objects."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Mapping, Sequence

import torch


class CellMapBaseDataset(ABC):
    """
    Abstract base class for CellMap dataset objects.

    This class defines the common interface that all CellMap dataset objects
    must implement, ensuring consistency across different dataset types.

    Note: `classes`, `input_arrays`, and `target_arrays` are not abstract
    properties because implementing classes define them as instance attributes
    in __init__, not as properties.
    """

    # These are instance attributes set in __init__, not properties
    classes: Sequence[str] | None
    input_arrays: Mapping[str, Mapping[str, Any]]
    target_arrays: Mapping[str, Mapping[str, Any]] | None

    @property
    @abstractmethod
    def class_counts(self) -> dict[str, float]:
        """
        Return the number of samples in each class, normalized by resolution.

        Returns
        -------
        dict[str, float]
            Dictionary mapping class names to their counts.
        """
        pass

    @property
    @abstractmethod
    def class_weights(self) -> dict[str, float]:
        """
        Return the class weights based on the number of samples in each class.

        Returns
        -------
        dict[str, float]
            Dictionary mapping class names to their weights.
        """
        pass

    @property
    @abstractmethod
    def validation_indices(self) -> Sequence[int]:
        """
        Return the indices for the validation set.

        Returns
        -------
        Sequence[int]
            List of validation indices.
        """
        pass

    @abstractmethod
    def to(
        self, device: str | torch.device, non_blocking: bool = True
    ) -> "CellMapBaseDataset":
        """
        Move the dataset to the specified device.

        Parameters
        ----------
        device : str | torch.device
            The target device.
        non_blocking : bool, optional
            Whether to use non-blocking transfer, by default True.

        Returns
        -------
        CellMapBaseDataset
            Self for method chaining.
        """
        pass

    @abstractmethod
    def set_raw_value_transforms(self, transforms: Callable) -> None:
        """
        Set the value transforms for raw input data.

        Parameters
        ----------
        transforms : Callable
            Transform function to apply to raw data.
        """
        pass

    @abstractmethod
    def set_target_value_transforms(self, transforms: Callable) -> None:
        """
        Set the value transforms for target data.

        Parameters
        ----------
        transforms : Callable
            Transform function to apply to target data.
        """
        pass
