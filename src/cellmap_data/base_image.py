"""Abstract base class for CellMap image objects."""

from abc import ABC, abstractmethod
from typing import Any, Mapping

import torch


class CellMapImageBase(ABC):
    """
    Abstract base class for CellMap image objects.

    This class defines the common interface that all CellMap image objects
    must implement, ensuring consistency across different image types.
    """

    @abstractmethod
    def __getitem__(self, center: Mapping[str, float]) -> torch.Tensor:
        """
        Return image data centered around the given point.

        Parameters
        ----------
        center : Mapping[str, float]
            The center coordinates in world units.

        Returns
        -------
        torch.Tensor
            The image data as a PyTorch tensor.
        """
        pass

    @property
    @abstractmethod
    def bounding_box(self) -> Mapping[str, tuple[float, float]] | None:
        """
        Return the bounding box of the image in world units.

        Returns
        -------
        Mapping[str, tuple[float, float]] | None
            Dictionary mapping axis names to (min, max) tuples, or None.
        """
        pass

    @property
    @abstractmethod
    def sampling_box(self) -> Mapping[str, tuple[float, float]] | None:
        """
        Return the sampling box of the image in world units.

        The sampling box is the region where centers can be drawn from and
        still have full samples drawn from within the bounding box.

        Returns
        -------
        Mapping[str, tuple[float, float]] | None
            Dictionary mapping axis names to (min, max) tuples, or None.
        """
        pass

    @property
    @abstractmethod
    def class_counts(self) -> float | dict[str, float]:
        """
        Return the number of voxels for each class in the image.

        Returns
        -------
        float | dict[str, float]
            Class counts, either as a single float or dictionary.
        """
        pass

    @abstractmethod
    def to(self, device: str | torch.device, non_blocking: bool = True) -> None:
        """
        Move the image data to the specified device.

        Parameters
        ----------
        device : str | torch.device
            The target device.
        non_blocking : bool, optional
            Whether to use non-blocking transfer, by default True.
        """
        pass

    @abstractmethod
    def set_spatial_transforms(self, transforms: Mapping[str, Any] | None) -> None:
        """
        Set spatial transformations for the image data.

        Parameters
        ----------
        transforms : Mapping[str, Any] | None
            Dictionary of spatial transformations to apply.
        """
        pass
