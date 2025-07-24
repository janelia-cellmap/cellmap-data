"""
Factory for creating dataset objects.
"""

from typing import Any, Mapping, Optional, Sequence
from .dataset import CellMapDataset
from .multidataset import CellMapMultiDataset


class DatasetFactory:
    """
    A factory for creating dataset objects.
    """

    @staticmethod
    def create_dataset(config: Mapping[str, Any]) -> CellMapDataset:
        """
        Creates a CellMapDataset from a configuration dictionary.

        Args:
            config: A dictionary containing the dataset configuration.

        Returns:
            A CellMapDataset object.
        """
        return CellMapDataset(**config)

    @staticmethod
    def create_multidataset(
        datasets: Sequence[CellMapDataset],
        **kwargs: Any,
    ) -> CellMapMultiDataset:
        """
        Creates a CellMapMultiDataset from a list of datasets.

        Args:
            datasets: A list of CellMapDataset objects.
            **kwargs: Additional arguments to pass to the CellMapMultiDataset constructor.

        Returns:
            A CellMapMultiDataset object.
        """
        return CellMapMultiDataset(datasets=datasets, **kwargs)
