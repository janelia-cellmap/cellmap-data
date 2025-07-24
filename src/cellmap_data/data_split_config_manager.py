"""
Configuration manager for CellMapDataSplit.
"""

from typing import Any, Mapping, Optional, Sequence, Callable
import torch


class DataSplitConfigManager:
    """
    Manages the configuration for CellMapDataSplit.
    """

    def __init__(
        self,
        input_arrays: Mapping[str, Mapping[str, Sequence[int | float]]],
        target_arrays: Optional[
            Mapping[str, Mapping[str, Sequence[int | float]]]
        ] = None,
        classes: Optional[Sequence[str]] = None,
        empty_value: Optional[int | float] = None,
        pad: Optional[int] = None,
        dataset_dict: Optional[dict] = None,
        csv_path: Optional[str] = None,
        spatial_transforms: Optional[Callable] = None,
        train_raw_value_transforms: Optional[Callable] = None,
        val_raw_value_transforms: Optional[Callable] = None,
        target_value_transforms: Optional[Callable] = None,
        class_relation_dict: Optional[dict[str, str]] = None,
        force_has_data: bool = False,
        context: Optional[list[int]] = None,
        device: Optional[str | torch.device] = None,
    ):
        self.input_arrays = input_arrays
        self.target_arrays = target_arrays
        self.classes = classes
        self.config = {
            "input_arrays": input_arrays,
            "target_arrays": target_arrays,
            "classes": classes,
            "empty_value": empty_value,
            "pad": pad,
            "dataset_dict": dataset_dict,
            "csv_path": csv_path,
            "spatial_transforms": spatial_transforms,
            "train_raw_value_transforms": train_raw_value_transforms,
            "val_raw_value_transforms": val_raw_value_transforms,
            "target_value_transforms": target_value_transforms,
            "class_relation_dict": class_relation_dict,
            "force_has_data": force_has_data,
            "context": context,
            "device": device,
        }
