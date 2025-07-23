"""
Pydantic schemas for configuration validation.
"""

from typing import Any, Dict, List, Optional, Sequence, Mapping, Callable, Union
from pydantic import BaseModel, Field, validator
import torch


class DatasetConfig(BaseModel):
    """Pydantic model for CellMapDataset configuration."""

    raw_path: str
    target_path: str
    classes: Optional[Sequence[str]] = None
    input_arrays: Mapping[str, Mapping[str, Sequence[Union[int, float]]]]
    target_arrays: Optional[Mapping[str, Mapping[str, Sequence[Union[int, float]]]]] = (
        None
    )
    spatial_transforms: Optional[Mapping[str, Mapping]] = None
    raw_value_transforms: Optional[Callable] = None
    target_value_transforms: Optional[
        Union[Callable, Sequence[Callable], Mapping[str, Callable]]
    ] = None
    class_relation_dict: Optional[Mapping[str, Sequence[str]]] = None
    is_train: bool = False
    axis_order: str = "zyx"
    context: Optional[Any] = None
    rng: Optional[Any] = None
    force_has_data: bool = False
    empty_value: Union[float, int] = torch.nan
    pad: bool = True
    device: Optional[Union[str, torch.device]] = None
    max_workers: Optional[int] = None

    class Config:
        arbitrary_types_allowed = True


class DataLoaderConfig(BaseModel):
    """Pydantic model for CellMapDataLoader configuration."""

    dataset: Any  # Should be a CellMapDataset or compatible
    batch_size: int = Field(..., gt=0)
    num_workers: int = Field(0, ge=0)
    pin_memory: bool = True
    shuffle: bool = False

    class Config:
        arbitrary_types_allowed = True


class DataSplitConfig(BaseModel):
    """Pydantic model for CellMapDataSplit configuration."""

    input_arrays: Mapping[str, Mapping[str, Sequence[Union[int, float]]]]
    target_arrays: Optional[Mapping[str, Mapping[str, Sequence[Union[int, float]]]]] = (
        None
    )
    classes: Optional[Sequence[str]] = None
    empty_value: Union[float, int] = torch.nan
    pad: Union[bool, str] = False
    datasets: Optional[Mapping[str, Sequence[Any]]] = None  # Sequence[CellMapDataset]
    dataset_dict: Optional[Mapping[str, Sequence[Mapping[str, str]]]] = None
    csv_path: Optional[str] = None
    spatial_transforms: Optional[Mapping[str, Mapping]] = None
    train_raw_value_transforms: Optional[Any] = None  # Callable
    val_raw_value_transforms: Optional[Any] = None  # Callable
    target_value_transforms: Optional[Any] = (
        None  # Union[Callable, Sequence[Callable], Mapping[str, Callable]]
    )
    class_relation_dict: Optional[Mapping[str, Sequence[str]]] = None
    force_has_data: bool = False
    context: Optional[Any] = None
    device: Optional[Union[str, torch.device]] = None

    class Config:
        arbitrary_types_allowed = True
