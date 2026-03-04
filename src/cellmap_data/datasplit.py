"""CellMapDataSplit: train/validation dataset management."""

from __future__ import annotations

import csv
import logging
import os
from typing import Any, Callable, Mapping, Optional, Sequence

import torch
import torchvision.transforms.v2 as T
from torch.utils.data import Subset
from tqdm import tqdm

from .dataset import CellMapDataset
from .multidataset import CellMapMultiDataset
from .transforms import Binarize, NaNtoNum

logger = logging.getLogger(__name__)


class CellMapDataSplit:
    """Manages train/validation splits for CellMap data.

    Reads dataset paths from a CSV file, a ``dataset_dict``, or a
    pre-built ``datasets`` mapping, then constructs
    :class:`CellMapDataset` objects and exposes combined datasets for
    training and validation.

    Parameters
    ----------
    input_arrays:
        ``{name: {"shape": (z,y,x), "scale": (z,y,x)}}``
    target_arrays:
        Same structure as *input_arrays*.
    classes:
        Segmentation class names.
    pad:
        Pad strategy: ``False``, ``True``, ``"train"``, or ``"validate"``.
    datasets:
        Pre-built ``{"train": [CellMapDataset, …], "validate": […]}``
        mapping.  Mutually exclusive with *dataset_dict* / *csv_path*.
    dataset_dict:
        ``{"train": [{"raw": path, "gt": path}, …], "validate": […]}``.
    csv_path:
        Path to CSV with rows ``split,raw_path,gt_path[,raw_name,gt_name]``.
    spatial_transforms:
        Augmentation config for training datasets.
    train_raw_value_transforms:
        Transform applied to raw data during training.
    val_raw_value_transforms:
        Transform applied to raw data during validation.
    target_value_transforms:
        Transform applied to GT labels.
    class_relation_dict:
        Mutual-exclusion class relations (stored, not used for inference).
    force_has_data:
        Skip empty-data check on each dataset.
    device:
        Ignored (API compatibility).
    """

    def __init__(
        self,
        input_arrays: Mapping[str, Mapping[str, Sequence[int | float]]],
        target_arrays: Optional[
            Mapping[str, Mapping[str, Sequence[int | float]]]
        ] = None,
        classes: Optional[Sequence[str]] = None,
        pad: bool | str = False,
        datasets: Optional[Mapping[str, Sequence[CellMapDataset]]] = None,
        dataset_dict: Optional[Mapping[str, Sequence[Mapping[str, str]]]] = None,
        csv_path: Optional[str] = None,
        spatial_transforms: Optional[Mapping[str, Any]] = None,
        train_raw_value_transforms: Optional[Callable] = T.Compose(
            [
                T.ToDtype(torch.float, scale=True),
                NaNtoNum({"nan": 0, "posinf": None, "neginf": None}),
            ]
        ),
        val_raw_value_transforms: Optional[Callable] = T.Compose(
            [
                T.ToDtype(torch.float, scale=True),
                NaNtoNum({"nan": 0, "posinf": None, "neginf": None}),
            ]
        ),
        target_value_transforms: Optional[Callable] = T.Compose(
            [T.ToDtype(torch.float), Binarize()]
        ),
        class_relation_dict: Optional[Mapping[str, Sequence[str]]] = None,
        force_has_data: bool = False,
        context: Optional[Any] = None,  # ignored, kept for API compat
        device: Optional[str | torch.device] = None,
    ) -> None:
        self.input_arrays = dict(input_arrays)
        self.target_arrays = dict(target_arrays) if target_arrays else {}
        self.classes = list(classes) if classes else []
        self.pad = pad
        self.spatial_transforms = spatial_transforms
        self.train_raw_value_transforms = train_raw_value_transforms
        self.val_raw_value_transforms = val_raw_value_transforms
        self.target_value_transforms = target_value_transforms
        self.class_relation_dict = class_relation_dict
        self.force_has_data = force_has_data

        # Storage for train/val datasets
        self.train_datasets: list[CellMapDataset] = []
        self._validation_datasets: list[CellMapDataset] = []

        if datasets is not None:
            self.train_datasets = list(datasets.get("train", []))
            self._validation_datasets = list(datasets.get("validate", []))
        elif dataset_dict is not None:
            self._construct(dataset_dict)
            self._verify_datasets()
        elif csv_path is not None:
            dataset_dict = self._parse_csv(csv_path)
            self._construct(dataset_dict)
            self._verify_datasets()
        # else: empty split; user can call _construct later or set datasets directly

    # ------------------------------------------------------------------
    # CSV parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_csv(
        csv_path: str,
    ) -> dict[str, list[dict[str, str]]]:
        """Parse the dataset CSV into a ``dataset_dict``.

        Expected CSV columns: ``split, raw_path, gt_path`` (and optionally
        ``raw_name``, ``gt_name`` which are ignored).
        """
        result: dict[str, list[dict[str, str]]] = {
            "train": [],
            "validate": [],
        }
        with open(csv_path, newline="") as fh:
            reader = csv.reader(fh)
            for row in reader:
                if not row or row[0].startswith("#"):
                    continue
                if len(row) < 3:
                    logger.warning("Skipping malformed CSV row: %s", row)
                    continue
                split = row[0].strip()
                raw_path = row[1].strip()
                gt_path = row[2].strip()
                if split not in result:
                    result[split] = []
                result[split].append({"raw": raw_path, "gt": gt_path})
        return result

    # ------------------------------------------------------------------
    # Dataset construction
    # ------------------------------------------------------------------

    def _construct(
        self, dataset_dict: Mapping[str, Sequence[Mapping[str, str]]]
    ) -> None:
        """Build CellMapDataset objects from the dict of raw/gt path pairs."""
        for split, entries in dataset_dict.items():
            is_train = split.lower().startswith("train")
            pad = (
                self.pad
                if isinstance(self.pad, bool)
                else (split.lower() in self.pad.lower())
            )
            raw_tx = (
                self.train_raw_value_transforms
                if is_train
                else self.val_raw_value_transforms
            )
            spatial_tx = self.spatial_transforms if is_train else None

            for entry in entries:
                raw_path = entry.get("raw", "")
                gt_path = entry.get("gt", "")
                if not raw_path:
                    continue
                try:
                    ds = CellMapDataset(
                        raw_path=raw_path,
                        target_path=gt_path,
                        classes=self.classes,
                        input_arrays=self.input_arrays,
                        target_arrays=self.target_arrays,
                        pad=pad,
                        spatial_transforms=spatial_tx,
                        raw_value_transforms=raw_tx,
                        target_value_transforms=self.target_value_transforms,
                        class_relation_dict=self.class_relation_dict,
                        force_has_data=self.force_has_data,
                    )
                    if is_train:
                        self.train_datasets.append(ds)
                    else:
                        self._validation_datasets.append(ds)
                except Exception as exc:
                    logger.warning(
                        "Skipping dataset raw=%r gt=%r: %s", raw_path, gt_path, exc
                    )

    def _verify_datasets(self) -> None:
        """Remove datasets that report no valid data."""
        if self.force_has_data:
            return
        self.train_datasets = [
            ds
            for ds in tqdm(
                self.train_datasets,
                desc="Verifying train datasets",
                leave=False,
            )
            if ds.verify()
        ]
        self._validation_datasets = [
            ds
            for ds in tqdm(
                self._validation_datasets,
                desc="Verifying val datasets",
                leave=False,
            )
            if ds.verify()
        ]

    # ------------------------------------------------------------------
    # Cached combined datasets
    # ------------------------------------------------------------------

    @property
    def train_datasets_combined(self) -> CellMapMultiDataset:
        """Combined training dataset for use with DataLoader."""
        if "train_datasets_combined" not in self.__dict__:
            self.__dict__["train_datasets_combined"] = CellMapMultiDataset(
                datasets=self.train_datasets,
                classes=self.classes,
                input_arrays=self.input_arrays,
                target_arrays=self.target_arrays,
            )
        return self.__dict__["train_datasets_combined"]

    @property
    def validation_datasets_combined(self) -> CellMapMultiDataset:
        """Combined validation dataset."""
        if "validation_datasets_combined" not in self.__dict__:
            self.__dict__["validation_datasets_combined"] = CellMapMultiDataset(
                datasets=self._validation_datasets,
                classes=self.classes,
                input_arrays=self.input_arrays,
                target_arrays=self.target_arrays,
            )
        return self.__dict__["validation_datasets_combined"]

    @property
    def validation_datasets(self) -> list[CellMapDataset]:
        """List of individual validation datasets."""
        return self._validation_datasets

    @property
    def validation_blocks(self) -> Subset:
        """Non-overlapping validation tile indices wrapped in a Subset."""
        if "validation_blocks" not in self.__dict__:
            combined = self.validation_datasets_combined
            indices = combined.validation_indices
            self.__dict__["validation_blocks"] = Subset(combined, indices)
        return self.__dict__["validation_blocks"]

    @property
    def class_counts(self) -> dict[str, Any]:
        """Train and validation class counts."""
        return {
            "train": self.train_datasets_combined.class_counts,
            "validate": self.validation_datasets_combined.class_counts,
        }

    # ------------------------------------------------------------------
    # Cache invalidation
    # ------------------------------------------------------------------

    def _invalidate(self) -> None:
        """Clear all cached combined-dataset properties."""
        for key in (
            "train_datasets_combined",
            "validation_datasets_combined",
            "validation_blocks",
        ):
            self.__dict__.pop(key, None)

    # ------------------------------------------------------------------
    # Setters (invalidate cache)
    # ------------------------------------------------------------------

    def set_raw_value_transforms(
        self,
        train_transforms: Optional[Callable] = None,
        val_transforms: Optional[Callable] = None,
    ) -> None:
        self.train_raw_value_transforms = train_transforms
        self.val_raw_value_transforms = val_transforms
        for ds in self.train_datasets:
            ds.set_raw_value_transforms(train_transforms)
        for ds in self._validation_datasets:
            ds.set_raw_value_transforms(val_transforms)
        self._invalidate()

    def set_target_value_transforms(self, transforms: Optional[Callable]) -> None:
        self.target_value_transforms = transforms
        for ds in self.train_datasets + self._validation_datasets:
            ds.set_target_value_transforms(transforms)
        self._invalidate()

    def set_spatial_transforms(
        self, spatial_transforms: Optional[Mapping[str, Any]]
    ) -> None:
        self.spatial_transforms = spatial_transforms
        for ds in self.train_datasets:
            ds.set_spatial_transforms(spatial_transforms)
        self._invalidate()

    def set_arrays(
        self,
        arrays: Mapping[str, Mapping[str, Sequence[int | float]]],
        type: str = "target",
        usage: str = "validate",
    ) -> None:
        if type == "target":
            self.target_arrays = dict(arrays)
        else:
            self.input_arrays = dict(arrays)
        self._invalidate()

    def to(self, device: str | torch.device) -> "CellMapDataSplit":
        self.train_datasets_combined.to(device)
        self.validation_datasets_combined.to(device)
        return self

    def __repr__(self) -> str:
        return (
            f"CellMapDataSplit("
            f"train={len(self.train_datasets)}, "
            f"val={len(self._validation_datasets)}, "
            f"classes={self.classes})"
        )
