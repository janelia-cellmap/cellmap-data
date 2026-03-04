"""CellMapDataLoader: thin wrapper around PyTorch DataLoader."""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterator, Optional, Sequence, Union

import torch
import torch.utils.data
from torch.utils.data import DataLoader, Dataset, Subset

from .sampler import ClassBalancedSampler

logger = logging.getLogger(__name__)


class CellMapDataLoader:
    """PyTorch-compatible DataLoader for CellMap datasets.

    Wraps :class:`torch.utils.data.DataLoader` with optional
    :class:`~cellmap_data.sampler.ClassBalancedSampler` for class-balanced
    training.

    Parameters
    ----------
    dataset:
        A :class:`~cellmap_data.dataset.CellMapDataset`,
        :class:`~cellmap_data.multidataset.CellMapMultiDataset`, or any
        compatible dataset / ``Subset``.
    classes:
        Class names.  Defaults to ``dataset.classes``.
    batch_size:
        Samples per batch.
    num_workers:
        DataLoader worker processes (0 = main process only).
    weighted_sampler:
        If ``True`` and ``is_train=True``, use
        :class:`ClassBalancedSampler` (requires ``dataset`` to implement
        ``get_crop_class_matrix()``).
    sampler:
        Explicit sampler; overrides *weighted_sampler*.
    is_train:
        Training mode (enables random sampling and the weighted sampler).
    device:
        Ignored — tensors are returned on CPU; move them in training loop.
    iterations_per_epoch:
        Number of samples per epoch when using ClassBalancedSampler.
        Defaults to ``len(dataset)``.
    **kwargs:
        Forwarded to :class:`torch.utils.data.DataLoader`.
    """

    def __init__(
        self,
        dataset: Dataset,
        classes: Optional[Sequence[str]] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        weighted_sampler: bool = False,
        sampler: Optional[Union[torch.utils.data.Sampler, Callable]] = None,
        is_train: bool = True,
        rng: Optional[torch.Generator] = None,
        device: Optional[str | torch.device] = None,
        iterations_per_epoch: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.dataset = dataset
        self.classes = (
            classes if classes is not None else getattr(dataset, "classes", [])
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.is_train = is_train
        self.iterations_per_epoch = iterations_per_epoch
        self._kwargs = kwargs

        # Resolve device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        # Build sampler
        if sampler is not None:
            self._sampler = sampler
        elif weighted_sampler and is_train:
            if hasattr(dataset, "get_crop_class_matrix"):
                n_samples = iterations_per_epoch or len(dataset)
                self._sampler: Any = ClassBalancedSampler(dataset, n_samples)
            else:
                logger.warning(
                    "weighted_sampler=True but dataset does not implement "
                    "get_crop_class_matrix(); falling back to default sampler."
                )
                self._sampler = None
        else:
            self._sampler = None

        # pin_memory: use on CUDA, skip otherwise to avoid issues
        pin = kwargs.pop("pin_memory", str(device).startswith("cuda"))

        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(is_train and self._sampler is None),
            sampler=self._sampler,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            pin_memory=pin,
            **self._kwargs,
        )

    # ------------------------------------------------------------------
    # Collation
    # ------------------------------------------------------------------

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict[str, Any]:
        """Stack tensor values; preserve string / non-tensor items."""
        if not batch:
            return {}
        keys = batch[0].keys()
        result: dict[str, Any] = {}
        for key in keys:
            values = [item[key] for item in batch]
            if isinstance(values[0], torch.Tensor):
                result[key] = torch.stack(values)
            elif key == "__metadata__":
                result[key] = values
            else:
                try:
                    result[key] = torch.stack(values)
                except (TypeError, RuntimeError):
                    result[key] = values
        return result

    # ------------------------------------------------------------------
    # DataLoader interface
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[dict]:
        return iter(self.loader)

    def __len__(self) -> int:
        return len(self.loader)

    def to(self, device: str | torch.device) -> "CellMapDataLoader":
        """Move the underlying dataset to *device* (no-op for CPU datasets)."""
        if hasattr(self.dataset, "to"):
            self.dataset.to(device)
        self.device = device
        return self

    def __repr__(self) -> str:
        return (
            f"CellMapDataLoader(dataset={self.dataset!r}, "
            f"batch_size={self.batch_size}, is_train={self.is_train})"
        )
