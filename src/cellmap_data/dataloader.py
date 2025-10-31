import os
import numpy as np
import torch
import torch.utils.data
import logging
from typing import Callable, Optional, Sequence, Union, Any

from .mutable_sampler import MutableSubsetRandomSampler
from .subdataset import CellMapSubset
from .dataset import CellMapDataset
from .multidataset import CellMapMultiDataset
from .dataset_writer import CellMapDatasetWriter

logger = logging.getLogger(__name__)


class CellMapDataLoader:
    """
    Optimized DataLoader wrapper for CellMapDataset that uses PyTorch's native DataLoader.
    
    This class provides a simplified, high-performance interface to PyTorch's DataLoader
    with optimizations for GPU training including prefetch_factor, persistent_workers,
    and pin_memory support.

    Attributes:
        dataset (CellMapMultiDataset | CellMapDataset | CellMapSubset): The dataset to load.
        classes (Iterable[str]): The classes to load.
        batch_size (int): The batch size.
        num_workers (int): The number of workers to use.
        weighted_sampler (bool): Whether to use a weighted sampler.
        sampler (Union[MutableSubsetRandomSampler, Callable, None]): The sampler to use.
        is_train (bool): Whether the data is for training and thus should be shuffled.
        rng (Optional[torch.Generator]): The random number generator to use.
        loader (torch.utils.data.DataLoader): The underlying PyTorch DataLoader.
        default_kwargs (dict): The default arguments (maintained for compatibility).

    Methods:
        refresh: If the sampler is a Callable, refresh the DataLoader with the current sampler.
        collate_fn: Combine a list of dictionaries from different sources into a single dictionary for output.

    """

    def __init__(
        self,
        dataset: (
            CellMapMultiDataset | CellMapDataset | CellMapSubset | CellMapDatasetWriter
        ),
        classes: Sequence[str] | None = None,
        batch_size: int = 1,
        num_workers: int = 0,
        weighted_sampler: bool = False,
        sampler: Union[MutableSubsetRandomSampler, Callable, None] = None,
        is_train: bool = True,
        rng: Optional[torch.Generator] = None,
        device: Optional[str | torch.device] = None,
        iterations_per_epoch: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize the CellMapDataLoader with optimized PyTorch DataLoader backend.

        Args:
            dataset (CellMapMultiDataset | CellMapDataset | CellMapSubset): The dataset to load.
            classes (Iterable[str]): The classes to load.
            batch_size (int): The batch size.
            num_workers (int): The number of workers to use.
            weighted_sampler (bool): Whether to use a weighted sampler. Defaults to False.
            sampler (Union[MutableSubsetRandomSampler, Callable, None]): The sampler to use.
            is_train (bool): Whether the data is for training and thus should be shuffled.
            rng (Optional[torch.Generator]): The random number generator to use.
            device (Optional[str | torch.device]): The device to use. Defaults to "cuda" or "mps" if available, else "cpu".
            iterations_per_epoch (Optional[int]): Number of iterations per epoch, only necessary when a subset is used with a weighted sampler (i.e. if total samples in the dataset are > 2^24).
            `**kwargs`: Additional PyTorch DataLoader arguments (pin_memory, drop_last, persistent_workers, prefetch_factor, etc.).

        """
        self.dataset = dataset
        self.classes = classes if classes is not None else dataset.classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.weighted_sampler = weighted_sampler
        self.sampler = sampler
        self.is_train = is_train
        self.rng = rng
        
        # Set device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.iterations_per_epoch = iterations_per_epoch

        # Extract DataLoader parameters with optimized defaults
        # pin_memory only works with CUDA, so default to True only when CUDA is available
        # and device is CUDA
        pin_memory_default = torch.cuda.is_available() and str(device).startswith("cuda")
        self._pin_memory = kwargs.pop("pin_memory", pin_memory_default)
        
        # Validate pin_memory setting
        if self._pin_memory and not str(device).startswith("cuda"):
            logger.warning(
                f"pin_memory=True is only supported with CUDA devices. "
                f"Setting pin_memory=False for device={device}"
            )
            self._pin_memory = False
        
        self._persistent_workers = kwargs.pop("persistent_workers", num_workers > 0)
        self._drop_last = kwargs.pop("drop_last", False)
        
        # Set prefetch_factor for better GPU utilization (default 2, increase for GPU training)
        # Only applicable when num_workers > 0
        if num_workers > 0:
            prefetch_factor = kwargs.pop("prefetch_factor", 2)
            # Validate prefetch_factor
            if not isinstance(prefetch_factor, int) or prefetch_factor < 1:
                raise ValueError(
                    f"prefetch_factor must be a positive integer (>= 1), "
                    f"got {prefetch_factor!r} of type {type(prefetch_factor).__name__}"
                )
            self._prefetch_factor = prefetch_factor
        else:
            # Remove prefetch_factor from kwargs if present (not used with num_workers=0)
            kwargs.pop("prefetch_factor", None)
            self._prefetch_factor = None

        # Move dataset to device if not using multiprocessing
        if num_workers == 0:
            self.dataset.to(device, non_blocking=True)

        # Setup sampler
        if self.sampler is None:
            if iterations_per_epoch is not None or (
                weighted_sampler and len(self.dataset) > 2**24
            ):
                assert (
                    iterations_per_epoch is not None
                ), "If the dataset has more than 2^24 samples, iterations_per_epoch must be specified to allow for subset selection. In between epochs, run `refresh()` to update the sampler."
                assert not isinstance(
                    self.dataset, CellMapDatasetWriter
                ), "CellMapDatasetWriter does not support random sampling."
                self.sampler = self.dataset.get_subset_random_sampler(
                    num_samples=iterations_per_epoch * batch_size,
                    weighted=weighted_sampler,
                    rng=self.rng,
                )
            elif weighted_sampler and isinstance(self.dataset, CellMapMultiDataset):
                self.sampler = self.dataset.get_weighted_sampler(
                    self.batch_size, self.rng
                )

        # Store all kwargs for compatibility
        self.default_kwargs = kwargs.copy()
        self.default_kwargs.update({
            "pin_memory": self._pin_memory,
            "persistent_workers": self._persistent_workers,
            "drop_last": self._drop_last,
        })
        if self._prefetch_factor is not None:
            self.default_kwargs["prefetch_factor"] = self._prefetch_factor

        # Initialize PyTorch DataLoader (will be created in refresh())
        self._pytorch_loader = None
        self.refresh()

        # For backward compatibility, expose loader attribute that iterates over self
        self.loader = self

    def __getitem__(self, indices: Union[int, Sequence[int]]) -> dict:
        """Get an item from the DataLoader."""
        if isinstance(indices, int):
            indices = [indices]
        return self.collate_fn([self.dataset[index] for index in indices])

    def __iter__(self):
        """Create an iterator over the dataset using PyTorch DataLoader."""
        return iter(self._pytorch_loader)

    def __len__(self) -> int:
        """Return the number of batches per epoch."""
        return len(self._pytorch_loader)

    def to(self, device: str | torch.device, non_blocking: bool = True):
        """Move the dataset to the specified device."""
        self.dataset.to(device, non_blocking=non_blocking)
        self.device = device
        # Recreate DataLoader for new device
        self.refresh()

    def refresh(self):
        """Refresh the DataLoader (recreate with current sampler state)."""
        if isinstance(self.sampler, MutableSubsetRandomSampler):
            self.sampler.refresh()

        # Determine sampler for PyTorch DataLoader
        dataloader_sampler = None
        shuffle = False
        
        if self.sampler is not None:
            if isinstance(self.sampler, MutableSubsetRandomSampler):
                dataloader_sampler = self.sampler
            elif callable(self.sampler):
                dataloader_sampler = self.sampler()
            else:
                dataloader_sampler = self.sampler
        else:
            # Use shuffle if training and no custom sampler
            shuffle = self.is_train

        # Create optimized PyTorch DataLoader
        dataloader_kwargs = {
            "batch_size": self.batch_size,
            "shuffle": shuffle if dataloader_sampler is None else False,
            "num_workers": self.num_workers,
            "collate_fn": self.collate_fn,
            "pin_memory": self._pin_memory,
            "drop_last": self._drop_last,
            "generator": self.rng,
        }
        
        # Add sampler if provided
        if dataloader_sampler is not None:
            dataloader_kwargs["sampler"] = dataloader_sampler
            
        # Add persistent_workers only if num_workers > 0
        if self.num_workers > 0:
            dataloader_kwargs["persistent_workers"] = self._persistent_workers
            if self._prefetch_factor is not None:
                dataloader_kwargs["prefetch_factor"] = self._prefetch_factor
        
        # Add any additional kwargs
        for key, value in self.default_kwargs.items():
            if key not in dataloader_kwargs:
                dataloader_kwargs[key] = value

        self._pytorch_loader = torch.utils.data.DataLoader(
            self.dataset,
            **dataloader_kwargs
        )

    def collate_fn(self, batch: Sequence) -> dict[str, torch.Tensor]:
        """
        Combine a list of dictionaries from different sources into a single dictionary for output.
        Simplified collate function that relies on PyTorch's optimized GPU transfer via pin_memory.
        """
        outputs = {}
        for b in batch:
            for key, value in b.items():
                if key not in outputs:
                    outputs[key] = []
                outputs[key].append(value)

        # Stack tensors and move to device
        for key, value in outputs.items():
            if key != "__metadata__":
                outputs[key] = torch.stack(value).to(self.device, non_blocking=True)

        return outputs
