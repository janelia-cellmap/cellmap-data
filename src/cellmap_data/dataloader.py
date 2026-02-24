import os
import platform
import logging
from typing import Callable, Optional, Sequence, Union

import torch
import torch.utils.data

from .dataset import CellMapDataset
from .dataset_writer import CellMapDatasetWriter
from .image import CellMapImage
from .multidataset import CellMapMultiDataset
from .mutable_sampler import MutableSubsetRandomSampler
from .subdataset import CellMapSubset

logger = logging.getLogger(__name__)


def _set_tensorstore_context(dataset, context) -> None:
    """
    Recursively set a TensorStore context on every CellMapImage in the dataset tree.

    This must be called before workers are spawned so the bounded cache_pool
    limit is picked up by every worker process (via fork inheritance on Linux,
    or via pickle on Windows/macOS spawn).

    If an image's TensorStore array has already been opened (``_array`` cached),
    the new context cannot affect that array; a warning is emitted.
    """
    if isinstance(dataset, CellMapMultiDataset):
        for ds in dataset.datasets:
            _set_tensorstore_context(ds, context)
    elif isinstance(dataset, CellMapSubset):
        _set_tensorstore_context(dataset.dataset, context)
    elif isinstance(dataset, CellMapDataset):
        dataset.context = context
        all_sources = list(dataset.input_sources.values()) + list(
            dataset.target_sources.values()
        )
        for source in all_sources:
            if isinstance(source, CellMapImage):
                _apply_context_to_image(source, context)
            elif isinstance(source, dict):
                for sub_source in source.values():
                    if isinstance(sub_source, CellMapImage):
                        _apply_context_to_image(sub_source, context)
    else:
        logger.warning(
            "Unsupported dataset type %s in _set_tensorstore_context; "
            "TensorStore context was not applied.",
            type(dataset).__name__,
        )


def _apply_context_to_image(image: "CellMapImage", context) -> None:
    """Set the TensorStore context on a single CellMapImage, warning if already opened."""
    if "_array" in getattr(image, "__dict__", {}):
        logger.warning(
            "TensorStore array already opened for %s; "
            "cache_pool limit will not apply to this image.",
            getattr(image, "path", image),
        )
    image.context = context


class CellMapDataLoader:
    """
    Optimized DataLoader wrapper for CellMapDataset that uses PyTorch's native DataLoader.

    This class provides a simplified, high-performance interface to PyTorch's DataLoader
    with optimizations for GPU training including prefetch_factor, persistent_workers,
    and pin_memory support.

    Attributes
    ----------
        dataset (CellMapMultiDataset | CellMapDataset | CellMapSubset): Dataset to load.
        classes (Iterable[str]): Classes to load.
        batch_size (int): Batch size.
        num_workers (int): Number of workers.
        weighted_sampler (bool): Whether to use a weighted sampler.
        sampler (Union[MutableSubsetRandomSampler, Callable, None]): Sampler to use.
        is_train (bool): Whether data is for training (shuffled).
        rng (Optional[torch.Generator]): Random number generator.
        loader (torch.utils.data.DataLoader): Underlying PyTorch DataLoader.
        default_kwargs (dict): Default arguments for compatibility.
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
        tensorstore_cache_bytes: Optional[int] = None,
        **kwargs,
    ):
        """
        Initializes the CellMapDataLoader with an optimized PyTorch DataLoader backend.

        Args:
        ----
            dataset: The dataset to load.
            classes: The classes to load.
            batch_size: The batch size.
            num_workers: The number of workers.
            weighted_sampler: Whether to use a weighted sampler.
            sampler: The sampler to use.
            is_train: Whether the data is for training (shuffled).
            rng: The random number generator.
            device: The device to use ("cuda", "mps", or "cpu").
            iterations_per_epoch: Iterations per epoch for large datasets.
            tensorstore_cache_bytes: Total TensorStore chunk-cache budget in bytes
                shared across all worker processes.  The budget is split evenly:
                ``per_worker = tensorstore_cache_bytes // max(1, num_workers)``.
                Defaults to the ``CELLMAP_TENSORSTORE_CACHE_BYTES`` environment
                variable if set, otherwise no limit is applied (TensorStore's
                default unbounded cache).  Set to ``0`` to disable caching
                entirely.  Bounding this value prevents persistent worker
                processes from accumulating chunk data unboundedly across epochs.
            **kwargs: Additional PyTorch DataLoader arguments.
        """
        self.dataset = dataset
        self.classes = classes if classes is not None else dataset.classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.weighted_sampler = weighted_sampler
        self.sampler = sampler
        self.is_train = is_train
        self.rng = rng

        if platform.system() == "Windows" and num_workers > 0:
            logger.warning(
                "CellMapDataLoader: num_workers=%d on Windows. "
                "The dataset uses a synchronous (single-thread) executor internally "
                "so TensorStore reads are never dispatched to ThreadPoolExecutor "
                "worker threads. If crashes persist, try num_workers=0.",
                num_workers,
            )

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

        # Bound TensorStore chunk-cache to prevent unbounded RAM growth in
        # persistent worker processes (Linux fork, Windows/macOS spawn).
        # Resolve from parameter, then env var, then leave unconfigured.
        if tensorstore_cache_bytes is None:
            _env = os.environ.get("CELLMAP_TENSORSTORE_CACHE_BYTES")
            if _env is not None:
                try:
                    tensorstore_cache_bytes = int(_env)
                except ValueError as exc:
                    raise ValueError(
                        "Invalid value for environment variable "
                        "CELLMAP_TENSORSTORE_CACHE_BYTES: "
                        f"{_env!r}. Expected an integer number of bytes."
                    ) from exc
        if tensorstore_cache_bytes is not None and tensorstore_cache_bytes < 0:
            raise ValueError(
                f"tensorstore_cache_bytes must be >= 0 when set; got {tensorstore_cache_bytes}"
            )
        self.tensorstore_cache_bytes = tensorstore_cache_bytes

        if tensorstore_cache_bytes is not None and not isinstance(
            dataset, CellMapDatasetWriter
        ):
            import tensorstore as ts

            effective_workers = max(1, num_workers)
            per_worker_bytes = tensorstore_cache_bytes // effective_workers
            bounded_ctx = ts.Context(
                {"cache_pool": {"total_bytes_limit": per_worker_bytes}}
            )
            _set_tensorstore_context(dataset, bounded_ctx)
            logger.info(
                "TensorStore cache bounded: total=%d bytes / %d worker(s) = %d bytes each",
                tensorstore_cache_bytes,
                effective_workers,
                per_worker_bytes,
            )

        # Extract DataLoader parameters with optimized defaults
        # pin_memory only works with CUDA, so default to True only when CUDA is available
        # and device is CUDA
        pin_memory_default = (
            torch.cuda.is_available()
            and str(device).startswith("cuda")
            and platform.system() != "Windows"
        )  # pin_memory has issues on Windows with CUDA
        self._pin_memory = kwargs.pop("pin_memory", pin_memory_default)

        # Validate pin_memory setting
        if self._pin_memory and not str(device).startswith("cuda"):
            logger.warning(
                "pin_memory=True is only supported with CUDA. Disabling for %s.",
                device,
            )
            self._pin_memory = False

        self._persistent_workers = kwargs.pop("persistent_workers", num_workers > 0)
        self._drop_last = kwargs.pop("drop_last", False)

        # Set prefetch_factor for better GPU utilization (default 2, increase for GPU training)
        # Only applicable when num_workers > 0
        if num_workers > 0:
            prefetch_factor = kwargs.pop("prefetch_factor", 2)
            if not isinstance(prefetch_factor, int) or prefetch_factor < 1:
                raise ValueError(
                    f"prefetch_factor must be a positive integer, got {prefetch_factor}"
                )
            self._prefetch_factor = prefetch_factor
        else:
            kwargs.pop("prefetch_factor", None)
            self._prefetch_factor = None

        # Setup sampler
        if self.sampler is None:
            if iterations_per_epoch is not None or (
                weighted_sampler and len(self.dataset) > 2**24
            ):
                if iterations_per_epoch is None:
                    raise ValueError(
                        "iterations_per_epoch must be specified for large datasets."
                    )
                if isinstance(self.dataset, CellMapDatasetWriter):
                    raise TypeError(
                        "CellMapDatasetWriter does not support random sampling."
                    )
                self.sampler = self.dataset.get_subset_random_sampler(
                    num_samples=iterations_per_epoch * batch_size,
                    weighted=weighted_sampler,
                    rng=self.rng,
                )
            elif weighted_sampler and isinstance(self.dataset, CellMapMultiDataset):
                self.sampler = self.dataset.get_weighted_sampler(
                    self.batch_size, self.rng
                )

        self.default_kwargs = kwargs
        self.default_kwargs.update(
            {
                "pin_memory": self._pin_memory,
                "persistent_workers": self._persistent_workers,
                "drop_last": self._drop_last,
            }
        )
        if self._prefetch_factor is not None:
            self.default_kwargs["prefetch_factor"] = self._prefetch_factor

        self._pytorch_loader = None
        self.refresh()

    @property
    def loader(self) -> torch.utils.data.DataLoader | None:
        """Return the DataLoader."""
        return self._pytorch_loader

    def __getitem__(self, indices: Union[int, Sequence[int]]) -> dict:
        """Get an item from the DataLoader."""
        if isinstance(indices, int):
            indices = [indices]
        return self.collate_fn([self.dataset[index] for index in indices])

    def __iter__(self):
        """Create an iterator over the dataset."""
        if self._pytorch_loader is None:
            self.refresh()
        return iter(self._pytorch_loader)

    def __len__(self) -> int | None:
        """Return the number of batches per epoch."""
        if self._pytorch_loader is None:
            return None
        return len(self._pytorch_loader)

    def to(self, device: str | torch.device, non_blocking: bool = True):
        """Move the dataset to the specified device."""
        self.dataset.to(device, non_blocking=non_blocking)
        self.device = device
        return self

    def refresh(self):
        """Refresh the DataLoader with the current sampler state."""
        if isinstance(self.sampler, MutableSubsetRandomSampler):
            self.sampler.refresh()

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
            shuffle = self.is_train

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

        dataloader_kwargs.pop("force_has_data", None)

        # Ensure that dataset is loaded onto CPU if pin_memory is used
        if self._pin_memory:
            self.dataset.to("cpu")

        self._pytorch_loader = torch.utils.data.DataLoader(
            self.dataset, **dataloader_kwargs
        )

    def collate_fn(self, batch: Sequence) -> dict[str, torch.Tensor]:
        """
        Collates a batch of samples into a single dictionary of tensors.
        """
        outputs = {}
        for b in batch:
            for key, value in b.items():
                if key not in outputs:
                    outputs[key] = []
                outputs[key].append(value)

        for key, value in outputs.items():
            if key != "__metadata__":
                outputs[key] = torch.stack(value)

        return outputs
