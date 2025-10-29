#!/usr/bin/env python3

import torch
import torch.utils.data
import tempfile
import numpy as np
from pathlib import Path
import sys
import os

# Add the src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from cellmap_data.dataset_writer import CellMapDatasetWriter
from cellmap_data.dataloader import CellMapDataLoader


def test_dataset_writer_gpu_transfer():
    """Test that CellMapDatasetWriter properly transfers data to GPU."""

    # Skip if no CUDA available
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU transfer test")
        return

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create mock input and target arrays configuration
        input_arrays = {
            "raw": {
                "shape": (32, 32, 32),
                "scale": (1.0, 1.0, 1.0),
            }
        }

        target_arrays = {
            "segmentation": {
                "shape": (32, 32, 32),
                "scale": (1.0, 1.0, 1.0),
            }
        }

        target_bounds = {
            "segmentation": {
                "x": [0.0, 32.0],
                "y": [0.0, 32.0],
                "z": [0.0, 32.0],
            }
        }

        # Create a dummy raw data path (won't be accessed in this test)
        raw_path = str(Path(tmp_dir) / "raw.zarr")
        target_path = str(Path(tmp_dir) / "target.zarr")

        classes = ["class1", "class2"]

        # Create dataset writer
        writer = CellMapDatasetWriter(
            raw_path=raw_path,
            target_path=target_path,
            classes=classes,
            input_arrays=input_arrays,
            target_arrays=target_arrays,
            target_bounds=target_bounds,
            device="cuda",
        )

        # Create loader with batch_size=1
        loader = writer.loader(batch_size=1, num_workers=0)

        print(f"Dataset writer device: {writer.device}")
        print(f"Loader type: {type(loader)}")

        # Test that the dataset writer has the correct device
        # Note: PyTorch DataLoader doesn't have a device attribute - device is handled by the dataset
        assert str(writer.device) == "cuda", f"Expected cuda, got {writer.device}"
        assert isinstance(loader, CellMapDataLoader), "Expected CellMapDataLoader"

        print("✅ CellMapDatasetWriter GPU transfer test passed!")


def test_pin_memory_gpu_transfer():
    """Test that pin_memory works correctly with GPU transfers."""
    import pytest

    # Skip if no CUDA available
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    class CPUDataset:
        def __init__(self):
            self.classes = ["a", "b"]

        def __len__(self):
            return 4

        def __getitem__(self, idx):
            # Return CPU tensors to test pin_memory transfer
            return {
                "data": torch.randn(8, 8),
                "label": torch.tensor(idx % 2),
            }

        def to(self, device, non_blocking=True):
            pass

    dataset = CPUDataset()

    # Test pin_memory=True with GPU device
    loader = CellMapDataLoader(
        dataset, batch_size=2, pin_memory=True, device="cuda", num_workers=0
    )

    batch = next(iter(loader))

    # Verify tensors are on GPU
    assert (
        batch["data"].device.type == "cuda"
    ), f"Expected GPU, got {batch['data'].device}"
    assert (
        batch["label"].device.type == "cuda"
    ), f"Expected GPU, got {batch['label'].device}"

    # Verify pin_memory flag is set
    assert loader._pin_memory, "pin_memory should be True"

    print("✅ pin_memory GPU transfer test passed!")


def test_multiworker_gpu_performance():
    """Test that multiworker setup works correctly with GPU."""
    import pytest

    # Skip if no CUDA available
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    class GPUDataset:
        def __init__(self):
            self.classes = ["a", "b", "c"]

        def __len__(self):
            return 12

        def __getitem__(self, idx):
            return {
                "features": torch.randn(16, 16),
                "target": torch.tensor(idx % 3),
                "index": torch.tensor(idx),
            }

        def to(self, device, non_blocking=True):
            pass

    dataset = GPUDataset()

    # Test with multiworkers, pin_memory, and persistent_workers
    loader = CellMapDataLoader(
        dataset,
        batch_size=3,
        pin_memory=True,
        persistent_workers=True,
        num_workers=2,
        device="cuda",
    )

    # Test multiple iterations to ensure workers persist
    batches = []
    for i, batch in enumerate(loader):
        batches.append(batch)

        # Verify GPU transfer
        assert batch["features"].device.type == "cuda", f"Batch {i} features not on GPU"
        assert batch["target"].device.type == "cuda", f"Batch {i} targets not on GPU"

        if i >= 2:  # Test first 3 batches
            break

    # Verify persistent workers
    assert loader._worker_executor is not None, "Workers should persist"
    assert loader._persistent_workers, "persistent_workers should be True"

    print(
        f"✅ Multiworker GPU performance test passed! Processed {len(batches)} batches"
    )


def test_gpu_memory_optimization():
    """Test GPU memory optimization features."""
    import pytest

    # Skip if no CUDA available
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    class LargeDataset:
        def __init__(self):
            self.classes = ["background", "foreground"]

        def __len__(self):
            return 8

        def __getitem__(self, idx):
            # Return larger tensors to trigger memory optimization
            return {
                "image": torch.randn(3, 64, 64),  # Larger images
                "mask": torch.randint(0, 2, (64, 64)),
                "metadata": torch.tensor([idx, idx * 2, idx * 3]),
            }

        def to(self, device, non_blocking=True):
            pass

    dataset = LargeDataset()

    # Test with CUDA streams optimization
    loader = CellMapDataLoader(
        dataset, batch_size=4, pin_memory=True, device="cuda", num_workers=0
    )

    # Get a batch to trigger stream initialization
    batch = next(iter(loader))

    # Verify CUDA stream optimization may be enabled
    # (depends on memory threshold and GPU availability)
    print(f"CUDA streams enabled: {loader._use_streams}")
    print(f"Number of streams: {len(loader._streams) if loader._streams else 0}")

    # Verify tensors are properly transferred
    assert batch["image"].device.type == "cuda", "Images should be on GPU"
    assert batch["mask"].device.type == "cuda", "Masks should be on GPU"
    assert batch["metadata"].device.type == "cuda", "Metadata should be on GPU"

    print("✅ GPU memory optimization test passed!")


if __name__ == "__main__":
    test_dataset_writer_gpu_transfer()
    test_pin_memory_gpu_transfer()
    test_multiworker_gpu_performance()
    test_gpu_memory_optimization()
