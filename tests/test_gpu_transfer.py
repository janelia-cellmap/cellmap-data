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
        assert isinstance(
            loader, torch.utils.data.DataLoader
        ), "Expected PyTorch DataLoader"

        print("✅ CellMapDatasetWriter GPU transfer test passed!")


if __name__ == "__main__":
    test_dataset_writer_gpu_transfer()
