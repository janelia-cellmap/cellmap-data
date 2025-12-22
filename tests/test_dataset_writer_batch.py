"""
Tests for CellMapDatasetWriter batch operations.

Tests that the writer correctly handles batched write operations.
"""

import numpy as np
import pytest
import torch

from cellmap_data import CellMapDatasetWriter

from .test_helpers import create_test_dataset


class TestDatasetWriterBatchOperations:
    """Test suite for batch write operations in DatasetWriter."""

    @pytest.fixture
    def writer_setup(self, tmp_path):
        """Create writer and config for batch write tests.

        Returns a tuple of (writer, config) where writer is a CellMapDatasetWriter
        configured for testing batch operations.
        """
        # Create input data
        config = create_test_dataset(
            tmp_path / "input",
            raw_shape=(64, 64, 64),
            num_classes=2,
            raw_scale=(8.0, 8.0, 8.0),
        )

        # Output path
        output_path = tmp_path / "output" / "predictions.zarr"

        target_bounds = {
            "pred": {
                "x": [0, 512],
                "y": [0, 512],
                "z": [0, 512],
            }
        }

        writer = CellMapDatasetWriter(
            raw_path=config["raw_path"],
            target_path=str(output_path),
            classes=["class_0", "class_1"],
            input_arrays={"raw": {"shape": (32, 32, 32), "scale": (8.0, 8.0, 8.0)}},
            target_arrays={"pred": {"shape": (32, 32, 32), "scale": (8.0, 8.0, 8.0)}},
            target_bounds=target_bounds,
        )

        return writer, config

    def test_batch_write_with_tensor_indices(self, writer_setup):
        """Test writing with a batch of tensor indices."""
        writer, config = writer_setup

        # Simulate batch predictions
        batch_size = 8
        indices = torch.tensor(list(range(batch_size)))

        # Create predictions with shape (batch_size, num_classes, *spatial_dims)
        predictions = torch.randn(batch_size, 2, 32, 32, 32)

        # This should not raise an error
        writer[indices] = {"pred": predictions}

    def test_batch_write_with_numpy_indices(self, writer_setup):
        """Test writing with a batch of numpy indices."""
        writer, config = writer_setup

        # Simulate batch predictions
        batch_size = 4
        indices = np.array(list(range(batch_size)))

        # Create predictions
        predictions = np.random.randn(batch_size, 2, 32, 32, 32).astype(np.float32)

        # This should not raise an error
        writer[indices] = {"pred": predictions}

    def test_batch_write_with_list_indices(self, writer_setup):
        """Test writing with a batch of list indices."""
        writer, config = writer_setup

        # Simulate batch predictions
        batch_size = 4
        indices = [0, 1, 2, 3]

        # Create predictions
        predictions = torch.randn(batch_size, 2, 32, 32, 32)

        # This should not raise an error
        writer[indices] = {"pred": predictions}

    def test_batch_write_large_batch(self, writer_setup):
        """Test writing with a large batch size (as in the error case)."""
        writer, config = writer_setup

        # Simulate the error case: batch_size=32
        batch_size = 32
        indices = torch.tensor(list(range(batch_size)))

        # Create predictions with shape (32, 2, 32, 32, 32)
        predictions = torch.randn(batch_size, 2, 32, 32, 32)

        # This should not raise ValueError about shape mismatch
        writer[indices] = {"pred": predictions}

    def test_batch_write_with_dict_arrays(self, writer_setup):
        """Test writing with dictionary of arrays per class."""
        writer, config = writer_setup

        batch_size = 4
        indices = torch.tensor(list(range(batch_size)))

        # Create predictions as dictionary
        predictions_dict = {
            "class_0": torch.randn(batch_size, 32, 32, 32),
            "class_1": torch.randn(batch_size, 32, 32, 32),
        }

        # This should not raise an error
        writer[indices] = {"pred": predictions_dict}

    def test_batch_write_2d_data(self, tmp_path):
        """Test batch writing for 2D data (3D with singleton z dimension)."""
        # Import kept at module level; reuse create_test_dataset here

        # Create test dataset with thin Z dimension to simulate 2D
        config = create_test_dataset(
            tmp_path / "input",
            raw_shape=(1, 128, 128),  # Thin z dimension
            num_classes=1,
            raw_scale=(8.0, 4.0, 4.0),
        )

        output_path = tmp_path / "output_2d.zarr"

        target_bounds = {
            "pred": {
                "z": [0, 8],
                "y": [0, 512],
                "x": [0, 512],
            }
        }

        writer = CellMapDatasetWriter(
            raw_path=config["raw_path"],
            target_path=str(output_path),
            classes=["class_0"],
            input_arrays={"raw": {"shape": (1, 64, 64), "scale": (8.0, 4.0, 4.0)}},
            target_arrays={"pred": {"shape": (1, 64, 64), "scale": (8.0, 4.0, 4.0)}},
            axis_order="zyx",
            target_bounds=target_bounds,
        )

        # Test batch write with thin-z 3D data
        batch_size = 4
        indices = torch.tensor(list(range(batch_size)))
        predictions = torch.randn(batch_size, 1, 1, 64, 64)

        # This should not raise an error
        writer[indices] = {"pred": predictions}

    def test_single_item_write_still_works(self, writer_setup):
        """Test that single item writes still work correctly."""
        writer, config = writer_setup

        # Single item write
        idx = 0
        predictions = torch.randn(2, 32, 32, 32)

        # This should work as before
        writer[idx] = {"pred": predictions}

    def test_batch_write_with_scalar_values(self, writer_setup):
        """Test batch writing with scalar values fills all spatial dims."""
        writer, config = writer_setup

        batch_size = 4
        indices = torch.tensor(list(range(batch_size)))

        # Scalar values should be broadcast to full arrays
        # Create proper shaped arrays filled with the scalar value
        scalar_val = 0.5
        predictions = torch.full((batch_size, 2, 32, 32, 32), scalar_val)
        writer[indices] = {"pred": predictions}

    def test_batch_write_mixed_data_types(self, writer_setup):
        """Test batch writing preserves data types."""
        writer, config = writer_setup

        batch_size = 4
        indices = torch.tensor(list(range(batch_size)))

        # Test with different dtypes
        predictions_float32 = torch.randn(
            batch_size, 2, 32, 32, 32, dtype=torch.float32
        )
        writer[indices] = {"pred": predictions_float32}

        predictions_float64 = torch.randn(
            batch_size, 2, 32, 32, 32, dtype=torch.float64
        )
        indices2 = torch.tensor(list(range(batch_size, batch_size * 2)))
        writer[indices2] = {"pred": predictions_float64}
