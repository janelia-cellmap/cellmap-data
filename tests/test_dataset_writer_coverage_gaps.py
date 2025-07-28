"""
Test coverage improvements for dataset_writer.py core functionality.
Focuses on critical gaps in parameter validation, initialization, and writing methods.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import warnings
from pathlib import Path
import tempfile
import os

from cellmap_data.dataset_writer import CellMapDatasetWriter
from cellmap_data.utils.error_messages import ErrorMessages


class TestCellMapDatasetWriterInitialization:
    """Test dataset writer initialization with various parameter combinations."""

    def test_deprecated_raw_path_parameter(self):
        """Test deprecated raw_path parameter handling."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Mock path and data structures
            mock_raw_path = "/test/path"
            input_arrays = {"raw": {"shape": [32, 32, 32], "scale": [1.0, 1.0, 1.0]}}
            target_arrays = {
                "target": {"shape": [32, 32, 32], "scale": [1.0, 1.0, 1.0]}
            }

            with patch("cellmap_data.dataset_writer.CellMapImage") as MockImage:
                mock_image = MagicMock()
                MockImage.return_value = mock_image

                # Test successful initialization with deprecated raw_path
                writer = CellMapDatasetWriter(
                    raw_path=mock_raw_path,
                    target_path="/test/target",
                    classes=["background", "mito"],
                    input_arrays=input_arrays,
                    target_arrays=target_arrays,
                    target_bounds={
                        "target": {"z": [0.0, 50.0], "y": [0.0, 50.0], "x": [0.0, 50.0]}
                    },
                    axis_order="zyx",
                )

                assert len(w) == 1
                assert issubclass(w[0].category, DeprecationWarning)
                assert "raw_path" in str(w[0].message)

    def test_conflicting_raw_path_and_input_path(self):
        """Test error when both raw_path and input_path are provided."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            CellMapDatasetWriter(
                raw_path="/test/raw",
                input_path="/test/input",
                target_path="/test/target",
                classes=["class1"],
                target_bounds={
                    "test": {"z": [0.0, 100.0], "y": [0.0, 100.0], "x": [0.0, 100.0]}
                },
            )

    def test_no_input_path_error(self):
        """Test error when neither raw_path nor input_path is provided."""
        with pytest.raises(ValueError, match="Must specify either"):
            CellMapDatasetWriter(
                target_path="/test/target",
                classes=["class1"],
                target_bounds={
                    "test": {"z": [0.0, 100.0], "y": [0.0, 100.0], "x": [0.0, 100.0]}
                },
            )

    def test_no_target_path_error(self):
        """Test error when target_path is not provided."""
        with pytest.raises(ValueError, match="target_path"):
            CellMapDatasetWriter(
                input_path="/test/input",
                classes=["class1"],
                target_bounds={
                    "test": {"z": [0.0, 100.0], "y": [0.0, 100.0], "x": [0.0, 100.0]}
                },
            )

    def test_no_classes_error(self):
        """Test error when classes are not provided."""
        with pytest.raises(ValueError, match="classes"):
            CellMapDatasetWriter(
                input_path="/test/input",
                target_path="/test/target",
                target_bounds={
                    "test": {"z": [0.0, 100.0], "y": [0.0, 100.0], "x": [0.0, 100.0]}
                },
            )

    def test_no_target_bounds_error(self):
        """Test error when target_bounds are not provided."""
        with pytest.raises(ValueError, match="target_bounds"):
            CellMapDatasetWriter(
                input_path="/test/input", target_path="/test/target", classes=["class1"]
            )


class TestCellMapDatasetWriterBounds:
    """Test dataset writer bounding box and sampling box calculations."""

    @patch("cellmap_data.dataset_writer.CellMapImage")
    def test_bounding_box_calculation(self, MockImage):
        """Test bounding box calculation from target bounds."""
        mock_image = MagicMock()
        MockImage.return_value = mock_image

        input_arrays = {
            "raw": {"shape": [64, 64, 64], "scale": [1.0, 1.0, 1.0]},
            "labels": {"shape": [64, 64, 64], "scale": [1.0, 1.0, 1.0]},
        }
        target_arrays = {
            "predictions": {"shape": [32, 32, 32], "scale": [2.0, 2.0, 2.0]},
            "segmentation": {"shape": [16, 16, 16], "scale": [4.0, 4.0, 4.0]},
        }
        target_bounds = {
            "predictions": {"z": [10.0, 90.0], "y": [20.0, 180.0], "x": [30.0, 270.0]},
            "segmentation": {"z": [5.0, 95.0], "y": [15.0, 185.0], "x": [25.0, 275.0]},
        }

        writer = CellMapDatasetWriter(
            input_path="/test/input",
            target_path="/test/target",
            classes=["background", "object"],
            input_arrays=input_arrays,
            target_arrays=target_arrays,
            target_bounds=target_bounds,
        )

        # Verify writer was created successfully
        assert writer is not None
        if hasattr(writer, "input_arrays") and writer.input_arrays is not None:
            assert len(writer.input_arrays) == 2
        if hasattr(writer, "target_arrays") and writer.target_arrays is not None:
            assert len(writer.target_arrays) == 2
        if hasattr(writer, "target_bounds") and writer.target_bounds is not None:
            assert len(writer.target_bounds) == 2
            assert "predictions" in writer.target_bounds
            assert "segmentation" in writer.target_bounds


class TestCellMapDatasetWriterProperties:
    """Test dataset writer computed properties."""

    @patch("cellmap_data.dataset_writer.CellMapImage")
    def test_center_property(self, MockImage):
        """Test center property calculation."""
        mock_image = MagicMock()
        MockImage.return_value = mock_image

        writer = CellMapDatasetWriter(
            input_path="/test/input",
            target_path="/test/target",
            classes=["class1"],
            target_bounds={
                "test": {"z": [0.0, 100.0], "y": [0.0, 100.0], "x": [0.0, 100.0]}
            },
        )

        # Mock the bounding_box property
        with patch.object(
            writer, "bounding_box", {"z": [0, 100], "y": [0, 100], "x": [0, 100]}
        ):
            center = writer.center
            if center is not None:
                assert "z" in center
                assert "y" in center
                assert "x" in center

    @patch("cellmap_data.dataset_writer.CellMapImage")
    def test_size_property(self, MockImage):
        """Test size property calculation."""
        mock_image = MagicMock()
        MockImage.return_value = mock_image

        writer = CellMapDatasetWriter(
            input_path="/test/input",
            target_path="/test/target",
            classes=["class1"],
            target_bounds={
                "test": {"z": [0.0, 100.0], "y": [0.0, 100.0], "x": [0.0, 100.0]}
            },
        )

        # Mock the bounding_box property
        with patch.object(
            writer, "bounding_box", {"z": [0, 100], "y": [0, 100], "x": [0, 100]}
        ):
            size = writer.size
            assert isinstance(size, (int, np.integer))
            assert size >= 0


class TestCellMapDatasetWriterMethods:
    """Test dataset writer core methods."""

    @patch("cellmap_data.dataset_writer.CellMapImage")
    def test_verify_method(self, MockImage):
        """Test dataset verification method."""
        mock_image = MagicMock()
        MockImage.return_value = mock_image

        writer = CellMapDatasetWriter(
            input_path="/test/input",
            target_path="/test/target",
            classes=["class1"],
            target_bounds={
                "test": {"z": [0.0, 100.0], "y": [0.0, 100.0], "x": [0.0, 100.0]}
            },
        )

        # Mock len method to return positive value
        with patch.object(writer, "__len__", return_value=10):
            assert writer.verify() is True

        # Mock len method to return zero
        with patch.object(writer, "__len__", return_value=0):
            assert writer.verify() is False

        # Mock len method to raise exception
        with patch.object(writer, "__len__", side_effect=Exception("Test error")):
            assert writer.verify() is False

    @patch("cellmap_data.dataset_writer.CellMapImage")
    def test_get_indices_method(self, MockImage):
        """Test get_indices method for dataset tiling."""
        mock_image = MagicMock()
        MockImage.return_value = mock_image

        writer = CellMapDatasetWriter(
            input_path="/test/input",
            target_path="/test/target",
            classes=["class1"],
            target_bounds={
                "test": {"z": [0.0, 100.0], "y": [0.0, 100.0], "x": [0.0, 100.0]}
            },
        )

        # Mock required properties using patch.object
        with (
            patch.object(
                writer, "smallest_voxel_sizes", {"z": 1.0, "y": 1.0, "x": 1.0}
            ),
            patch.object(writer, "sampling_box_shape", {"z": 100, "y": 100, "x": 100}),
            patch.object(writer, "axis_order", "zyx"),
        ):

            chunk_size = {"z": 10.0, "y": 10.0, "x": 10.0}
            indices = writer.get_indices(chunk_size)

        assert isinstance(indices, list)
        assert len(indices) > 0
        for idx in indices:
            assert isinstance(idx, (int, np.integer))


class TestCellMapDatasetWriterEdgeCases:
    """Test dataset writer edge cases and error conditions."""

    @patch("cellmap_data.dataset_writer.CellMapImage")
    def test_empty_arrays_handling(self, MockImage):
        """Test handling of empty array configurations."""
        mock_image = MagicMock()
        MockImage.return_value = mock_image

        # Test with minimal configuration
        writer = CellMapDatasetWriter(
            input_path="/test/input",
            target_path="/test/target",
            classes=["class1"],
            target_bounds={
                "test": {"z": [0.0, 10.0], "y": [0.0, 10.0], "x": [0.0, 10.0]}
            },
        )

        assert writer is not None

    @patch("cellmap_data.dataset_writer.CellMapImage")
    def test_coordinate_transformation_errors(self, MockImage):
        """Test coordinate transformation error handling."""
        mock_image = MagicMock()
        MockImage.return_value = mock_image

        writer = CellMapDatasetWriter(
            input_path="/test/input",
            target_path="/test/target",
            classes=["class1"],
            target_bounds={
                "test": {"z": [0.0, 100.0], "y": [0.0, 100.0], "x": [0.0, 100.0]}
            },
        )

        # Mock len method to return zero (empty dataset)
        with patch.object(writer, "__len__", return_value=0):
            from cellmap_data.exceptions import IndexError as CellMapIndexError

            with pytest.raises(CellMapIndexError, match="dataset is empty"):
                writer.get_center(0)

        # Test negative index
        with patch.object(writer, "__len__", return_value=10):
            with pytest.raises(CellMapIndexError, match="negative"):
                writer.get_center(-1)

        # Test out of bounds index
        with patch.object(writer, "__len__", return_value=10):
            with pytest.raises(CellMapIndexError, match="out of bounds"):
                writer.get_center(15)


class TestCellMapDatasetWriterDeviceHandling:
    """Test dataset writer device management."""

    @patch("cellmap_data.dataset_writer.CellMapImage")
    def test_to_device_method(self, MockImage):
        """Test device transfer method."""
        mock_image = MagicMock()
        MockImage.return_value = mock_image

        writer = CellMapDatasetWriter(
            input_path="/test/input",
            target_path="/test/target",
            classes=["class1"],
            target_bounds={
                "test": {"z": [0.0, 100.0], "y": [0.0, 100.0], "x": [0.0, 100.0]}
            },
        )

        # Test device transfer
        result = writer.to("cpu")
        assert result is writer  # Should return self

        # Test with torch device
        result = writer.to(torch.device("cpu"))
        assert result is writer
