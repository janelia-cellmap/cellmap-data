"""
Test coverage improvements for dataset.py core functionality.
Focuses on critical gaps in parameter validation, initialization, and data access methods.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import warnings
from pathlib import Path
import tempfile
import os

from cellmap_data.dataset import CellMapDataset
from cellmap_data.image import CellMapImage
from cellmap_data.utils.error_messages import ErrorMessages


class TestCellMapDatasetInitialization:
    """Test dataset initialization with various parameter combinations."""

    def test_deprecated_raw_path_parameter(self):
        """Test deprecated raw_path parameter handling."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            dataset = CellMapDataset(
                raw_path="/test/path",  # deprecated parameter
                target_path="/test/target",
                input_arrays={
                    "test": {"shape": [100, 100, 100], "scale": [1.0, 1.0, 1.0]}
                },
                classes=["class1"],
            )

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "raw_path" in str(w[0].message)
            assert dataset.input_path == "/test/path"
            assert dataset.raw_path == "/test/path"

    def test_conflicting_raw_path_and_input_path(self):
        """Test error when both raw_path and input_path are provided."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            CellMapDataset(
                raw_path="/test/raw",
                input_path="/test/input",
                target_path="/test/target",
                input_arrays={
                    "test": {"shape": [100, 100, 100], "scale": [1.0, 1.0, 1.0]}
                },
                classes=["class1"],
            )

    def test_deprecated_class_relation_dict_parameter(self):
        """Test deprecated class_relation_dict parameter handling."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            dataset = CellMapDataset(
                input_path="/test/path",
                target_path="/test/target",
                input_arrays={
                    "test": {"shape": [100, 100, 100], "scale": [1.0, 1.0, 1.0]}
                },
                classes=["class1"],
                class_relation_dict={"test": ["relation"]},  # deprecated parameter
            )

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "class_relation_dict" in str(w[0].message)
            assert dataset.class_relationships == {"test": ["relation"]}

    def test_conflicting_class_relationships_parameters(self):
        """Test error when both class_relationships and class_relation_dict are provided."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            CellMapDataset(
                input_path="/test/path",
                target_path="/test/target",
                input_arrays={
                    "test": {"shape": [100, 100, 100], "scale": [1.0, 1.0, 1.0]}
                },
                classes=["class1"],
                class_relationships={"new": ["relation"]},
                class_relation_dict={"old": ["relation"]},
            )


class TestCellMapDatasetParameterValidation:
    """Test parameter validation in dataset initialization."""

    def test_missing_input_path(self):
        """Test error when input_path is missing."""
        with pytest.raises(ValueError, match="input_path.*required"):
            CellMapDataset(
                target_path="/test/target",
                input_arrays={
                    "test": {"shape": [100, 100, 100], "scale": [1.0, 1.0, 1.0]}
                },
                classes=["class1"],
            )

    def test_missing_target_path(self):
        """Test error when target_path is missing."""
        with pytest.raises(ValueError, match="target_path.*required"):
            CellMapDataset(
                input_path="/test/path",
                input_arrays={
                    "test": {"shape": [100, 100, 100], "scale": [1.0, 1.0, 1.0]}
                },
                classes=["class1"],
            )

    def test_missing_input_arrays(self):
        """Test error when input_arrays is missing."""
        with pytest.raises(ValueError, match="input_arrays.*required"):
            CellMapDataset(
                input_path="/test/path", target_path="/test/target", classes=["class1"]
            )

    def test_raw_only_mode_initialization(self):
        """Test initialization in raw-only mode (classes=None)."""
        dataset = CellMapDataset(
            input_path="/test/path",
            target_path="/test/target",
            input_arrays={"test": {"shape": [100, 100, 100], "scale": [1.0, 1.0, 1.0]}},
            # classes=None (default)
        )

        assert dataset.raw_only is True
        assert dataset.classes == []

    def test_device_parameter_handling(self):
        """Test device parameter conversion to torch.device."""
        dataset = CellMapDataset(
            input_path="/test/path",
            target_path="/test/target",
            input_arrays={"test": {"shape": [100, 100, 100], "scale": [1.0, 1.0, 1.0]}},
            classes=["class1"],
            device="cuda:0",
        )

        assert isinstance(dataset._device, torch.device)
        assert str(dataset._device) == "cuda:0"


class TestCellMapDatasetInternalMethods:
    """Test internal methods and property access."""

    @patch("cellmap_data.dataset.CellMapImage")
    def test_input_sources_creation(self, mock_cellmap_image):
        """Test creation of input sources from input_arrays."""
        mock_image = Mock(spec=CellMapImage)
        mock_cellmap_image.return_value = mock_image

        input_arrays = {
            "array1": {"shape": [100, 100, 100], "scale": [1.0, 1.0, 1.0]},
            "array2": {"shape": [50, 50, 50], "scale": [0.5, 0.5, 0.5]},
        }

        dataset = CellMapDataset(
            input_path="/test/path",
            target_path="/test/target",
            input_arrays=input_arrays,
            classes=["class1"],
        )

        assert len(dataset.input_sources) == 2
        assert "array1" in dataset.input_sources
        assert "array2" in dataset.input_sources
        assert mock_cellmap_image.call_count == 2

    def test_target_path_splitting(self):
        """Test target path splitting functionality."""
        dataset = CellMapDataset(
            input_path="/test/path",
            target_path="/test/target.zarr/labels",
            input_arrays={"test": {"shape": [100, 100, 100], "scale": [1.0, 1.0, 1.0]}},
            classes=["class1"],
        )

        # This tests the split_target_path function indirectly
        assert hasattr(dataset, "target_path_str")
        assert hasattr(dataset, "classes_with_path")

    def test_attribute_initialization(self):
        """Test all attributes are properly initialized."""
        spatial_transforms = [Mock()]
        raw_value_transforms = Mock()
        target_value_transforms = Mock()
        class_relationships = {"test": ["relationship"]}
        context = np.array([1, 2, 3])
        rng = torch.Generator()

        dataset = CellMapDataset(
            input_path="/test/path",
            target_path="/test/target",
            input_arrays={"test": {"shape": [100, 100, 100], "scale": [1.0, 1.0, 1.0]}},
            classes=["class1"],
            target_arrays={"target": {"shape": [50, 50, 50], "scale": [0.5, 0.5, 0.5]}},
            spatial_transforms=spatial_transforms,
            raw_value_transforms=raw_value_transforms,
            target_value_transforms=target_value_transforms,
            class_relationships=class_relationships,
            is_train=True,
            axis_order="zyx",
            context=context,
            rng=rng,
            force_has_data=True,
            empty_value=-1,
            pad=True,
        )

        # Verify all attributes are set correctly
        assert dataset.input_path == "/test/path"
        assert dataset.target_path == "/test/target"
        assert dataset.classes == ["class1"]
        assert dataset.raw_only is False
        assert dataset.target_arrays == {
            "target": {"shape": [50, 50, 50], "scale": [0.5, 0.5, 0.5]}
        }
        assert dataset.spatial_transforms == spatial_transforms
        assert dataset.raw_value_transforms == raw_value_transforms
        assert dataset.target_value_transforms == target_value_transforms
        assert dataset.class_relationships == class_relationships
        assert (
            dataset.class_relation_dict == class_relationships
        )  # backward compatibility
        assert dataset.is_train is True
        assert dataset.axis_order == "zyx"
        assert np.array_equal(dataset.context, context)
        assert dataset._rng == rng
        assert dataset.force_has_data is True
        assert dataset.empty_value == -1
        assert dataset.pad is True
        assert dataset._current_center is None
        assert dataset._current_spatial_transforms is None


class TestCellMapDatasetDataAccess:
    """Test data access methods and properties."""

    @patch("cellmap_data.dataset.CellMapImage")
    def test_getitem_method(self, mock_cellmap_image):
        """Test __getitem__ method functionality."""
        # Setup mock - use MagicMock to support magic methods
        mock_image = MagicMock(spec=CellMapImage)
        mock_image.__getitem__.return_value = torch.zeros((10, 10, 10))
        mock_cellmap_image.return_value = mock_image

        dataset = CellMapDataset(
            input_path="/test/path",
            target_path="/test/target",
            input_arrays={"test": {"shape": [100, 100, 100], "scale": [1.0, 1.0, 1.0]}},
            classes=["class1"],
        )

        # Test data access
        result = dataset[0]

        # Verify the mock was called appropriately
        # Note: Actual implementation details may vary
        assert mock_image.__getitem__.called

    @patch("cellmap_data.dataset.CellMapImage")
    def test_len_method(self, mock_cellmap_image):
        """Test __len__ method functionality."""
        mock_image = MagicMock(spec=CellMapImage)
        mock_image.__len__.return_value = 100
        mock_cellmap_image.return_value = mock_image

        dataset = CellMapDataset(
            input_path="/test/path",
            target_path="/test/target",
            input_arrays={"test": {"shape": [100, 100, 100], "scale": [1.0, 1.0, 1.0]}},
            classes=["class1"],
        )

        # Test length calculation
        length = len(dataset)

        # Verify the mock was called
        assert mock_image.__len__.called

    def test_property_access(self):
        """Test various property accessors."""
        dataset = CellMapDataset(
            input_path="/test/path",
            target_path="/test/target",
            input_arrays={"test": {"shape": [100, 100, 100], "scale": [1.0, 1.0, 1.0]}},
            classes=["class1"],
        )

        # Test that properties are accessible without errors
        assert dataset.input_path == "/test/path"
        assert dataset.raw_path == "/test/path"  # backward compatibility
        assert dataset.target_path == "/test/target"
        assert isinstance(dataset.input_sources, dict)


class TestCellMapDatasetEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_input_arrays(self):
        """Test behavior with empty input_arrays."""
        dataset = CellMapDataset(
            input_path="/test/path",
            target_path="/test/target",
            input_arrays={},  # empty dict
            classes=["class1"],
        )

        assert len(dataset.input_sources) == 0

    def test_none_target_arrays(self):
        """Test behavior with None target_arrays."""
        dataset = CellMapDataset(
            input_path="/test/path",
            target_path="/test/target",
            input_arrays={"test": {"shape": [100, 100, 100], "scale": [1.0, 1.0, 1.0]}},
            classes=["class1"],
            target_arrays=None,
        )

        assert dataset.target_arrays == {}

    def test_empty_classes_list(self):
        """Test behavior with empty classes list."""
        dataset = CellMapDataset(
            input_path="/test/path",
            target_path="/test/target",
            input_arrays={"test": {"shape": [100, 100, 100], "scale": [1.0, 1.0, 1.0]}},
            classes=[],  # empty list
        )

        assert dataset.classes == []
        assert dataset.raw_only is False  # classes is not None

    @patch("cellmap_data.dataset.CellMapImage")
    def test_multiple_input_arrays(self, mock_cellmap_image):
        """Test handling of multiple input arrays."""
        mock_image = Mock(spec=CellMapImage)
        mock_cellmap_image.return_value = mock_image

        input_arrays = {
            f"array_{i}": {"shape": [100, 100, 100], "scale": [1.0, 1.0, 1.0]}
            for i in range(5)
        }

        dataset = CellMapDataset(
            input_path="/test/path",
            target_path="/test/target",
            input_arrays=input_arrays,
            classes=["class1"],
        )

        assert len(dataset.input_sources) == 5
        assert mock_cellmap_image.call_count == 5

        for i in range(5):
            assert f"array_{i}" in dataset.input_sources


if __name__ == "__main__":
    pytest.main([__file__])
