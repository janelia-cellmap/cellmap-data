"""Tests for class_relationships parameter migration in datasplit.py."""

import pytest
import warnings
import torch
from unittest.mock import patch
from cellmap_data import CellMapDataSplit


class TestClassRelationshipsMigration:
    """Test class for validating the class_relationships parameter migration."""

    @pytest.fixture
    def basic_test_data(self):
        """Basic test data for CellMapDataSplit."""
        # Create minimal test arrays
        input_arrays = {
            "raw": {
                "shape": (10, 10, 10),
                "scale": [1.0, 1.0, 1.0],
            }
        }
        target_arrays = {
            "gt": {
                "shape": (10, 10, 10),
                "scale": [1.0, 1.0, 1.0],
            }
        }

        # Create minimal dataset dict instead of indices
        dataset_dict = {
            "train": [{"raw": "/tmp/test_raw.zarr", "gt": "/tmp/test_gt.zarr"}],
            "validate": [{"raw": "/tmp/test_raw.zarr", "gt": "/tmp/test_gt.zarr"}],
        }

        return {
            "input_arrays": input_arrays,
            "target_arrays": target_arrays,
            "dataset_dict": dataset_dict,
            "classes": ["background", "foreground"],
        }

    def test_new_parameter_works(self, basic_test_data):
        """Test that the new class_relationships parameter works correctly."""
        class_relationships = {"background": ["bg"], "foreground": ["fg"]}

        with patch("cellmap_data.datasplit.CellMapDataset"):
            split = CellMapDataSplit(
                class_relationships=class_relationships, **basic_test_data
            )

            assert split.class_relationships == class_relationships
            assert hasattr(
                split, "class_relation_dict"
            )  # Legacy attribute should exist
            assert split.class_relation_dict == class_relationships

    def test_legacy_parameter_with_warning(self, basic_test_data):
        """Test that the legacy class_relation_dict parameter works with deprecation warning."""
        class_relation_dict = {"background": ["bg"], "foreground": ["fg"]}

        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")

            with patch("cellmap_data.datasplit.CellMapDataset"):
                split = CellMapDataSplit(
                    class_relation_dict=class_relation_dict, **basic_test_data
                )

                # Check that a deprecation warning was raised
                assert len(warning_list) == 1
                assert issubclass(warning_list[0].category, DeprecationWarning)
                assert "Parameter 'class_relation_dict' is deprecated" in str(
                    warning_list[0].message
                )
                assert "class_relationships" in str(warning_list[0].message)

        # Check that the parameter was properly mapped
        assert split.class_relationships == class_relation_dict
        assert split.class_relation_dict == class_relation_dict

    def test_both_parameters_raise_error(self, basic_test_data):
        """Test that specifying both parameters raises an error."""
        class_relationships = {"background": ["bg"], "foreground": ["fg"]}
        class_relation_dict = {"background": ["bg"], "foreground": ["fg"]}

        with pytest.raises(ValueError, match="Cannot specify both"):
            with patch("cellmap_data.datasplit.CellMapDataset"):
                CellMapDataSplit(
                    class_relationships=class_relationships,
                    class_relation_dict=class_relation_dict,
                    **basic_test_data,
                )

    def test_neither_parameter_uses_none(self, basic_test_data):
        """Test that not specifying either parameter defaults to None."""
        with patch("cellmap_data.datasplit.CellMapDataset"):
            split = CellMapDataSplit(**basic_test_data)

            assert split.class_relationships is None
            assert split.class_relation_dict is None

    def test_parameter_passed_to_datasets(self, basic_test_data):
        """Test that the class_relationships parameter is properly passed to child datasets."""
        class_relationships = {"background": ["bg"], "foreground": ["fg"]}

        with patch("cellmap_data.datasplit.CellMapDataset") as mock_dataset:
            split = CellMapDataSplit(
                class_relationships=class_relationships, **basic_test_data
            )

            # Check that CellMapDataset was called with class_relationships parameter
            assert (
                mock_dataset.call_count >= 2
            )  # At least one for train, one for validation

            # Check that calls used the new parameter name
            for call in mock_dataset.call_args_list:
                assert "class_relationships" in call[1]
                assert call[1]["class_relationships"] == class_relationships
                assert (
                    "class_relation_dict" not in call[1]
                )  # Old parameter should not be passed

    def test_legacy_parameter_passed_to_datasets(self, basic_test_data):
        """Test that the legacy parameter is properly converted when passed to child datasets."""
        class_relation_dict = {"background": ["bg"], "foreground": ["fg"]}

        with patch("cellmap_data.datasplit.CellMapDataset") as mock_dataset:
            with warnings.catch_warnings():
                warnings.simplefilter(
                    "ignore"
                )  # Suppress deprecation warning for this test

                split = CellMapDataSplit(
                    class_relation_dict=class_relation_dict, **basic_test_data
                )

            # Check that CellMapDataset was called with the new parameter name
            assert mock_dataset.call_count >= 2

            for call in mock_dataset.call_args_list:
                assert "class_relationships" in call[1]
                assert call[1]["class_relationships"] == class_relation_dict
                assert (
                    "class_relation_dict" not in call[1]
                )  # Old parameter should not be passed

    def test_internal_usage_consistency(self, basic_test_data):
        """Test that internal usage is consistent with the new parameter name."""
        class_relationships = {"background": ["bg"], "foreground": ["fg"]}

        with patch("cellmap_data.datasplit.CellMapDataset"):
            split = CellMapDataSplit(
                class_relationships=class_relationships, **basic_test_data
            )

            # Check that both attributes reference the same object
            assert split.class_relationships is split.class_relation_dict
            assert id(split.class_relationships) == id(split.class_relation_dict)
