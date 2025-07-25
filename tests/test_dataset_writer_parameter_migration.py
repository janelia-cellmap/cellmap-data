"""Test parameter migration from raw_path to input_path in dataset_writer.py"""

import pytest
import inspect
from cellmap_data.dataset_writer import CellMapDatasetWriter


class TestDatasetWriterParameterMigration:
    """Test the raw_path -> input_path parameter migration in CellMapDatasetWriter."""

    def test_constructor_has_input_path_parameter(self):
        """Test that the constructor includes the input_path parameter."""
        signature = inspect.signature(CellMapDatasetWriter.__init__)
        params = signature.parameters

        assert "input_path" in params
        assert "raw_path" in params  # Keep for backward compatibility

        # input_path should be optional (has default None)
        assert params["input_path"].default is None
        assert params["raw_path"].default is None

    def test_both_parameters_specified_raises_error(self):
        """Test that specifying both raw_path and input_path raises an error."""
        # Mock the minimum required parameters to test parameter validation
        from cellmap_data.utils.error_handling import ValidationError

        with pytest.raises(
            ValidationError,
            match="Cannot specify both 'raw_path' and 'input_path' parameters",
        ):
            try:
                CellMapDatasetWriter(
                    raw_path="/test/raw",
                    input_path="/test/input",
                    target_path="/test/target",
                    classes=["class1"],
                    input_arrays={
                        "array1": {"scale": [1, 1, 1], "shape": [100, 100, 100]}
                    },
                    target_arrays={
                        "array1": {"scale": [1, 1, 1], "shape": [100, 100, 100]}
                    },
                    target_bounds={"array1": {"class1": [0, 100, 0, 100, 0, 100]}},
                )
            except Exception as e:
                # Re-raise only the ValueError we're looking for, suppress others
                if "Cannot specify both" in str(e):
                    raise
                # Other errors are expected due to incomplete test setup

    def test_neither_parameter_specified_raises_error(self):
        """Test that specifying neither parameter raises an error."""
        from cellmap_data.utils.error_handling import ValidationError

        with pytest.raises(
            ValidationError, match="Parameter 'input_path' is required but not provided"
        ):
            try:
                CellMapDatasetWriter(
                    target_path="/test/target",
                    classes=["class1"],
                    input_arrays={
                        "array1": {"scale": [1, 1, 1], "shape": [100, 100, 100]}
                    },
                    target_arrays={
                        "array1": {"scale": [1, 1, 1], "shape": [100, 100, 100]}
                    },
                    target_bounds={"array1": {"class1": [0, 100, 0, 100, 0, 100]}},
                )
            except Exception as e:
                # Re-raise ValidationError for input_path parameter
                if "Parameter 'input_path' is required but not provided" in str(e):
                    raise
                # Other errors are expected due to incomplete test setup

    def test_parameter_migration_code_structure(self):
        """Test that the parameter migration code is present in the constructor."""

        source = inspect.getsource(CellMapDatasetWriter.__init__)

        # Check for migration logic using the new error handling approach
        assert "validate_parameter_conflict" in source
        assert "raw_path" in source
        assert "input_path" in source
        assert "if raw_path is not None:" in source  # New structure
        assert "raw_path' is deprecated" in source
        assert "Use 'input_path' instead" in source
        assert "DeprecationWarning" in source
        assert "input_path = raw_path" in source
        assert "validate_parameter_required" in source

    def test_internal_usage_uses_input_path(self):
        """Test that internal code uses input_path instead of raw_path."""

        source = inspect.getsource(CellMapDatasetWriter)

        # Check that CellMapImage is called with input_path
        assert "CellMapImage(\n                self.input_path," in source

        # Check that __repr__ uses input_path
        assert "Input path: {self.input_path}" in source

        # Should not have old raw_path usage in key places
        assert "CellMapImage(\n                self.raw_path," not in source

    def test_deprecation_warning_structure(self):
        """Test that the deprecation warning code is properly structured."""

        source = inspect.getsource(CellMapDatasetWriter.__init__)

        # Check for proper warning structure
        assert "import warnings" in source
        assert "warnings.warn(" in source
        assert "stacklevel=2" in source
