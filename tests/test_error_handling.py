"""Tests for error handling utilities."""

import pytest
import warnings
from cellmap_data.utils.error_handling import (
    ErrorMessages,
    StandardWarnings,
    ValidationError,
    validate_parameter_conflict,
    validate_parameter_required,
    validate_parameter_type,
    create_bounds_error,
    create_shape_mismatch_error,
)


class TestErrorMessages:
    """Test the ErrorMessages template class."""

    def test_parameter_messages(self):
        """Test parameter-related error message templates."""
        assert "{parameter}" in ErrorMessages.PARAMETER_REQUIRED
        assert "{param1}" in ErrorMessages.PARAMETER_CONFLICT
        assert "{param2}" in ErrorMessages.PARAMETER_CONFLICT
        assert "{parameter}" in ErrorMessages.PARAMETER_DEPRECATED
        assert "{replacement}" in ErrorMessages.PARAMETER_DEPRECATED

    def test_file_messages(self):
        """Test file-related error message templates."""
        assert "{path}" in ErrorMessages.FILE_NOT_FOUND
        assert "{path}" in ErrorMessages.FILE_EXISTS
        assert "overwrite=True" in ErrorMessages.FILE_EXISTS

    def test_data_messages(self):
        """Test data-related error message templates."""
        assert "{expected_shape}" in ErrorMessages.DATA_SHAPE_MISMATCH
        assert "{actual_shape}" in ErrorMessages.DATA_SHAPE_MISMATCH
        assert "{data_type}" in ErrorMessages.DATA_TYPE_UNSUPPORTED

    def test_coordinate_messages(self):
        """Test coordinate-related error message templates."""
        assert "{coordinate}" in ErrorMessages.COORDINATE_OUT_OF_BOUNDS
        assert "{axis}" in ErrorMessages.COORDINATE_OUT_OF_BOUNDS
        assert "{min_val}" in ErrorMessages.COORDINATE_OUT_OF_BOUNDS
        assert "{max_val}" in ErrorMessages.COORDINATE_OUT_OF_BOUNDS


class TestStandardWarnings:
    """Test the StandardWarnings utility class."""

    def test_parameter_deprecated_warning(self):
        """Test parameter deprecation warning."""
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")

            StandardWarnings.parameter_deprecated("old_param", "new_param")

            assert len(warning_list) == 1
            assert issubclass(warning_list[0].category, DeprecationWarning)
            assert "old_param" in str(warning_list[0].message)
            assert "new_param" in str(warning_list[0].message)
            assert "deprecated" in str(warning_list[0].message)

    def test_fallback_driver_warning(self):
        """Test fallback driver warning."""
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")

            StandardWarnings.fallback_driver("zarr3", "connection failed")

            assert len(warning_list) == 1
            assert issubclass(warning_list[0].category, UserWarning)
            assert "zarr3" in str(warning_list[0].message)
            assert "connection failed" in str(warning_list[0].message)
            assert "Falling back" in str(warning_list[0].message)

    def test_performance_warning(self):
        """Test performance warning."""
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")

            StandardWarnings.performance_warning("memory usage high")

            assert len(warning_list) == 1
            assert issubclass(warning_list[0].category, UserWarning)
            assert "memory usage high" in str(warning_list[0].message)
            assert "Performance warning" in str(warning_list[0].message)


class TestValidationError:
    """Test the ValidationError class."""

    def test_basic_formatting(self):
        """Test basic message formatting."""
        error = ValidationError(
            "Test {param} with {value}", param="test_param", value=42
        )
        assert "Test test_param with 42" in str(error)

    def test_inheritance(self):
        """Test that ValidationError inherits from ValueError."""
        error = ValidationError("Test message")
        assert isinstance(error, ValueError)
        assert isinstance(error, ValidationError)


class TestValidationFunctions:
    """Test the validation utility functions."""

    def test_validate_parameter_conflict_both_none(self):
        """Test that no error is raised when both parameters are None."""
        # Should not raise any exception
        validate_parameter_conflict("param1", None, "param2", None)

    def test_validate_parameter_conflict_one_set(self):
        """Test that no error is raised when only one parameter is set."""
        # Should not raise any exception
        validate_parameter_conflict("param1", "value1", "param2", None)
        validate_parameter_conflict("param1", None, "param2", "value2")

    def test_validate_parameter_conflict_both_set(self):
        """Test that error is raised when both parameters are set."""
        with pytest.raises(ValidationError) as exc_info:
            validate_parameter_conflict("param1", "value1", "param2", "value2")

        assert "Cannot specify both" in str(exc_info.value)
        assert "param1" in str(exc_info.value)
        assert "param2" in str(exc_info.value)

    def test_validate_parameter_required_provided(self):
        """Test that no error is raised when required parameter is provided."""
        # Should not raise any exception
        validate_parameter_required("test_param", "value")
        validate_parameter_required("test_param", 0)  # Falsy but not None
        validate_parameter_required("test_param", [])  # Empty but not None

    def test_validate_parameter_required_missing(self):
        """Test that error is raised when required parameter is missing."""
        with pytest.raises(ValidationError) as exc_info:
            validate_parameter_required("required_param", None)

        assert "Parameter 'required_param' is required" in str(exc_info.value)

    def test_validate_parameter_type_correct(self):
        """Test that no error is raised when parameter type is correct."""
        # Should not raise any exception
        validate_parameter_type("string_param", "value", str)
        validate_parameter_type("int_param", 42, int)
        validate_parameter_type("list_param", [], list)
        validate_parameter_type("optional_param", None, str)  # None is allowed

    def test_validate_parameter_type_incorrect(self):
        """Test that error is raised when parameter type is incorrect."""
        with pytest.raises(ValidationError) as exc_info:
            validate_parameter_type("string_param", 42, str)

        assert "Parameter 'string_param' must be of type str, got int" in str(
            exc_info.value
        )


class TestErrorMessageCreators:
    """Test the error message creation utility functions."""

    def test_create_bounds_error(self):
        """Test coordinate bounds error message creation."""
        error_msg = create_bounds_error(5.5, "x", 0.0, 10.0)

        assert "5.5" in error_msg
        assert "x" in error_msg
        assert "0.0" in error_msg
        assert "10.0" in error_msg
        assert "out of bounds" in error_msg

    def test_create_shape_mismatch_error(self):
        """Test shape mismatch error message creation."""
        error_msg = create_shape_mismatch_error((100, 100, 100), (50, 50, 50))

        assert "(100, 100, 100)" in error_msg
        assert "(50, 50, 50)" in error_msg
        assert "shape mismatch" in error_msg.lower()


class TestIntegrationWithExistingCode:
    """Test integration with existing CellMap-Data patterns."""

    def test_parameter_migration_pattern(self):
        """Test the error handling with parameter migration pattern."""

        # Simulate the pattern used in dataset_writer.py
        def mock_constructor(input_path=None, raw_path=None):
            validate_parameter_conflict("input_path", input_path, "raw_path", raw_path)

            if raw_path is not None:
                StandardWarnings.parameter_deprecated("raw_path", "input_path")
                input_path = raw_path

            validate_parameter_required("input_path", input_path)
            return input_path

        # Test normal usage
        result = mock_constructor(input_path="/test/path")
        assert result == "/test/path"

        # Test deprecated parameter with warning
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            result = mock_constructor(raw_path="/test/path")

            assert result == "/test/path"
            assert len(warning_list) == 1
            assert "deprecated" in str(warning_list[0].message)

        # Test conflict
        with pytest.raises(ValidationError):
            mock_constructor(input_path="/path1", raw_path="/path2")

        # Test missing required parameter
        with pytest.raises(ValidationError):
            mock_constructor()

    def test_driver_fallback_pattern(self):
        """Test the driver fallback warning pattern."""

        # Simulate the pattern used in image.py
        def mock_driver_operation():
            try:
                # Simulate a driver failure
                raise ValueError("Connection failed")
            except ValueError as e:
                StandardWarnings.fallback_driver("zarr3", str(e))
                return "fallback_result"

        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            result = mock_driver_operation()

            assert result == "fallback_result"
            assert len(warning_list) == 1
            assert "zarr3" in str(warning_list[0].message)
            assert "Connection failed" in str(warning_list[0].message)
