"""
Focused unit tests for P0 critical fixes.
This module tests the specific methods we added to fix the P0 issues.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from cellmap_data.transforms.augment.random_contrast import RandomContrast
from cellmap_data.exceptions import (
    IndexError as CellMapIndexError,
    CoordinateTransformError,
)


class TestRandomContrastFix:
    """Test cases for the RandomContrast NaN handling fix."""

    def test_forward_valid_input(self):
        """Test that valid inputs work correctly."""
        transform = RandomContrast(contrast_range=(0.8, 1.2))

        # Test with normal tensor
        x = torch.randn(3, 64, 64, 64)
        result = transform.forward(x)

        assert result.shape == x.shape
        assert result.dtype == x.dtype
        assert not torch.any(torch.isnan(result))
        assert not torch.any(torch.isinf(result))

    def test_forward_empty_tensor(self):
        """Test that empty tensors are handled properly."""
        transform = RandomContrast()

        x = torch.empty(0, 64, 64)
        result = transform.forward(x)

        assert result.shape == x.shape
        assert result.numel() == 0

    def test_forward_nan_input_raises_error(self):
        """Test that NaN inputs raise ValueError."""
        transform = RandomContrast()

        x = torch.randn(3, 64, 64)
        x[0, 0, 0] = float("nan")

        with pytest.raises(ValueError, match="Input tensor contains NaN values"):
            transform.forward(x)

    def test_forward_inf_input_raises_error(self):
        """Test that infinite inputs raise ValueError."""
        transform = RandomContrast()

        x = torch.randn(3, 64, 64)
        x[0, 0, 0] = float("inf")

        with pytest.raises(ValueError, match="Input tensor contains infinite values"):
            transform.forward(x)

    def test_extreme_contrast_ratios(self):
        """Test that extreme but valid contrast ratios work."""
        # Very low contrast
        transform = RandomContrast(contrast_range=(0.01, 0.01))
        x = torch.randn(3, 64, 64)
        result = transform.forward(x)
        assert not torch.any(torch.isnan(result))

        # Very high contrast
        transform = RandomContrast(contrast_range=(10.0, 10.0))
        x = torch.randn(3, 64, 64)
        result = transform.forward(x)
        assert not torch.any(torch.isnan(result))

    def test_different_dtypes(self):
        """Test that different tensor dtypes are handled correctly."""
        transform = RandomContrast()

        for dtype in [torch.float32, torch.float64, torch.float16]:
            x = torch.randn(3, 32, 32, dtype=dtype)
            result = transform.forward(x)

            assert result.dtype == dtype
            assert not torch.any(torch.isnan(result))
            assert not torch.any(torch.isinf(result))

    def test_no_hack_code_remains(self):
        """Test that the NaN hack has been completely removed."""
        # Read the source file and verify no hack comments remain
        source_file = (
            Path(__file__).parent.parent
            / "src"
            / "cellmap_data"
            / "transforms"
            / "augment"
            / "random_contrast.py"
        )
        if source_file.exists():
            content = source_file.read_text()

            # These should not exist in the fixed code
            assert "Hack to avoid NaN" not in content
            assert "torch.nan_to_num" not in content

    def test_numerical_stability_with_edge_cases(self):
        """Test numerical stability with edge case values."""
        transform = RandomContrast()

        # Test with very small values
        x = torch.full((3, 32, 32), 1e-10)
        result = transform.forward(x)
        assert not torch.any(torch.isnan(result))

        # Test with values near dtype limits
        x = torch.full((3, 32, 32), 0.9)  # Close to uint8 max when scaled
        result = transform.forward(x)
        assert not torch.any(torch.isnan(result))


class TestCoordinateTransformationMethods:
    """Test the specific coordinate transformation methods we added."""

    def test_validate_index_bounds_method_exists(self):
        """Test that the _validate_index_bounds method exists and is callable."""
        from cellmap_data.dataset_writer import CellMapDatasetWriter

        # Check that the method exists
        assert hasattr(CellMapDatasetWriter, "_validate_index_bounds")

        # It should be a callable method
        method = getattr(CellMapDatasetWriter, "_validate_index_bounds")
        assert callable(method)

    def test_safe_unravel_index_method_exists(self):
        """Test that the _safe_unravel_index method exists and is callable."""
        from cellmap_data.dataset_writer import CellMapDatasetWriter

        # Check that the method exists
        assert hasattr(CellMapDatasetWriter, "_safe_unravel_index")

        # It should be a callable method
        method = getattr(CellMapDatasetWriter, "_safe_unravel_index")
        assert callable(method)

    def test_bounds_validation_logic(self):
        """Test the bounds validation logic in isolation."""
        # Create a mock instance to test the method directly
        mock_writer = MagicMock()
        mock_writer.__len__ = MagicMock(return_value=100)

        # Import the actual method
        from cellmap_data.dataset_writer import CellMapDatasetWriter

        validate_method = CellMapDatasetWriter._validate_index_bounds

        # Test valid cases
        try:
            validate_method(mock_writer, 0)
            validate_method(mock_writer, 50)
            validate_method(mock_writer, 99)
        except Exception as e:
            pytest.fail(f"Valid indices should not raise exceptions: {e}")

        # Test invalid cases
        with pytest.raises(CellMapIndexError, match="Index -1 is negative"):
            validate_method(mock_writer, -1)

        with pytest.raises(CellMapIndexError, match="Index 100 is out of bounds"):
            validate_method(mock_writer, 100)

        # Test empty dataset
        mock_writer.__len__ = MagicMock(return_value=0)
        with pytest.raises(CellMapIndexError, match="dataset is empty"):
            validate_method(mock_writer, 0)


class TestCodeQualityAssurance:
    """Test that the hacks have been properly removed."""

    def test_dataset_writer_no_hack_remains(self):
        """Test that the coordinate transformation hack has been removed."""
        from pathlib import Path

        source_file = (
            Path(__file__).parent.parent / "src" / "cellmap_data" / "dataset_writer.py"
        )
        if source_file.exists():
            content = source_file.read_text()

            # These should not exist in the fixed code
            assert "hacky temporary fix" not in content
            assert "TODO: This is a hacky" not in content
            assert "center = [self.sampling_box_shape[c] - 1" not in content

    def test_random_contrast_no_hack_remains(self):
        """Test that the RandomContrast hack has been removed."""
        from pathlib import Path

        source_file = (
            Path(__file__).parent.parent
            / "src"
            / "cellmap_data"
            / "transforms"
            / "augment"
            / "random_contrast.py"
        )
        if source_file.exists():
            content = source_file.read_text()

            # These should not exist in the fixed code
            assert "Hack to avoid NaN" not in content
            assert "torch.nan_to_num" not in content

    def test_proper_exception_imports(self):
        """Test that the proper exceptions are imported and available."""
        from cellmap_data.exceptions import (
            IndexError as CellMapIndexError,
            CoordinateTransformError,
        )

        # These should be proper Exception subclasses
        assert issubclass(CellMapIndexError, Exception)
        assert issubclass(CoordinateTransformError, Exception)

        # Test that we can create instances
        index_error = CellMapIndexError("Test message")
        coord_error = CoordinateTransformError("Test message")

        assert str(index_error) == "Test message"
        assert str(coord_error) == "Test message"


if __name__ == "__main__":
    pytest.main([__file__])
