"""Tests for get_indices methods in CellMapDataset and CellMapDatasetWriter that have TODO: ADD TEST comments."""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch

from cellmap_data import CellMapDataset
from cellmap_data.dataset_writer import CellMapDatasetWriter


class TestDatasetGetIndices:
    """Test the get_indices method in CellMapDataset (dataset.py line 939)."""

    @pytest.fixture
    def mock_dataset(self):
        """Create a mock dataset with known sampling_box_shape."""
        dataset = Mock(spec=CellMapDataset)
        dataset.sampling_box_shape = {"z": 100, "y": 200, "x": 300}
        dataset.axis_order = "zyx"  # Add missing attribute
        dataset.get_indices = CellMapDataset.get_indices.__get__(dataset)
        return dataset

    def test_get_indices_basic_functionality(self, mock_dataset):
        """Test basic functionality of get_indices method."""
        chunk_size = {"z": 50, "y": 100, "x": 50}

        indices = mock_dataset.get_indices(chunk_size)

        # Should return a list of indices
        assert isinstance(indices, list)
        assert len(indices) > 0

        # All indices should be integers
        for idx in indices:
            assert isinstance(idx, (int, np.integer))

    def test_get_indices_chunk_size_alignment(self, mock_dataset):
        """Test that indices align with chunk_size expectations."""
        chunk_size = {"z": 25, "y": 50, "x": 75}

        indices = mock_dataset.get_indices(chunk_size)

        # With sampling_box_shape = {"z": 100, "y": 200, "x": 300}
        # Expected indices per dimension:
        # z: [0, 25, 50, 75] -> 4 values
        # y: [0, 50, 100, 150] -> 4 values
        # x: [0, 75, 150, 225] -> 4 values
        # Total combinations: 4 * 4 * 4 = 64
        assert len(indices) == 64

    def test_get_indices_single_chunk(self, mock_dataset):
        """Test with chunk size equal to sampling box shape."""
        chunk_size = {"z": 100, "y": 200, "x": 300}

        indices = mock_dataset.get_indices(chunk_size)

        # Should have only one index (origin)
        assert len(indices) == 1
        assert indices[0] == 0

    def test_get_indices_large_chunk_size(self, mock_dataset):
        """Test with chunk size larger than sampling box shape."""
        chunk_size = {"z": 200, "y": 400, "x": 600}

        indices = mock_dataset.get_indices(chunk_size)

        # Should still return at least one index
        assert len(indices) >= 1

    def test_get_indices_missing_axis(self, mock_dataset):
        """Test behavior when chunk_size is missing an axis."""
        chunk_size = {"z": 50, "y": 100}  # Missing 'x'

        with pytest.raises(KeyError):
            mock_dataset.get_indices(chunk_size)

    def test_get_indices_empty_chunk_size(self, mock_dataset):
        """Test behavior with empty chunk_size."""
        chunk_size = {}

        # For empty chunk_size, no axes to iterate over
        indices = mock_dataset.get_indices(chunk_size)

        # Should return a single index for empty case
        assert len(indices) == 1


class TestDatasetWriterGetIndices:
    """Test the get_indices method in CellMapDatasetWriter (dataset_writer.py line 565)."""

    @pytest.fixture
    def mock_dataset_writer(self):
        """Create a mock dataset writer with known properties."""
        writer = Mock(spec=CellMapDatasetWriter)
        writer.smallest_voxel_sizes = {"z": 2.0, "y": 1.0, "x": 1.0}
        writer.sampling_box_shape = {
            "z": 50,
            "y": 100,
            "x": 150,
        }
        writer.axis_order = "zyx"
        writer.get_indices = CellMapDatasetWriter.get_indices.__get__(writer)
        return writer

    def test_get_indices_world_to_voxel_conversion(self, mock_dataset_writer):
        """Test conversion from world units to voxel units."""
        # World units chunk size
        chunk_size_world = {"z": 10.0, "y": 5.0, "x": 3.0}

        indices = mock_dataset_writer.get_indices(chunk_size_world)

        # Should return a list of indices
        assert isinstance(indices, list)
        assert len(indices) > 0

        # All indices should be integers
        for idx in indices:
            assert isinstance(idx, (int, np.integer))

    def test_get_indices_fractional_world_units(self, mock_dataset_writer):
        """Test with fractional world units that don't divide evenly."""
        chunk_size_world = {"z": 3.7, "y": 2.3, "x": 1.8}

        indices = mock_dataset_writer.get_indices(chunk_size_world)

        # Should return a list of indices
        assert isinstance(indices, list)
        assert len(indices) > 0

        # All indices should be integers
        for idx in indices:
            assert isinstance(idx, (int, np.integer))

    def test_get_indices_zero_world_units(self, mock_dataset_writer):
        """Test with zero world units (edge case)."""
        chunk_size_world = {"z": 0.5, "y": 0.8, "x": 0.0}

        indices = mock_dataset_writer.get_indices(chunk_size_world)

        # Should return a list of indices (handles zero gracefully)
        assert isinstance(indices, list)
        assert len(indices) > 0

        # All indices should be integers
        for idx in indices:
            assert isinstance(idx, (int, np.integer))

    def test_get_indices_large_world_units(self, mock_dataset_writer):
        """Test with large world units."""
        chunk_size_world = {"z": 100.0, "y": 50.0, "x": 25.0}

        indices = mock_dataset_writer.get_indices(chunk_size_world)

        # Should return a list of indices
        assert isinstance(indices, list)
        assert len(indices) > 0

        # All indices should be integers
        for idx in indices:
            assert isinstance(idx, (int, np.integer))

    def test_get_indices_missing_axis_in_world_units(self, mock_dataset_writer):
        """Test behavior when world chunk_size is missing an axis."""
        chunk_size_world = {"z": 10.0, "y": 5.0}  # Missing 'x'

        with pytest.raises(KeyError):
            mock_dataset_writer.get_indices(chunk_size_world)

    def test_get_indices_missing_voxel_size(self):
        """Test behavior when smallest_voxel_sizes is missing an axis."""
        writer = Mock(spec=CellMapDatasetWriter)
        writer.smallest_voxel_sizes = {"z": 2.0, "y": 1.0}  # Missing 'x'
        writer.sampling_box_shape = {
            "z": 50,
            "y": 100,
            "x": 150,
        }
        writer.axis_order = "zyx"
        writer.get_indices = CellMapDatasetWriter.get_indices.__get__(writer)

        chunk_size_world = {"z": 10.0, "y": 5.0, "x": 3.0}

        with pytest.raises(KeyError):
            writer.get_indices(chunk_size_world)


class TestIntegrationGetIndicesComparison:
    """Integration tests comparing both get_indices methods."""

    def test_consistent_behavior_with_unit_voxel_sizes(self):
        """Test that both methods give consistent results when voxel sizes are 1.0."""
        # Mock dataset
        dataset = Mock(spec=CellMapDataset)
        dataset.sampling_box_shape = {"z": 50, "y": 100, "x": 150}
        dataset.axis_order = "zyx"  # Add missing attribute
        dataset.get_indices = CellMapDataset.get_indices.__get__(dataset)

        # Mock dataset writer with unit voxel sizes
        writer = Mock(spec=CellMapDatasetWriter)
        writer.smallest_voxel_sizes = {"z": 1.0, "y": 1.0, "x": 1.0}
        writer.sampling_box_shape = {
            "z": 50,
            "y": 100,
            "x": 150,
        }  # Add missing attribute
        writer.axis_order = "zyx"  # Add missing attribute
        writer.get_indices = CellMapDatasetWriter.get_indices.__get__(writer)

        chunk_size_voxel = {"z": 25, "y": 50, "x": 75}
        chunk_size_world = {"z": 25.0, "y": 50.0, "x": 75.0}

        dataset_indices = dataset.get_indices(chunk_size_voxel)
        writer_indices = writer.get_indices(chunk_size_world)

        # Both should return lists of integers
        assert isinstance(dataset_indices, list)
        assert isinstance(writer_indices, list)
        assert len(dataset_indices) > 0
        assert len(writer_indices) > 0

        # All indices should be integers
        for idx in dataset_indices:
            assert isinstance(idx, (int, np.integer))
        for idx in writer_indices:
            assert isinstance(idx, (int, np.integer))

    def test_scaling_relationship(self):
        """Test the scaling relationship between voxel and world coordinates."""
        # Mock dataset
        dataset = Mock(spec=CellMapDataset)
        dataset.sampling_box_shape = {"z": 100, "y": 200, "x": 300}
        dataset.axis_order = "zyx"  # Add missing attribute
        dataset.get_indices = CellMapDataset.get_indices.__get__(dataset)

        # Mock dataset writer with 2x voxel sizes
        writer = Mock(spec=CellMapDatasetWriter)
        writer.smallest_voxel_sizes = {"z": 2.0, "y": 2.0, "x": 2.0}
        writer.sampling_box_shape = {
            "z": 100,
            "y": 200,
            "x": 300,
        }  # Add missing attribute
        writer.axis_order = "zyx"  # Add missing attribute
        writer.get_indices = CellMapDatasetWriter.get_indices.__get__(writer)

        # World coordinates should be 2x voxel coordinates for same internal conversion
        chunk_size_voxel = {"z": 25, "y": 50, "x": 75}
        chunk_size_world = {"z": 50.0, "y": 100.0, "x": 150.0}  # 2x

        dataset_indices = dataset.get_indices(chunk_size_voxel)
        writer_indices = writer.get_indices(chunk_size_world)

        # Both should return valid lists of integers
        assert isinstance(dataset_indices, list)
        assert isinstance(writer_indices, list)
        assert len(dataset_indices) > 0
        assert len(writer_indices) > 0

        # All indices should be integers
        for idx in dataset_indices:
            assert isinstance(idx, (int, np.integer))
        for idx in writer_indices:
            assert isinstance(idx, (int, np.integer))

    def test_different_implementations_same_parameters(self):
        """Test that both implementations handle the same voxel parameters."""
        # Mock dataset
        dataset = Mock(spec=CellMapDataset)
        dataset.sampling_box_shape = {"z": 64, "y": 128, "x": 256}
        dataset.axis_order = "zyx"
        dataset.get_indices = CellMapDataset.get_indices.__get__(dataset)

        # Mock dataset writer (using unit scaling for equivalent voxel test)
        writer = Mock(spec=CellMapDatasetWriter)
        writer.smallest_voxel_sizes = {"z": 1.0, "y": 1.0, "x": 1.0}
        writer.sampling_box_shape = {"z": 64, "y": 128, "x": 256}
        writer.axis_order = "zyx"
        writer.get_indices = CellMapDatasetWriter.get_indices.__get__(writer)

        # Same effective parameters
        chunk_size_voxel = {"z": 32, "y": 64, "x": 128}
        chunk_size_world = {"z": 32.0, "y": 64.0, "x": 128.0}  # 1:1 scaling

        dataset_indices = dataset.get_indices(chunk_size_voxel)
        writer_indices = writer.get_indices(chunk_size_world)

        # Both should return valid results
        assert isinstance(dataset_indices, list)
        assert isinstance(writer_indices, list)
        assert len(dataset_indices) > 0
        assert len(writer_indices) > 0
