"""
Tests for EmptyImage and ImageWriter classes.

Tests empty image handling and image writing functionality.
"""

import pytest
from upath import UPath
from pathlib import Path
import os

from cellmap_data import EmptyImage, ImageWriter

from .test_helpers import create_test_image_data, create_test_zarr_array


@pytest.fixture
def tmp_upath(tmp_path: Path):
    """Return a temporary directory (as :class:`upathlib.UPath` object)
    which is unique to each test function invocation.
    The temporary directory is created as a subdirectory
    of the base temporary directory, with configurable retention,
    as discussed in :ref:`temporary directory location and retention`.
    """
    return UPath(tmp_path)


class TestEmptyImage:
    """Test suite for EmptyImage class."""

    def test_initialization_basic(self):
        """Test basic EmptyImage initialization."""
        empty_image = EmptyImage(
            label_class="test_class",
            scale=(8.0, 8.0, 8.0),
            voxel_shape=(16, 16, 16),
            axis_order="zyx",
        )

        assert empty_image.label_class == "test_class"
        assert empty_image.scale == {"z": 8.0, "y": 8.0, "x": 8.0}
        assert empty_image.output_shape == {"z": 16, "y": 16, "x": 16}

    def test_empty_image_shape(self):
        """Test that EmptyImage has correct shape."""
        shape = (32, 32, 32)
        empty_image = EmptyImage(
            label_class="empty",
            scale=(4.0, 4.0, 4.0),
            voxel_shape=shape,
            axis_order="zyx",
        )

        assert empty_image.output_shape == {"z": 32, "y": 32, "x": 32}

    def test_empty_image_2d(self):
        """Test EmptyImage with 2D shape."""
        empty_image = EmptyImage(
            label_class="empty_2d",
            scale=(4.0, 4.0),
            voxel_shape=(64, 64),
            axis_order="yx",
        )

        assert empty_image.axes == "yx"
        assert len(empty_image.output_shape) == 2

    def test_empty_image_different_scales(self):
        """Test EmptyImage with different scales per axis."""
        empty_image = EmptyImage(
            label_class="anisotropic",
            scale=(16.0, 4.0, 4.0),
            voxel_shape=(16, 32, 32),
            axis_order="zyx",
        )

        assert empty_image.scale == {"z": 16.0, "y": 4.0, "x": 4.0}
        assert empty_image.output_size == {"z": 256.0, "y": 128.0, "x": 128.0}

    def test_empty_image_value_transform(self):
        """Test EmptyImage with value transform."""

        def dummy_transform(x):
            return x * 2

        empty_image = EmptyImage(
            label_class="test",
            scale=(4.0, 4.0, 4.0),
            voxel_shape=(8, 8, 8),
        )
        empty_image.value_transform = dummy_transform

        assert empty_image.value_transform is not None

    def test_empty_image_device(self):
        """Test EmptyImage device assignment."""
        empty_image = EmptyImage(
            label_class="test",
            scale=(4.0, 4.0, 4.0),
            voxel_shape=(8, 8, 8),
        )
        empty_image.to("cpu")

        assert empty_image.store.device.type == "cpu"

    def test_empty_image_pad_parameter(self):
        """Test EmptyImage with pad parameter."""
        empty_image = EmptyImage(
            label_class="test",
            scale=(4.0, 4.0, 4.0),
            voxel_shape=(8, 8, 8),
        )
        empty_image.pad = True
        empty_image.pad_value = 0.0

        assert empty_image.pad is True
        assert empty_image.pad_value == 0.0


class TestImageWriter:
    """Test suite for ImageWriter class."""

    @pytest.fixture
    def output_path(self, tmp_upath):
        """Create output path for writing."""
        return tmp_upath / "output.zarr"

    def test_image_writer_initialization(self, output_path):
        """Test ImageWriter initialization."""
        writer = ImageWriter(
            path=output_path.path,
            target_class="output_class",
            scale=(8.0, 8.0, 8.0),
            write_voxel_shape=(32, 32, 32),
            axis_order="zyx",
            bounding_box={"z": [0, 256], "y": [0, 256], "x": [0, 256]},
        )

        assert writer.path.endswith(output_path.path + os.path.sep + "s0")
        assert writer.target_class == "output_class"

    def test_image_writer_with_existing_data(self, tmp_upath):
        """Test ImageWriter with pre-existing data."""
        # Create existing zarr array
        data = create_test_image_data((32, 32, 32), pattern="gradient")
        path = tmp_upath / "existing.zarr"
        create_test_zarr_array(path, data)

        # Create writer for same path
        writer = ImageWriter(
            path=path.path,
            target_class="test",
            scale=(4.0, 4.0, 4.0),
            write_voxel_shape=(16, 16, 16),
            bounding_box={"z": [0, 128], "y": [0, 128], "x": [0, 128]},
        )

        assert writer.path.endswith(path.path + os.path.sep + "s0")

    def test_image_writer_different_shapes(self, tmp_upath):
        """Test ImageWriter with different output shapes."""
        shapes = [(16, 16, 16), (32, 32, 32), (64, 32, 16)]

        for i, shape in enumerate(shapes):
            path = tmp_upath / f"output_{i}.zarr"
            writer = ImageWriter(
                path=str(path),
                target_class="test",
                scale=(4.0, 4.0, 4.0),
                write_voxel_shape=shape,
                bounding_box={"z": [0, 256], "y": [0, 128], "x": [0, 64]},
            )

            assert writer.write_voxel_shape == {
                "z": shape[0],
                "y": shape[1],
                "x": shape[2],
            }

    def test_image_writer_2d(self, tmp_upath):
        """Test ImageWriter for 2D images."""
        path = tmp_upath / "output_2d.zarr"
        writer = ImageWriter(
            path=str(path),
            target_class="test_2d",
            scale=(4.0, 4.0),
            write_voxel_shape=(64, 64),
            axis_order="yx",
            bounding_box={"y": [0, 256], "x": [0, 256]},
        )

        assert writer.axes == "yx"
        assert len(writer.write_voxel_shape) == 2

    def test_image_writer_value_transform(self, tmp_upath):
        """Test ImageWriter with value transform."""

        def normalize(x):
            return x / 255.0

        path = tmp_upath / "output.zarr"
        writer = ImageWriter(
            path=str(path),
            target_class="test",
            scale=(4.0, 4.0, 4.0),
            write_voxel_shape=(16, 16, 16),
            bounding_box={"z": [0, 64], "y": [0, 64], "x": [0, 64]},
        )
        writer.value_transform = normalize

        assert writer.value_transform is not None

    def test_image_writer_interpolation(self, tmp_upath):
        """Test ImageWriter with different interpolation modes."""
        for interp in ["nearest", "linear"]:
            path = tmp_upath / f"output_{interp}.zarr"
            writer = ImageWriter(
                path=str(path),
                target_class="test",
                scale=(4.0, 4.0, 4.0),
                write_voxel_shape=(16, 16, 16),
                bounding_box={"z": [0, 64], "y": [0, 64], "x": [0, 64]},
            )
            writer.interpolation = interp

            assert writer.interpolation == interp

    def test_image_writer_anisotropic_scale(self, tmp_upath):
        """Test ImageWriter with anisotropic voxel sizes."""
        path = tmp_upath / "anisotropic.zarr"
        writer = ImageWriter(
            path=str(path),
            target_class="test",
            scale=(16.0, 4.0, 4.0),  # Anisotropic
            write_voxel_shape=(16, 32, 32),
            axis_order="zyx",
            bounding_box={"z": [0, 256], "y": [0, 128], "x": [0, 128]},
        )

        assert writer.scale == {"z": 16.0, "y": 4.0, "x": 4.0}
        # Output size should account for scale
        assert writer.write_world_shape == {"z": 256.0, "y": 128.0, "x": 128.0}

    def test_image_writer_context(self, tmp_upath):
        """Test ImageWriter with TensorStore context."""
        import tensorstore as ts

        path = tmp_upath / "output.zarr"
        context = ts.Context()

        writer = ImageWriter(
            path=str(path),
            target_class="test",
            scale=(4.0, 4.0, 4.0),
            write_voxel_shape=(16, 16, 16),
            context=context,
            bounding_box={"z": [0, 64], "y": [0, 64], "x": [0, 64]},
        )

        assert writer.context is context


class TestEmptyImageIntegration:
    """Integration tests for EmptyImage with dataset operations."""

    def test_empty_image_as_placeholder(self):
        """Test using EmptyImage as placeholder in dataset."""
        # EmptyImage can be used when data is missing
        empty = EmptyImage(
            label_class="missing_class",
            scale=(8.0, 8.0, 8.0),
            voxel_shape=(32, 32, 32),
        )

        # Should have proper attributes
        assert empty.label_class == "missing_class"
        assert empty.output_shape is not None

    def test_empty_image_collection(self):
        """Test collection of EmptyImages."""
        # Create multiple empty images for different classes
        empty_images = []
        for i in range(3):
            empty = EmptyImage(
                label_class=f"class_{i}",
                scale=(4.0, 4.0, 4.0),
                voxel_shape=(16, 16, 16),
            )
            empty_images.append(empty)

        assert len(empty_images) == 3
        assert all(img.label_class.startswith("class_") for img in empty_images)


class TestImageWriterIntegration:
    """Integration tests for ImageWriter functionality."""

    def test_writer_output_preparation(self, tmp_upath):
        """Test preparing outputs for writing."""
        path = tmp_upath / "predictions.zarr"

        writer = ImageWriter(
            path=path.path,
            target_class="predictions",
            scale=(8.0, 8.0, 8.0),
            write_voxel_shape=(32, 32, 32),
            bounding_box={"z": [0, 256], "y": [0, 256], "x": [0, 256]},
        )

        # Writer should be ready to write
        assert writer.path.endswith(path.path + os.path.sep + "s0")
        assert writer.write_voxel_shape is not None

    def test_multiple_writers_different_classes(self, tmp_upath):
        """Test multiple writers for different classes."""
        classes = ["class_0", "class_1", "class_2"]
        writers = []

        for class_name in classes:
            path = tmp_upath / f"{class_name}.zarr"
            writer = ImageWriter(
                path=str(path),
                target_class=class_name,
                scale=(4.0, 4.0, 4.0),
                write_voxel_shape=(16, 16, 16),
                bounding_box={"z": [0, 64], "y": [0, 64], "x": [0, 64]},
            )
            writers.append(writer)

        assert len(writers) == 3
        assert all(w.target_class in classes for w in writers)
