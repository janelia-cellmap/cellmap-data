"""
Tests for CellMapDatasetWriter class.

Tests writing predictions and outputs using real data.
"""

import pytest
import torchvision.transforms.v2 as T
import torch

from cellmap_data import CellMapDatasetWriter

from .test_helpers import create_test_dataset


class TestCellMapDatasetWriter:
    """Test suite for CellMapDatasetWriter class."""

    @pytest.fixture
    def writer_config(self, tmp_path):
        """Create configuration for writer tests."""
        # Create input data
        input_config = create_test_dataset(
            tmp_path / "input",
            raw_shape=(64, 64, 64),
            num_classes=2,
            raw_scale=(8.0, 8.0, 8.0),
        )

        # Output path
        output_path = tmp_path / "output" / "predictions.zarr"

        return {
            "input_config": input_config,
            "output_path": str(output_path),
        }

    def test_initialization_basic(self, writer_config):
        """Test basic DatasetWriter initialization."""
        config = writer_config["input_config"]

        input_arrays = {"raw": {"shape": (32, 32, 32), "scale": (8.0, 8.0, 8.0)}}
        target_arrays = {
            "predictions": {"shape": (32, 32, 32), "scale": (8.0, 8.0, 8.0)}
        }

        target_bounds = {
            "predictions": {
                "x": [0, 256],
                "y": [0, 256],
                "z": [0, 256],
            }
        }
        writer = CellMapDatasetWriter(
            raw_path=config["raw_path"],
            target_path=writer_config["output_path"],
            classes=["class_0", "class_1"],
            input_arrays=input_arrays,
            target_arrays=target_arrays,
            target_bounds=target_bounds,
        )

        assert writer is not None
        assert writer.raw_path == config["raw_path"]
        assert writer.target_path == writer_config["output_path"]

    def test_classes_parameter(self, writer_config):
        """Test classes parameter."""
        config = writer_config["input_config"]

        classes = ["class_0", "class_1", "class_2"]

        target_bounds = {
            "pred": {
                "x": [0, 128],
                "y": [0, 128],
                "z": [0, 128],
            }
        }
        writer = CellMapDatasetWriter(
            raw_path=config["raw_path"],
            target_path=writer_config["output_path"],
            classes=classes,
            input_arrays={"raw": {"shape": (16, 16, 16), "scale": (8.0, 8.0, 8.0)}},
            target_arrays={"pred": {"shape": (16, 16, 16), "scale": (8.0, 8.0, 8.0)}},
            target_bounds=target_bounds,
        )

        assert writer.classes == classes

    def test_input_arrays_configuration(self, writer_config):
        """Test input arrays configuration."""
        config = writer_config["input_config"]

        input_arrays = {
            "raw_4nm": {"shape": (32, 32, 32), "scale": (4.0, 4.0, 4.0)},
            "raw_8nm": {"shape": (16, 16, 16), "scale": (8.0, 8.0, 8.0)},
        }

        target_bounds = {
            "pred": {
                "x": [0, 128],
                "y": [0, 128],
                "z": [0, 128],
            }
        }
        writer = CellMapDatasetWriter(
            raw_path=config["raw_path"],
            target_path=writer_config["output_path"],
            classes=["class_0"],
            input_arrays=input_arrays,
            target_arrays={"pred": {"shape": (16, 16, 16), "scale": (8.0, 8.0, 8.0)}},
            target_bounds=target_bounds,
        )

        assert "raw_4nm" in writer.input_arrays
        assert "raw_8nm" in writer.input_arrays

    def test_target_arrays_configuration(self, writer_config):
        """Test target arrays configuration."""
        config = writer_config["input_config"]

        target_arrays = {
            "predictions": {"shape": (32, 32, 32), "scale": (8.0, 8.0, 8.0)},
            "confidences": {"shape": (32, 32, 32), "scale": (8.0, 8.0, 8.0)},
        }

        target_bounds = {
            "predictions": {
                "x": [0, 256],
                "y": [0, 256],
                "z": [0, 256],
            },
            "confidences": {
                "x": [0, 256],
                "y": [0, 256],
                "z": [0, 256],
            },
        }
        writer = CellMapDatasetWriter(
            raw_path=config["raw_path"],
            target_path=writer_config["output_path"],
            classes=["class_0"],
            input_arrays={"raw": {"shape": (32, 32, 32), "scale": (8.0, 8.0, 8.0)}},
            target_arrays=target_arrays,
            target_bounds=target_bounds,
        )

        assert "predictions" in writer.target_arrays
        assert "confidences" in writer.target_arrays

    def test_target_bounds_parameter(self, writer_config):
        """Test target bounds parameter."""
        config = writer_config["input_config"]

        target_bounds = {
            "pred": {
                "x": [0, 512],
                "y": [0, 512],
                "z": [0, 64],
            }
        }

        writer = CellMapDatasetWriter(
            raw_path=config["raw_path"],
            target_path=writer_config["output_path"],
            classes=["class_0"],
            input_arrays={"raw": {"shape": (16, 16, 16), "scale": (8.0, 8.0, 8.0)}},
            target_arrays={"pred": {"shape": (16, 16, 16), "scale": (8.0, 8.0, 8.0)}},
            target_bounds=target_bounds,
        )

        assert writer is not None

    def test_axis_order_parameter(self, writer_config):
        """Test axis order parameter."""
        config = writer_config["input_config"]

        target_bounds = {
            "pred": {
                "x": [0, 128],
                "y": [0, 128],
                "z": [0, 128],
            }
        }
        for axis_order in ["zyx", "xyz", "yxz"]:
            writer = CellMapDatasetWriter(
                raw_path=config["raw_path"],
                target_path=writer_config["output_path"],
                classes=["class_0"],
                input_arrays={"raw": {"shape": (16, 16, 16), "scale": (8.0, 8.0, 8.0)}},
                target_arrays={
                    "pred": {"shape": (16, 16, 16), "scale": (8.0, 8.0, 8.0)}
                },
                axis_order=axis_order,
                target_bounds=target_bounds,
            )
            assert writer.axis_order == axis_order

    def test_pad_parameter(self, writer_config):
        """Test pad parameter."""
        config = writer_config["input_config"]

        target_bounds = {
            "pred": {
                "x": [0, 128],
                "y": [0, 128],
                "z": [0, 128],
            }
        }
        writer_pad = CellMapDatasetWriter(
            raw_path=config["raw_path"],
            target_path=writer_config["output_path"],
            classes=["class_0"],
            input_arrays={"raw": {"shape": (16, 16, 16), "scale": (8.0, 8.0, 8.0)}},
            target_arrays={"pred": {"shape": (16, 16, 16), "scale": (8.0, 8.0, 8.0)}},
            target_bounds=target_bounds,
        )
        assert writer_pad.input_sources["raw"].pad is True

        writer_no_pad = CellMapDatasetWriter(
            raw_path=config["raw_path"],
            target_path=writer_config["output_path"],
            classes=["class_0"],
            input_arrays={"raw": {"shape": (16, 16, 16), "scale": (8.0, 8.0, 8.0)}},
            target_arrays={"pred": {"shape": (16, 16, 16), "scale": (8.0, 8.0, 8.0)}},
            target_bounds=target_bounds,
        )
        assert writer_no_pad.input_sources["raw"].pad is True

    def test_device_parameter(self, writer_config):
        """Test device parameter."""
        config = writer_config["input_config"]

        target_bounds = {
            "pred": {
                "x": [0, 128],
                "y": [0, 128],
                "z": [0, 128],
            }
        }
        writer = CellMapDatasetWriter(
            raw_path=config["raw_path"],
            target_path=writer_config["output_path"],
            classes=["class_0"],
            input_arrays={"raw": {"shape": (16, 16, 16), "scale": (8.0, 8.0, 8.0)}},
            target_arrays={"pred": {"shape": (16, 16, 16), "scale": (8.0, 8.0, 8.0)}},
            device="cpu",
            target_bounds=target_bounds,
        )

        assert writer is not None

    def test_context_parameter(self, writer_config):
        """Test TensorStore context parameter."""
        import tensorstore as ts

        config = writer_config["input_config"]
        context = ts.Context()

        target_bounds = {
            "pred": {
                "x": [0, 128],
                "y": [0, 128],
                "z": [0, 128],
            }
        }
        writer = CellMapDatasetWriter(
            raw_path=config["raw_path"],
            target_path=writer_config["output_path"],
            classes=["class_0"],
            input_arrays={"raw": {"shape": (16, 16, 16), "scale": (8.0, 8.0, 8.0)}},
            target_arrays={"pred": {"shape": (16, 16, 16), "scale": (8.0, 8.0, 8.0)}},
            context=context,
            target_bounds=target_bounds,
        )

        assert writer.context is context


class TestWriterOperations:
    """Test writer operations and functionality."""

    def test_writer_with_value_transforms(self, tmp_path):
        """Test writer with value transforms."""
        config = create_test_dataset(
            tmp_path / "input",
            raw_shape=(32, 32, 32),
            num_classes=2,
        )

        output_path = tmp_path / "output.zarr"

        raw_transform = T.ToDtype(torch.float, scale=True)

        target_bounds = {
            "pred": {
                "x": [0, 128],
                "y": [0, 128],
                "z": [0, 128],
            }
        }
        writer = CellMapDatasetWriter(
            raw_path=config["raw_path"],
            target_path=str(output_path),
            classes=["class_0"],
            input_arrays={"raw": {"shape": (16, 16, 16), "scale": (8.0, 8.0, 8.0)}},
            target_arrays={"pred": {"shape": (16, 16, 16), "scale": (8.0, 8.0, 8.0)}},
            raw_value_transforms=raw_transform,
            target_bounds=target_bounds,
        )

        assert writer.raw_value_transforms is not None

    def test_writer_different_input_output_shapes(self, tmp_path):
        """Test writer with different input and output shapes."""
        config = create_test_dataset(
            tmp_path / "input",
            raw_shape=(64, 64, 64),
            num_classes=2,
        )

        output_path = tmp_path / "output.zarr"

        # Input larger than output
        target_bounds = {
            "pred": {
                "x": [0, 128],
                "y": [0, 128],
                "z": [0, 128],
            }
        }
        writer = CellMapDatasetWriter(
            raw_path=config["raw_path"],
            target_path=str(output_path),
            classes=["class_0"],
            input_arrays={"raw": {"shape": (32, 32, 32), "scale": (8.0, 8.0, 8.0)}},
            target_arrays={"pred": {"shape": (16, 16, 16), "scale": (8.0, 8.0, 8.0)}},
            target_bounds=target_bounds,
        )

        assert writer.input_arrays["raw"]["shape"] == (32, 32, 32)
        assert writer.target_arrays["pred"]["shape"] == (16, 16, 16)

    def test_writer_anisotropic_resolution(self, tmp_path):
        """Test writer with anisotropic voxel sizes."""
        config = create_test_dataset(
            tmp_path / "input",
            raw_shape=(32, 64, 64),
            raw_scale=(16.0, 4.0, 4.0),
            num_classes=2,
        )

        output_path = tmp_path / "output.zarr"

        target_bounds = {
            "pred": {
                "x": [0, 128],
                "y": [0, 256],
                "z": [0, 512],
            }
        }
        writer = CellMapDatasetWriter(
            raw_path=config["raw_path"],
            target_path=str(output_path),
            classes=["class_0"],
            input_arrays={"raw": {"shape": (16, 32, 32), "scale": (16.0, 4.0, 4.0)}},
            target_arrays={"pred": {"shape": (16, 32, 32), "scale": (16.0, 4.0, 4.0)}},
            target_bounds=target_bounds,
        )

        assert writer.input_arrays["raw"]["scale"] == (16.0, 4.0, 4.0)


class TestWriterIntegration:
    """Integration tests for writer functionality."""

    def test_writer_prediction_workflow(self, tmp_path):
        """Test complete prediction writing workflow."""
        # Create input data
        config = create_test_dataset(
            tmp_path / "input",
            raw_shape=(64, 64, 64),
            num_classes=2,
        )

        output_path = tmp_path / "predictions.zarr"

        # Create writer
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

        # Writer should be ready
        assert writer is not None

    def test_writer_with_bounds(self, tmp_path):
        """Test writer with specific spatial bounds."""
        config = create_test_dataset(
            tmp_path / "input",
            raw_shape=(128, 128, 128),
            num_classes=2,
        )

        output_path = tmp_path / "predictions.zarr"

        # Only write to specific region
        target_bounds = {
            "pred": {
                "x": [32, 96],
                "y": [32, 96],
                "z": [0, 64],
            }
        }

        writer = CellMapDatasetWriter(
            raw_path=config["raw_path"],
            target_path=str(output_path),
            classes=["class_0"],
            input_arrays={"raw": {"shape": (32, 32, 32), "scale": (8.0, 8.0, 8.0)}},
            target_arrays={"pred": {"shape": (32, 32, 32), "scale": (8.0, 8.0, 8.0)}},
            target_bounds=target_bounds,
        )

        assert writer is not None

    def test_multi_output_writer(self, tmp_path):
        """Test writer with multiple output arrays."""
        config = create_test_dataset(
            tmp_path / "input",
            raw_shape=(64, 64, 64),
            num_classes=3,
        )

        output_path = tmp_path / "predictions.zarr"

        # Multiple outputs
        target_arrays = {
            "predictions": {"shape": (32, 32, 32), "scale": (8.0, 8.0, 8.0)},
            "uncertainties": {"shape": (32, 32, 32), "scale": (8.0, 8.0, 8.0)},
            "embeddings": {"shape": (32, 32, 32), "scale": (8.0, 8.0, 8.0)},
        }

        target_bounds = {
            "predictions": {
                "x": [0, 512],
                "y": [0, 512],
                "z": [0, 512],
            },
            "uncertainties": {
                "x": [0, 512],
                "y": [0, 512],
                "z": [0, 512],
            },
            "embeddings": {
                "x": [0, 512],
                "y": [0, 512],
                "z": [0, 512],
            },
        }
        writer = CellMapDatasetWriter(
            raw_path=config["raw_path"],
            target_path=str(output_path),
            classes=["class_0", "class_1", "class_2"],
            input_arrays={"raw": {"shape": (32, 32, 32), "scale": (8.0, 8.0, 8.0)}},
            target_arrays=target_arrays,
            target_bounds=target_bounds,
        )

        assert len(writer.target_arrays) == 3

    def test_writer_2d_output(self, tmp_path):
        """Test writer for 2D outputs."""
        # Create 2D input data
        from .test_helpers import create_test_image_data, create_test_zarr_array

        input_path = tmp_path / "input_2d.zarr"
        data_2d = create_test_image_data((128, 128), pattern="gradient")
        create_test_zarr_array(input_path, data_2d, axes=("y", "x"), scale=(4.0, 4.0))

        output_path = tmp_path / "output_2d.zarr"

        target_bounds = {
            "pred": {
                "x": [0, 512],
                "y": [0, 512],
            }
        }
        writer = CellMapDatasetWriter(
            raw_path=str(input_path),
            target_path=str(output_path),
            classes=["class_0"],
            input_arrays={"raw": {"shape": (64, 64), "scale": (4.0, 4.0)}},
            target_arrays={"pred": {"shape": (64, 64), "scale": (4.0, 4.0)}},
            axis_order="yx",
            target_bounds=target_bounds,
        )

        assert writer.axis_order == "yx"
