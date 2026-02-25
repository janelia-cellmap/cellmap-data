"""
Tests for CellMapDataset class.

Tests dataset creation, data loading, and transformations using real data.
"""

import pytest
import torch
import torchvision.transforms.v2 as T

from cellmap_data import CellMapDataset
from cellmap_data.transforms import Binarize

from .test_helpers import create_minimal_test_dataset, create_test_dataset


class TestCellMapDataset:
    """Test suite for CellMapDataset class."""

    @pytest.fixture
    def minimal_dataset_config(self, tmp_path):
        """Create a minimal dataset configuration."""
        return create_minimal_test_dataset(tmp_path)

    @pytest.fixture
    def standard_dataset_config(self, tmp_path):
        """Create a standard dataset configuration."""
        return create_test_dataset(
            tmp_path,
            raw_shape=(32, 32, 32),
            num_classes=3,
            raw_scale=(8.0, 8.0, 8.0),
        )

    def test_initialization_basic(self, minimal_dataset_config):
        """Test basic dataset initialization."""
        config = minimal_dataset_config

        input_arrays = {
            "raw": {
                "shape": (8, 8, 8),
                "scale": (4.0, 4.0, 4.0),
            }
        }

        target_arrays = {
            "gt": {
                "shape": (8, 8, 8),
                "scale": (4.0, 4.0, 4.0),
            }
        }

        dataset = CellMapDataset(
            raw_path=config["raw_path"],
            target_path=config["gt_path"],
            classes=config["classes"],
            input_arrays=input_arrays,
            target_arrays=target_arrays,
            is_train=True,
            force_has_data=True,
        )

        assert dataset.raw_path == config["raw_path"]
        assert dataset.classes == config["classes"]
        assert dataset.is_train is True
        assert len(dataset.classes) == 2

    def test_initialization_without_classes(self, minimal_dataset_config):
        """Test dataset initialization without classes (raw data only)."""
        config = minimal_dataset_config

        input_arrays = {
            "raw": {
                "shape": (8, 8, 8),
                "scale": (4.0, 4.0, 4.0),
            }
        }

        dataset = CellMapDataset(
            raw_path=config["raw_path"],
            target_path=config["gt_path"],
            classes=None,
            input_arrays=input_arrays,
            is_train=False,
            force_has_data=True,
        )

        assert dataset.raw_only is True
        assert dataset.classes == []

    def test_input_arrays_configuration(self, minimal_dataset_config):
        """Test input arrays configuration."""
        config = minimal_dataset_config

        input_arrays = {
            "raw_4nm": {
                "shape": (16, 16, 16),
                "scale": (4.0, 4.0, 4.0),
            },
            "raw_8nm": {
                "shape": (8, 8, 8),
                "scale": (8.0, 8.0, 8.0),
            },
        }

        target_arrays = {
            "gt": {
                "shape": (8, 8, 8),
                "scale": (8.0, 8.0, 8.0),
            }
        }

        dataset = CellMapDataset(
            raw_path=config["raw_path"],
            target_path=config["gt_path"],
            classes=config["classes"],
            input_arrays=input_arrays,
            target_arrays=target_arrays,
        )

        assert "raw_4nm" in dataset.input_arrays
        assert "raw_8nm" in dataset.input_arrays
        assert dataset.input_arrays["raw_4nm"]["shape"] == (16, 16, 16)

    def test_target_arrays_configuration(self, minimal_dataset_config):
        """Test target arrays configuration."""
        config = minimal_dataset_config

        input_arrays = {
            "raw": {
                "shape": (8, 8, 8),
                "scale": (4.0, 4.0, 4.0),
            }
        }

        target_arrays = {
            "labels": {
                "shape": (8, 8, 8),
                "scale": (4.0, 4.0, 4.0),
            },
            "distances": {
                "shape": (8, 8, 8),
                "scale": (4.0, 4.0, 4.0),
            },
        }

        dataset = CellMapDataset(
            raw_path=config["raw_path"],
            target_path=config["gt_path"],
            classes=config["classes"],
            input_arrays=input_arrays,
            target_arrays=target_arrays,
        )

        assert "labels" in dataset.target_arrays
        assert "distances" in dataset.target_arrays

    def test_spatial_transforms_configuration(self, minimal_dataset_config):
        """Test spatial transforms configuration."""
        config = minimal_dataset_config

        input_arrays = {"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}}
        target_arrays = {"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}}

        spatial_transforms = {
            "mirror": {"axes": {"x": 0.5, "y": 0.5, "z": 0.2}},
            "rotate": {"axes": {"z": [-30, 30]}},
            "transpose": {"axes": ["x", "y"]},
        }

        dataset = CellMapDataset(
            raw_path=config["raw_path"],
            target_path=config["gt_path"],
            classes=config["classes"],
            input_arrays=input_arrays,
            target_arrays=target_arrays,
            spatial_transforms=spatial_transforms,
            is_train=True,
            force_has_data=True,
        )

        assert dataset.spatial_transforms is not None
        assert "mirror" in dataset.spatial_transforms
        assert "rotate" in dataset.spatial_transforms

    def test_value_transforms_configuration(self, minimal_dataset_config):
        """Test value transforms configuration."""
        config = minimal_dataset_config

        input_arrays = {"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}}
        target_arrays = {"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}}

        raw_transforms = T.Compose(
            [
                T.ToDtype(torch.float, scale=True),
            ]
        )

        target_transforms = T.Compose(
            [
                Binarize(threshold=0.5),
            ]
        )

        dataset = CellMapDataset(
            raw_path=config["raw_path"],
            target_path=config["gt_path"],
            classes=config["classes"],
            input_arrays=input_arrays,
            target_arrays=target_arrays,
            raw_value_transforms=raw_transforms,
            target_value_transforms=target_transforms,
        )

        assert dataset.raw_value_transforms is not None
        assert dataset.target_value_transforms is not None

    def test_class_relation_dict(self, minimal_dataset_config):
        """Test class relationship dictionary."""
        config = minimal_dataset_config

        input_arrays = {"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}}
        target_arrays = {"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}}

        class_relation_dict = {
            "class_0": ["class_1"],
            "class_1": ["class_0"],
        }

        dataset = CellMapDataset(
            raw_path=config["raw_path"],
            target_path=config["gt_path"],
            classes=config["classes"],
            input_arrays=input_arrays,
            target_arrays=target_arrays,
            class_relation_dict=class_relation_dict,
        )

        assert dataset.class_relation_dict is not None
        assert "class_0" in dataset.class_relation_dict

    def test_axis_order_parameter(self, minimal_dataset_config):
        """Test different axis orders."""
        config = minimal_dataset_config

        input_arrays = {"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}}
        target_arrays = {"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}}

        for axis_order in ["zyx", "xyz", "yxz"]:
            dataset = CellMapDataset(
                raw_path=config["raw_path"],
                target_path=config["gt_path"],
                classes=config["classes"],
                input_arrays=input_arrays,
                target_arrays=target_arrays,
                axis_order=axis_order,
            )
            assert dataset.axis_order == axis_order

    def test_is_train_parameter(self, minimal_dataset_config):
        """Test is_train parameter."""
        config = minimal_dataset_config

        input_arrays = {"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}}
        target_arrays = {"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}}

        # Training dataset
        train_dataset = CellMapDataset(
            raw_path=config["raw_path"],
            target_path=config["gt_path"],
            classes=config["classes"],
            input_arrays=input_arrays,
            target_arrays=target_arrays,
            is_train=True,
            force_has_data=True,
        )
        assert train_dataset.is_train is True

        # Validation dataset
        val_dataset = CellMapDataset(
            raw_path=config["raw_path"],
            target_path=config["gt_path"],
            classes=config["classes"],
            input_arrays=input_arrays,
            target_arrays=target_arrays,
            is_train=False,
            force_has_data=True,
        )
        assert val_dataset.is_train is False

    def test_pad_parameter(self, minimal_dataset_config):
        """Test pad parameter."""
        config = minimal_dataset_config

        input_arrays = {"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}}
        target_arrays = {"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}}

        # With padding
        dataset_pad = CellMapDataset(
            raw_path=config["raw_path"],
            target_path=config["gt_path"],
            classes=config["classes"],
            input_arrays=input_arrays,
            target_arrays=target_arrays,
            pad=True,
        )
        assert dataset_pad.pad is True

        # Without padding
        dataset_no_pad = CellMapDataset(
            raw_path=config["raw_path"],
            target_path=config["gt_path"],
            classes=config["classes"],
            input_arrays=input_arrays,
            target_arrays=target_arrays,
            pad=False,
        )
        assert dataset_no_pad.pad is False

    def test_empty_value_parameter(self, minimal_dataset_config):
        """Test empty_value parameter."""
        config = minimal_dataset_config

        input_arrays = {"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}}
        target_arrays = {"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}}

        # Test with NaN
        dataset_nan = CellMapDataset(
            raw_path=config["raw_path"],
            target_path=config["gt_path"],
            classes=config["classes"],
            input_arrays=input_arrays,
            target_arrays=target_arrays,
            empty_value=torch.nan,
        )
        assert torch.isnan(torch.tensor(dataset_nan.empty_value))

        # Test with numeric value
        dataset_zero = CellMapDataset(
            raw_path=config["raw_path"],
            target_path=config["gt_path"],
            classes=config["classes"],
            input_arrays=input_arrays,
            target_arrays=target_arrays,
            empty_value=0.0,
        )
        assert dataset_zero.empty_value == 0.0

    def test_device_parameter(self, minimal_dataset_config):
        """Test device parameter."""
        config = minimal_dataset_config

        input_arrays = {"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}}
        target_arrays = {"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}}

        # CPU device
        dataset_cpu = CellMapDataset(
            raw_path=config["raw_path"],
            target_path=config["gt_path"],
            classes=config["classes"],
            input_arrays=input_arrays,
            target_arrays=target_arrays,
            device="cpu",
        )
        # Device should be set (exact value checked in image tests)
        assert dataset_cpu is not None

    def test_force_has_data_parameter(self, minimal_dataset_config):
        """Test force_has_data parameter."""
        config = minimal_dataset_config

        input_arrays = {"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}}
        target_arrays = {"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}}

        dataset = CellMapDataset(
            raw_path=config["raw_path"],
            target_path=config["gt_path"],
            classes=config["classes"],
            input_arrays=input_arrays,
            target_arrays=target_arrays,
            force_has_data=True,
        )

        assert dataset.force_has_data is True

    def test_rng_parameter(self, minimal_dataset_config):
        """Test random number generator parameter."""
        config = minimal_dataset_config

        input_arrays = {"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}}
        target_arrays = {"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}}

        # Create custom RNG
        rng = torch.Generator()
        rng.manual_seed(42)

        dataset = CellMapDataset(
            raw_path=config["raw_path"],
            target_path=config["gt_path"],
            classes=config["classes"],
            input_arrays=input_arrays,
            target_arrays=target_arrays,
            rng=rng,
        )

        assert dataset._rng is rng

    def test_context_parameter(self, minimal_dataset_config):
        """Test TensorStore context parameter."""
        import tensorstore as ts

        config = minimal_dataset_config

        input_arrays = {"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}}
        target_arrays = {"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}}

        context = ts.Context()

        dataset = CellMapDataset(
            raw_path=config["raw_path"],
            target_path=config["gt_path"],
            classes=config["classes"],
            input_arrays=input_arrays,
            target_arrays=target_arrays,
            context=context,
        )

        assert dataset.context is context

    def test_max_workers_parameter(self, minimal_dataset_config):
        """Test max_workers parameter."""
        config = minimal_dataset_config

        input_arrays = {"raw": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}}
        target_arrays = {"gt": {"shape": (8, 8, 8), "scale": (4.0, 4.0, 4.0)}}

        dataset = CellMapDataset(
            raw_path=config["raw_path"],
            target_path=config["gt_path"],
            classes=config["classes"],
            input_arrays=input_arrays,
            target_arrays=target_arrays,
            max_workers=4,
        )

        # Dataset should be created successfully
        assert dataset is not None

    def test_empty_dataset_creation(self):
        """Test CellMapDataset.empty() static method."""
        from cellmap_data import CellMapDataset

        # Create an empty dataset
        empty_dataset = CellMapDataset.empty()

        # Verify basic properties
        assert empty_dataset is not None
        assert isinstance(empty_dataset, CellMapDataset)
        assert empty_dataset.has_data is False
        assert len(empty_dataset) == 0

        # Verify the newly initialized attributes
        assert hasattr(empty_dataset, "sampling_box_shape")
        assert isinstance(empty_dataset.sampling_box_shape, dict)
        assert all(v == 0 for v in empty_dataset.sampling_box_shape.values())

        # Verify axis_order is set (should have default value)
        assert hasattr(empty_dataset, "axis_order")
        assert len(empty_dataset.sampling_box_shape) == len(empty_dataset.axis_order)

    def test_empty_datasetsampling_box_shape(self):
        """Test that empty dataset has correct sampling_box_shape initialization."""
        from cellmap_data import CellMapDataset

        empty_dataset = CellMapDataset.empty()

        # Verify sampling_box_shape keys match axis_order
        assert set(empty_dataset.sampling_box_shape.keys()) == set(
            empty_dataset.axis_order
        )

        # Verify all dimensions are 0
        for axis in empty_dataset.axis_order:
            assert empty_dataset.sampling_box_shape[axis] == 0

    def test_bounding_box_shape(self, minimal_dataset_config):
        """Test that bounding_box_shape property correctly computes shape from bounding box."""
        config = minimal_dataset_config

        input_arrays = {
            "raw": {
                "shape": (8, 8, 8),
                "scale": (4.0, 4.0, 4.0),
            }
        }

        target_arrays = {
            "gt": {
                "shape": (8, 8, 8),
                "scale": (4.0, 4.0, 4.0),
            }
        }

        dataset = CellMapDataset(
            raw_path=config["raw_path"],
            target_path=config["gt_path"],
            classes=config["classes"],
            input_arrays=input_arrays,
            target_arrays=target_arrays,
            force_has_data=True,
        )

        # Access bounding_box_shape
        bbox_shape = dataset.bounding_box_shape

        # Verify it's a dict with expected keys
        assert isinstance(bbox_shape, dict)
        assert set(bbox_shape.keys()) == set(dataset.axis_order)

        # Verify all values are positive integers
        for axis, size in bbox_shape.items():
            assert isinstance(size, (int, float))
            assert size > 0

    def test_size_property(self, minimal_dataset_config):
        """Test that size property correctly computes dataset size from bounding box.

        This test ensures the bug fix in PR #61 is covered: size must use
        .values() not .items() to properly unpack bounding box numeric bounds.
        """
        config = minimal_dataset_config

        input_arrays = {
            "raw": {
                "shape": (8, 8, 8),
                "scale": (4.0, 4.0, 4.0),
            }
        }

        target_arrays = {
            "gt": {
                "shape": (8, 8, 8),
                "scale": (4.0, 4.0, 4.0),
            }
        }

        dataset = CellMapDataset(
            raw_path=config["raw_path"],
            target_path=config["gt_path"],
            classes=config["classes"],
            input_arrays=input_arrays,
            target_arrays=target_arrays,
            force_has_data=True,
        )

        # Access the size property - this would have raised TypeError before the fix
        size = dataset.size

        # Verify it's a positive integer
        assert isinstance(size, int)
        assert size > 0

        # Verify it matches the product of bounding box dimensions
        import numpy as np

        expected_size = int(
            np.prod([stop - start for start, stop in dataset.bounding_box.values()])
        )
        assert size == expected_size

    def test_size_property_with_known_dimensions(self, tmp_path):
        """Test size property with specific known bounding box dimensions."""
        # Create a dataset with known dimensions
        config = create_test_dataset(
            tmp_path,
            raw_shape=(40, 30, 20),  # Known shape
            num_classes=2,
            raw_scale=(4.0, 4.0, 4.0),
        )

        input_arrays = {
            "raw": {
                "shape": (8, 8, 8),
                "scale": (4.0, 4.0, 4.0),
            }
        }

        target_arrays = {
            "gt": {
                "shape": (8, 8, 8),
                "scale": (4.0, 4.0, 4.0),
            }
        }

        dataset = CellMapDataset(
            raw_path=config["raw_path"],
            target_path=config["gt_path"],
            classes=config["classes"],
            input_arrays=input_arrays,
            target_arrays=target_arrays,
            force_has_data=True,
        )

        # Get the size
        size = dataset.size

        # Verify it's the product of the bounding box dimensions
        # The bounding box should correspond to the raw data shape
        bbox = dataset.bounding_box
        import numpy as np

        # Calculate expected size from bounding box
        dims = [stop - start for start, stop in bbox.values()]
        expected_size = int(np.prod(dims))

        assert size == expected_size
        assert size > 0
