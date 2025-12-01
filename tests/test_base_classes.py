"""Tests for base abstract classes."""

from abc import ABC

import pytest
import torch

from cellmap_data.base_dataset import CellMapBaseDataset
from cellmap_data.base_image import CellMapImageBase


class TestCellMapBaseDataset:
    """Test the CellMapBaseDataset abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that CellMapBaseDataset cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            CellMapBaseDataset()

    def test_incomplete_implementation_raises_error(self):
        """Test that incomplete implementations cannot be instantiated."""

        # Missing all abstract methods
        class IncompleteDataset(CellMapBaseDataset):
            pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteDataset()

        # Missing some abstract methods
        class PartialDataset(CellMapBaseDataset):
            @property
            def class_counts(self):
                return {}

            @property
            def class_weights(self):
                return {}

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            PartialDataset()

    def test_complete_implementation_can_be_instantiated(self):
        """Test that complete implementations can be instantiated."""

        class CompleteDataset(CellMapBaseDataset):
            def __init__(self):
                self.classes = ["class1", "class2"]
                self.input_arrays = {"raw": {}}
                self.target_arrays = {"labels": {}}

            @property
            def class_counts(self):
                return {"class1": 100.0, "class2": 200.0}

            @property
            def class_weights(self):
                return {"class1": 0.67, "class2": 0.33}

            @property
            def validation_indices(self):
                return [0, 1, 2]

            def to(self, device, non_blocking=True):
                return self

            def set_raw_value_transforms(self, transforms):
                pass

            def set_target_value_transforms(self, transforms):
                pass

        # Should not raise
        dataset = CompleteDataset()
        assert isinstance(dataset, CellMapBaseDataset)
        assert dataset.classes == ["class1", "class2"]
        assert dataset.class_counts == {"class1": 100.0, "class2": 200.0}
        assert dataset.class_weights == {"class1": 0.67, "class2": 0.33}
        assert dataset.validation_indices == [0, 1, 2]
        assert dataset.to("cpu") is dataset
        dataset.set_raw_value_transforms(lambda x: x)
        dataset.set_target_value_transforms(lambda x: x)

    def test_attributes_are_defined(self):
        """Test that expected attributes are defined in the base class."""
        # Check type annotations exist
        assert hasattr(CellMapBaseDataset, "__annotations__")
        annotations = CellMapBaseDataset.__annotations__
        assert "classes" in annotations
        assert "input_arrays" in annotations
        assert "target_arrays" in annotations


class TestCellMapImageBase:
    """Test the CellMapImageBase abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that CellMapImageBase cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            CellMapImageBase()

    def test_incomplete_implementation_raises_error(self):
        """Test that incomplete implementations cannot be instantiated."""

        # Missing all abstract methods
        class IncompleteImage(CellMapImageBase):
            pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteImage()

        # Missing some abstract methods
        class PartialImage(CellMapImageBase):
            @property
            def bounding_box(self):
                return {"x": (0, 100), "y": (0, 100)}

            @property
            def sampling_box(self):
                return {"x": (10, 90), "y": (10, 90)}

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            PartialImage()

    def test_complete_implementation_can_be_instantiated(self):
        """Test that complete implementations can be instantiated."""

        class CompleteImage(CellMapImageBase):
            def __getitem__(self, center):
                return torch.zeros((1, 64, 64))

            @property
            def bounding_box(self):
                return {"x": (0.0, 100.0), "y": (0.0, 100.0)}

            @property
            def sampling_box(self):
                return {"x": (10.0, 90.0), "y": (10.0, 90.0)}

            @property
            def class_counts(self):
                return 1000.0

            def to(self, device, non_blocking=True):
                pass

            def set_spatial_transforms(self, transforms):
                pass

        # Should not raise
        image = CompleteImage()
        assert isinstance(image, CellMapImageBase)
        center = {"x": 50.0, "y": 50.0}
        result = image[center]
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 64, 64)
        assert image.bounding_box == {"x": (0.0, 100.0), "y": (0.0, 100.0)}
        assert image.sampling_box == {"x": (10.0, 90.0), "y": (10.0, 90.0)}
        assert image.class_counts == 1000.0
        image.to("cpu")
        image.set_spatial_transforms(None)

    def test_class_counts_supports_dict_return_type(self):
        """Test that class_counts can return a dictionary."""

        class MultiClassImage(CellMapImageBase):
            def __getitem__(self, center):
                return torch.zeros((1, 64, 64))

            @property
            def bounding_box(self):
                return {"x": (0.0, 100.0)}

            @property
            def sampling_box(self):
                return {"x": (10.0, 90.0)}

            @property
            def class_counts(self):
                return {"class1": 500.0, "class2": 300.0, "class3": 200.0}

            def to(self, device, non_blocking=True):
                pass

            def set_spatial_transforms(self, transforms):
                pass

        image = MultiClassImage()
        counts = image.class_counts
        assert isinstance(counts, dict)
        assert counts == {"class1": 500.0, "class2": 300.0, "class3": 200.0}

    def test_bounding_box_can_be_none(self):
        """Test that bounding_box property can return None."""

        class UnboundedImage(CellMapImageBase):
            def __getitem__(self, center):
                return torch.zeros((1, 64, 64))

            @property
            def bounding_box(self):
                return None

            @property
            def sampling_box(self):
                return None

            @property
            def class_counts(self):
                return 1000.0

            def to(self, device, non_blocking=True):
                pass

            def set_spatial_transforms(self, transforms):
                pass

        image = UnboundedImage()
        assert image.bounding_box is None
        assert image.sampling_box is None
