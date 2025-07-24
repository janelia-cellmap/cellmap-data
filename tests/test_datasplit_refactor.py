import pytest
from unittest.mock import MagicMock, patch
from cellmap_data.datasplit import CellMapDataSplit
from cellmap_data.dataset_factory import DatasetFactory
from cellmap_data.data_split_config_manager import DataSplitConfigManager


@pytest.fixture
def mock_config_manager():
    with patch("cellmap_data.datasplit.DataSplitConfigManager") as mock:
        instance = mock.return_value
        instance.config = {
            "dataset_dict": {"train": [{"raw": "dummy_raw", "gt": "dummy_gt"}]},
            "csv_path": None,
            "force_has_data": False,
            "spatial_transforms": None,
            "train_raw_value_transforms": None,
            "val_raw_value_transforms": None,
            "target_value_transforms": None,
            "context": None,
            "empty_value": 0,
            "class_relation_dict": None,
            "pad": 0,
            "device": "cpu",
        }
        instance.input_arrays = {}
        instance.target_arrays = {}
        instance.classes = []
        yield mock


@pytest.fixture
def mock_dataset_factory():
    with patch("cellmap_data.datasplit.DatasetFactory") as mock:
        yield mock


@pytest.fixture
def mock_dataset():
    mock_ds = MagicMock()
    mock_ds.has_data = True
    mock_ds.raw_path = "dummy_path"
    return mock_ds


def test_datasplit_initialization(
    mock_config_manager, mock_dataset_factory, mock_dataset
):
    # Arrange
    mock_config_manager.return_value.config["dataset_dict"] = {
        "train": [{"raw": "dummy_raw", "gt": "dummy_gt"}]
    }

    with patch(
        "cellmap_data.datasplit.CellMapDataSplit.get_datasets_from_dict"
    ) as mock_get_datasets:
        mock_get_datasets.return_value = {"train": [mock_dataset], "validate": []}

        # Act
        datasplit = CellMapDataSplit(
            input_arrays={},
            target_arrays={},
            classes=[],
            empty_value=0,
            pad=0,
            dataset_dict={"train": [{"raw": "dummy_raw", "gt": "dummy_gt"}]},
        )

        # Assert
        assert datasplit._config_manager is not None
        assert datasplit._dataset_factory is not None
        mock_config_manager.assert_called_once()
        mock_dataset_factory.assert_called_once()
        mock_get_datasets.assert_called_once()
        assert datasplit.train_datasets == [mock_dataset]
        assert datasplit.validation_datasets == []


def test_datasplit_uses_factory_to_construct_multidataset(
    mock_config_manager, mock_dataset_factory, mock_dataset
):
    # Arrange
    with patch(
        "cellmap_data.datasplit.CellMapDataSplit.get_datasets_from_dict"
    ) as mock_get_datasets:
        mock_get_datasets.return_value = {"train": [mock_dataset], "validate": []}

        datasplit = CellMapDataSplit(
            input_arrays={},
            target_arrays={},
            classes=[],
            empty_value=0,
            pad=0,
            dataset_dict={"train": [{"raw": "dummy_raw", "gt": "dummy_gt"}]},
        )

        factory_instance = mock_dataset_factory.return_value

        # Act
        datasplit.get_train_dataset()

        # Assert
        factory_instance.create_multidataset.assert_called_once()


def test_datasplit_backward_compatibility_attributes(
    mock_config_manager, mock_dataset_factory
):
    # Arrange
    with patch(
        "cellmap_data.datasplit.CellMapDataSplit.get_datasets_from_dict"
    ) as mock_get_datasets:
        mock_get_datasets.return_value = {"train": [MagicMock()], "validate": []}

        # Act
        datasplit = CellMapDataSplit(
            input_arrays={"test_in": {}},
            target_arrays={"test_out": {}},
            classes=["a", "b"],
            empty_value=1,
            pad=5,
            dataset_dict={"train": [{"raw": "dummy_raw", "gt": "dummy_gt"}]},
        )

        # Assert
        assert hasattr(datasplit, "force_has_data")
        assert hasattr(datasplit, "context")
        assert datasplit.input_arrays == {"test_in": {}}
        assert datasplit.target_arrays == {"test_out": {}}
        assert datasplit.classes == ["a", "b"]
