import pytest
from cellmap_data.validation.validation import ConfigValidator
from cellmap_data.validation.schemas import DatasetConfig, DataLoaderConfig
from unittest.mock import MagicMock


# Minimal valid dataset config
@pytest.fixture
def valid_dataset_config():
    return {
        "raw_path": "raw.zarr",
        "target_path": "target.zarr",
        "classes": ["class1"],
        "input_arrays": {"s0": {"shape": (1, 64, 64, 64), "scale": (1, 1, 1, 1)}},
    }


# Minimal valid dataloader config
@pytest.fixture
def valid_dataloader_config():
    return {
        "dataset": MagicMock(),
        "batch_size": 4,
        "num_workers": 2,
    }


def test_valid_dataset_config(valid_dataset_config):
    assert ConfigValidator.validate_dataset_config(valid_dataset_config)


def test_invalid_dataset_config_missing_key(valid_dataset_config):
    del valid_dataset_config["raw_path"]
    assert not ConfigValidator.validate_dataset_config(valid_dataset_config)


def test_invalid_dataset_config_bad_type(valid_dataset_config):
    valid_dataset_config["classes"] = "not-a-list"
    assert not ConfigValidator.validate_dataset_config(valid_dataset_config)


def test_valid_dataloader_config(valid_dataloader_config):
    assert ConfigValidator.validate_dataloader_config(valid_dataloader_config)


def test_invalid_dataloader_config_missing_key(valid_dataloader_config):
    del valid_dataloader_config["dataset"]
    assert not ConfigValidator.validate_dataloader_config(valid_dataloader_config)


def test_invalid_dataloader_config_bad_type(valid_dataloader_config):
    valid_dataloader_config["batch_size"] = "not-an-int"
    assert not ConfigValidator.validate_dataloader_config(valid_dataloader_config)


def test_invalid_dataloader_config_bad_value(valid_dataloader_config):
    valid_dataloader_config["num_workers"] = -1
    assert not ConfigValidator.validate_dataloader_config(valid_dataloader_config)


def test_dataset_config_pydantic_model(valid_dataset_config):
    config = DatasetConfig(**valid_dataset_config)
    assert config.raw_path == "raw.zarr"
    assert config.is_train is False


def test_dataloader_config_pydantic_model(valid_dataloader_config):
    config = DataLoaderConfig(**valid_dataloader_config)
    assert config.batch_size == 4
    assert config.shuffle is False
