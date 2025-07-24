import torch
from cellmap_data.dataloader import CellMapDataLoader


def test_dataloader_basic(dummy_dataset):
    loader = CellMapDataLoader(dummy_dataset, batch_size=2, num_workers=0)
    batch = next(iter(loader.loader))
    assert "x" in batch and "y" in batch
    assert batch["x"].shape[0] == 2
    assert batch["x"].device.type == loader.device


def test_dataloader_to_device(dummy_dataset):
    loader = CellMapDataLoader(dummy_dataset, batch_size=2, num_workers=0)
    loader.to("cpu")
    assert loader.device == "cpu"


def test_dataloader_refresh(dummy_dataset):
    loader = CellMapDataLoader(dummy_dataset, batch_size=2, num_workers=0)
    loader.refresh()
    batch = next(iter(loader.loader))
    assert batch["x"].shape[0] == 2


def test_memory_calculation_accuracy(dummy_dataset):
    """Test that memory calculation in CellMapDataLoader is accurate."""
    loader = CellMapDataLoader(dummy_dataset, batch_size=4, num_workers=0, device="cpu")

    # Calculate memory
    memory_mb = loader._calculate_batch_memory_mb()

    # Manual verification
    batch_size = 4
    input_elements = batch_size * dummy_dataset.num_features
    target_elements = batch_size * 1  # target is a single value
    total_elements = input_elements + target_elements
    expected_mb = (total_elements * 4) / (1024 * 1024)  # float32 = 4 bytes

    assert (
        abs(memory_mb - expected_mb) < 0.01
    ), f"Memory calculation mismatch: {memory_mb:.3f} vs {expected_mb:.3f}"


def test_memory_calculation_edge_cases(empty_mock_dataset):
    """Test memory calculation with an empty dataset."""
    # Test with an empty dataset
    loader = CellMapDataLoader(empty_mock_dataset, batch_size=1)
    assert loader._calculate_batch_memory_mb() == 0
