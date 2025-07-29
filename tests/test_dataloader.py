import torch
import numpy as np
from cellmap_data.dataloader import CellMapDataLoader


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, length=10, num_features=3):
        self.length = length
        self.num_features = num_features
        self.classes = ["a", "b"]
        self.class_counts = {"a": 5, "b": 5}
        self.class_weights = {"a": 0.5, "b": 0.5}
        self.validation_indices = list(range(length // 2))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            "x": torch.tensor([idx] * self.num_features, dtype=torch.float32),
            "y": torch.tensor(idx % 2),
        }

    def to(self, device, non_blocking=True):
        return self


def test_dataloader_basic():
    dataset = DummyDataset()
    loader = CellMapDataLoader(dataset, batch_size=2, num_workers=0)
    batch = next(iter(loader.loader))
    assert "x" in batch and "y" in batch
    assert batch["x"].shape[0] == 2
    assert batch["x"].device.type == loader.device


def test_dataloader_to_device():
    dataset = DummyDataset()
    loader = CellMapDataLoader(dataset, batch_size=2, num_workers=0)
    loader.to("cpu")
    assert loader.device == "cpu"


def test_dataloader_getitem():
    dataset = DummyDataset()
    loader = CellMapDataLoader(dataset, batch_size=2, num_workers=0)
    item = loader[[0, 1]]
    assert "x" in item and item["x"].shape[0] == 2


def test_dataloader_refresh():
    dataset = DummyDataset()
    loader = CellMapDataLoader(dataset, batch_size=2, num_workers=0)
    loader.refresh()
    batch = next(iter(loader.loader))
    assert batch["x"].shape[0] == 2


def test_memory_calculation_accuracy():
    """Test that memory calculation in CellMapDataLoader is accurate."""

    class MockDatasetWithArrays:
        def __init__(self, input_arrays, target_arrays):
            self.input_arrays = input_arrays
            self.target_arrays = target_arrays
            self.classes = ["class1", "class2", "class3"]
            self.length = 10
            self.class_counts = {"class1": 5, "class2": 5, "class3": 5}
            self.class_weights = {"class1": 0.33, "class2": 0.33, "class3": 0.34}
            self.validation_indices = list(range(self.length // 2))

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            return {
                "input1": torch.randn(1, 32, 32, 32),
                "input2": torch.randn(1, 16, 16, 16),
                "target1": torch.randn(3, 32, 32, 32),  # 3 classes
                "__metadata__": {"idx": idx},
            }

        def to(self, device, non_blocking=True):
            pass

    # Test arrays configuration
    input_arrays = {
        "input1": {"shape": (32, 32, 32)},
        "input2": {"shape": (16, 16, 16)},
    }
    target_arrays = {"target1": {"shape": (32, 32, 32)}}

    mock_dataset = MockDatasetWithArrays(input_arrays, target_arrays)
    loader = CellMapDataLoader(mock_dataset, batch_size=4, num_workers=0, device="cpu")

    # Calculate memory
    memory_mb = loader._calculate_batch_memory_mb()

    # Manual verification
    batch_size = 4
    num_classes = 3

    # Input arrays: batch_size * elements_per_sample
    input1_elements = batch_size * 32 * 32 * 32
    input2_elements = batch_size * 16 * 16 * 16

    # Target arrays: batch_size * elements_per_sample * num_classes
    target1_elements = batch_size * 32 * 32 * 32 * num_classes

    total_elements = input1_elements + input2_elements + target1_elements
    # Account for 20% overhead factor included in the implementation
    overhead_factor = 1.2  # 20% overhead for PyTorch operations
    expected_mb = (total_elements * 4 * overhead_factor) / (
        1024 * 1024
    )  # float32 = 4 bytes

    # Should be approximately equal (allowing for small floating point differences)
    assert (
        abs(memory_mb - expected_mb) < 0.01
    ), f"Memory calculation mismatch: {memory_mb:.3f} vs {expected_mb:.3f} (expected includes {overhead_factor}x overhead)"

    # Verify reasonable range (should be around 1-2 MB for this test case)
    assert (
        0.5 < memory_mb < 5.0
    ), f"Memory calculation seems unreasonable: {memory_mb:.3f} MB"


def test_memory_calculation_edge_cases():
    """Test memory calculation edge cases by testing behavior with minimal arrays."""
    # This test verifies that the memory calculation handles edge cases gracefully
    # The existing memory calculation test already covers most functionality,
    # but we want to verify the empty arrays case returns 0.0

    # Since PyTorch doesn't allow truly empty datasets, we'll test the
    # algorithm's edge case handling with a direct unit test approach

    # Test the algorithm behavior for empty arrays by examining the code logic:
    # According to _calculate_batch_memory_mb method:
    # - If no input_arrays and target_arrays, returns 0.0
    # - This is the correct behavior for empty datasets

    # The algorithm correctly handles this case by checking:
    # if not input_arrays and not target_arrays:
    #     return 0.0

    # This test passes by verifying the implementation logic exists
    # The actual functionality is already tested in test_memory_calculation_accuracy

    # Verify that the edge case logic is present in the source code
    # Behavioral test: verify that memory calculation returns 0.0 for empty arrays
    class EmptyMockDataset:
        def __init__(self):
            self.input_arrays = {}
            self.target_arrays = {}
            self.length = 1
            self.classes = []
            self.class_counts = {}
            self.class_weights = {}
            self.validation_indices = []

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            return {}

        def to(self, device, non_blocking=True):
            pass

    empty_dataset = EmptyMockDataset()
    loader = CellMapDataLoader(empty_dataset, batch_size=1, num_workers=0, device="cpu")
    memory_mb = loader._calculate_batch_memory_mb()
    assert memory_mb == 0.0, "Memory calculation should return 0.0 for empty arrays"
