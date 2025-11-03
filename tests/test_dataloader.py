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
    """
    Test that PyTorch DataLoader handles memory optimization correctly.
    
    This test verifies that the dataloader uses pin_memory and prefetch_factor
    for optimized GPU transfer, replacing the old custom memory calculation.
    """

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
    loader = CellMapDataLoader(mock_dataset, batch_size=4, num_workers=2, device="cpu")

    # Verify PyTorch DataLoader optimization settings
    assert loader._pytorch_loader is not None, "PyTorch loader should be initialized"
    assert loader._prefetch_factor == 2, "prefetch_factor should be set to default 2"
    
    # Test that batches can be loaded successfully
    batch = next(iter(loader))
    assert "input1" in batch and "input2" in batch and "target1" in batch
    assert batch["input1"].shape[0] == 4, "Batch should have correct size"


def test_memory_calculation_edge_cases():
    """Test that PyTorch DataLoader handles edge cases gracefully."""
    # This test verifies that the dataloader can handle minimal/empty datasets
    # PyTorch's DataLoader is robust and handles these cases automatically

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
            return {"empty": torch.tensor([idx])}

        def to(self, device, non_blocking=True):
            pass

    empty_dataset = EmptyMockDataset()
    loader = CellMapDataLoader(empty_dataset, batch_size=1, num_workers=0, device="cpu")
    
    # Verify loader can handle empty dataset configuration
    assert loader._pytorch_loader is not None, "PyTorch loader should be initialized"
    
    # Verify we can iterate over the dataset
    batch = next(iter(loader))
    assert "empty" in batch, "Should handle minimal dataset"
    assert batch["empty"].shape[0] == 1, "Should have correct batch size"


def test_pin_memory_parameter():
    """Test that pin_memory parameter works correctly."""

    class CPUDataset:
        def __init__(self, length=4):
            self.length = length
            self.classes = ["a", "b"]

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            # Return CPU tensors to test pin_memory
            return {
                "x": torch.randn(2, 4),
                "y": torch.tensor(idx % 2),
            }

        def to(self, device, non_blocking=True):
            pass

    dataset = CPUDataset()

    # Test pin_memory=False (default)
    loader_no_pin = CellMapDataLoader(
        dataset, batch_size=2, pin_memory=False, device="cpu", num_workers=0
    )
    batch_no_pin = next(iter(loader_no_pin))
    assert not batch_no_pin[
        "x"
    ].is_pinned(), "Tensor should not be pinned when pin_memory=False"

    # Test pin_memory=True
    loader_pin = CellMapDataLoader(
        dataset, batch_size=2, pin_memory=True, device="cpu", num_workers=0
    )
    batch_pin = next(iter(loader_pin))
    assert batch_pin["x"].is_pinned(), "Tensor should be pinned when pin_memory=True"

    # Additional check: if CUDA is available, verify pinned tensor can be moved to GPU
    if torch.cuda.is_available():
        try:
            gpu_tensor = batch_pin["x"].to("cuda", non_blocking=True)
            assert gpu_tensor.device.type == "cuda", "Tensor should be on CUDA device"
        except Exception as e:
            assert False, f"Failed to move pinned tensor to CUDA: {e}"

    # Verify pin_memory setting is stored correctly
    assert not loader_no_pin._pin_memory, "pin_memory flag should be False"
    assert loader_pin._pin_memory, "pin_memory flag should be True"


def test_drop_last_parameter():
    """Test that drop_last parameter works correctly."""
    dataset = DummyDataset(length=13)  # 13 samples, odd number to test drop_last
    batch_size = 4

    # Test drop_last=False (default) - should include incomplete final batch
    loader_no_drop = CellMapDataLoader(
        dataset, batch_size=batch_size, drop_last=False, num_workers=0
    )
    expected_batches_no_drop = (
        len(dataset) + batch_size - 1
    ) // batch_size  # Ceiling division
    assert (
        len(loader_no_drop) == expected_batches_no_drop
    ), f"Expected {expected_batches_no_drop} batches with drop_last=False"

    batches_no_drop = list(loader_no_drop)
    assert (
        len(batches_no_drop) == expected_batches_no_drop
    ), "Should generate expected number of batches"
    assert (
        len(batches_no_drop[-1]["x"]) == 1
    ), "Final batch should have 1 sample (13 % 4 = 1)"

    # Test drop_last=True - should drop incomplete final batch
    loader_drop = CellMapDataLoader(
        dataset, batch_size=batch_size, drop_last=True, num_workers=0
    )
    expected_batches_drop = len(dataset) // batch_size  # Floor division
    assert (
        len(loader_drop) == expected_batches_drop
    ), f"Expected {expected_batches_drop} batches with drop_last=True"

    batches_drop = list(loader_drop)
    assert (
        len(batches_drop) == expected_batches_drop
    ), "Should generate expected number of batches"
    for batch in batches_drop:
        assert (
            len(batch["x"]) == batch_size
        ), "All batches should have exactly batch_size samples"

    # Verify drop_last setting is stored correctly
    assert not loader_no_drop._drop_last, "drop_last flag should be False"
    assert loader_drop._drop_last, "drop_last flag should be True"


def test_persistent_workers_parameter():
    """Test that persistent_workers parameter works correctly."""
    dataset = DummyDataset(length=8)

    # Test persistent_workers=False
    loader_no_persist = CellMapDataLoader(
        dataset, batch_size=2, persistent_workers=False, num_workers=2
    )
    assert (
        not loader_no_persist._persistent_workers
    ), "persistent_workers flag should be False"

    # Get a batch to initialize workers
    batch1 = next(iter(loader_no_persist))
    assert batch1["x"].shape[0] == 2, "Batch should have correct size"

    # Test persistent_workers=True - workers should persist with PyTorch DataLoader
    loader_persist = CellMapDataLoader(
        dataset, batch_size=2, persistent_workers=True, num_workers=2
    )
    assert loader_persist._persistent_workers, "persistent_workers flag should be True"

    # Get batches to verify workers persist - PyTorch manages worker lifecycle
    batch1 = next(iter(loader_persist))
    pytorch_loader_1 = loader_persist._pytorch_loader

    batch2 = next(iter(loader_persist))
    pytorch_loader_2 = loader_persist._pytorch_loader

    # PyTorch loader should be the same object (persistent between batches in same epoch)
    assert (
        pytorch_loader_1 is pytorch_loader_2
    ), "PyTorch loader should persist between iterations"
    assert pytorch_loader_1 is not None, "PyTorch loader should exist"


def test_pytorch_dataloader_compatibility():
    """Test that other PyTorch DataLoader parameters are accepted and stored."""
    dataset = DummyDataset()

    # Test various PyTorch DataLoader parameters
    loader = CellMapDataLoader(
        dataset,
        batch_size=2,
        timeout=30,
        prefetch_factor=3,
        worker_init_fn=None,
        generator=None,
        num_workers=0,
    )

    # Verify parameters are stored in default_kwargs for compatibility
    assert "timeout" in loader.default_kwargs, "timeout should be stored"
    assert (
        "prefetch_factor" in loader.default_kwargs
    ), "prefetch_factor should be stored"
    assert "worker_init_fn" in loader.default_kwargs, "worker_init_fn should be stored"
    assert "generator" in loader.default_kwargs, "generator should be stored"

    assert loader.default_kwargs["timeout"] == 30, "timeout value should be correct"
    assert (
        loader.default_kwargs["prefetch_factor"] == 3
    ), "prefetch_factor value should be correct"

    # Should still work normally
    batch = next(iter(loader))
    assert (
        batch["x"].shape[0] == 2
    ), "Dataloader should work with compatibility parameters"


def test_combined_pytorch_parameters():
    """Test that multiple PyTorch DataLoader parameters work together."""
    dataset = DummyDataset(length=10)

    # Test combination of implemented parameters
    loader = CellMapDataLoader(
        dataset,
        batch_size=3,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
        num_workers=2,
        device="cpu",
    )

    # Verify all settings
    assert loader._pin_memory, "pin_memory should be True"
    assert loader._persistent_workers, "persistent_workers should be True"
    assert loader._drop_last, "drop_last should be True"
    assert loader.num_workers == 2, "num_workers should be 2"

    # Verify behavior
    expected_batches = len(dataset) // 3  # drop_last=True
    assert (
        len(loader) == expected_batches
    ), "Should calculate correct number of batches with drop_last=True"

    batches = list(loader)
    assert len(batches) == expected_batches, "Should generate correct number of batches"

    for batch in batches:
        assert len(batch["x"]) == 3, "All batches should have exactly 3 samples"
        assert batch["x"].is_pinned(), "Tensors should be pinned"


def test_direct_iteration_support():
    """Test that the dataloader supports direct iteration (new feature)."""
    dataset = DummyDataset(length=6)
    loader = CellMapDataLoader(dataset, batch_size=2, num_workers=0)

    # Test direct iteration (new feature)
    batches_direct = []
    for batch in loader:
        batches_direct.append(batch)
        assert "x" in batch and "y" in batch, "Batch should contain expected keys"
        assert batch["x"].shape[0] == 2, "Batch should have correct size"

    assert (
        len(batches_direct) == 3
    ), "Should generate 3 batches for 6 samples with batch_size=2"

    # Test backward compatibility - iter(loader.loader) should still work
    batches_compat = []
    for batch in loader.loader:
        batches_compat.append(batch)
        assert "x" in batch and "y" in batch, "Batch should contain expected keys"
        assert batch["x"].shape[0] == 2, "Batch should have correct size"

    assert len(batches_compat) == 3, "Backward compatibility iteration should work"


def test_length_calculation_with_drop_last():
    """Test that __len__ correctly accounts for drop_last parameter."""
    dataset = DummyDataset(length=10)

    # Test with drop_last=False
    loader_no_drop = CellMapDataLoader(
        dataset, batch_size=3, drop_last=False, num_workers=0
    )
    expected_no_drop = (10 + 3 - 1) // 3  # Ceiling division: 4 batches
    assert (
        len(loader_no_drop) == expected_no_drop
    ), f"Expected {expected_no_drop} batches with drop_last=False"

    # Test with drop_last=True
    loader_drop = CellMapDataLoader(
        dataset, batch_size=3, drop_last=True, num_workers=0
    )
    expected_drop = 10 // 3  # Floor division: 3 batches
    assert (
        len(loader_drop) == expected_drop
    ), f"Expected {expected_drop} batches with drop_last=True"


def test_pin_memory_validation():
    """Test that pin_memory is properly validated for non-CUDA devices."""
    dataset = DummyDataset(length=8)
    
    # Test pin_memory with CPU device (should be set to False with warning)
    loader = CellMapDataLoader(
        dataset, 
        batch_size=2, 
        pin_memory=True,  # User explicitly sets True
        device="cpu",     # But device is CPU
        num_workers=0
    )
    # Should be automatically set to False for CPU device
    assert not loader._pin_memory, "pin_memory should be False for CPU device"


def test_prefetch_factor_validation():
    """Test that prefetch_factor is properly validated."""
    dataset = DummyDataset(length=8)
    
    # Test valid prefetch_factor
    loader = CellMapDataLoader(
        dataset, 
        batch_size=2, 
        num_workers=2,
        prefetch_factor=4
    )
    assert loader._prefetch_factor == 4, "prefetch_factor should be set correctly"
    
    # Test invalid prefetch_factor (negative)
    try:
        CellMapDataLoader(
            dataset,
            batch_size=2,
            num_workers=2,
            prefetch_factor=-1
        )
        assert False, "Should raise ValueError for negative prefetch_factor"
    except ValueError as e:
        assert "prefetch_factor must be a positive integer" in str(e)
    
    # Test invalid prefetch_factor (zero)
    try:
        CellMapDataLoader(
            dataset,
            batch_size=2,
            num_workers=2,
            prefetch_factor=0
        )
        assert False, "Should raise ValueError for zero prefetch_factor"
    except ValueError as e:
        assert "prefetch_factor must be a positive integer" in str(e)
    
    # Test prefetch_factor ignored when num_workers=0
    loader = CellMapDataLoader(
        dataset,
        batch_size=2,
        num_workers=0,
        prefetch_factor=4  # Should be ignored
    )
    assert loader._prefetch_factor is None, "prefetch_factor should be None when num_workers=0"
