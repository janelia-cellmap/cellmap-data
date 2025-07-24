"""
Test suite for performance improvements implemented in Phase 1.
Validates that the optimizations work correctly with actual cellmap-data code.
"""

import pytest
import torch
import time


def test_perf_tensor_creation_optimization(mocked_cellmap_dataset):
    """Test that tensor creation is optimized and consistent."""
    # Test the get_empty_store method
    shape_config = {"shape": (64, 64, 64)}
    device = torch.device("cpu")

    # Create empty tensor using the optimized method
    empty_tensor = mocked_cellmap_dataset.get_empty_store(shape_config, device)

    # Verify tensor properties
    assert isinstance(
        empty_tensor, torch.Tensor
    ), "get_empty_store should return a torch.Tensor"
    assert empty_tensor.shape == (64, 64, 64), f"Shape mismatch: {empty_tensor.shape}"
    assert empty_tensor.device == device, f"Device mismatch: {empty_tensor.device}"

    # Test that tensor is properly initialized (should be NaN for empty values)
    assert torch.isnan(
        empty_tensor
    ).all(), "Empty tensor should be filled with NaN values"

    # Test memory efficiency - empty tensor should not use excessive memory
    tensor_size_bytes = empty_tensor.element_size() * empty_tensor.nelement()
    expected_size = 64 * 64 * 64 * 4  # float32 is 4 bytes
    assert (
        tensor_size_bytes == expected_size
    ), f"Memory usage mismatch: {tensor_size_bytes} vs {expected_size}"

    # Test that multiple empty tensors can be created consistently
    empty_tensor_2 = mocked_cellmap_dataset.get_empty_store(shape_config, device)
    # Compare NaN tensors properly - NaN != NaN, so check that both are all NaN
    assert torch.isnan(
        empty_tensor_2
    ).all(), "Second empty tensor should also be filled with NaN"
    assert (
        empty_tensor.shape == empty_tensor_2.shape
    ), "Multiple empty tensors should have same shape"


def test_perf_device_consistency_fix(mocked_cellmap_dataset):
    """Test that device consistency issues are resolved."""
    # Test device consistency between different tensor operations
    device = torch.device("cpu")

    # Create a regular tensor
    regular_tensor = torch.ones((64, 64, 64), device=device)

    # Create an empty tensor using our optimized method
    empty_tensor = mocked_cellmap_dataset.get_empty_store(
        {"shape": (64, 64, 64)}, device
    )

    # Test that both tensors are on the same device
    assert (
        regular_tensor.device == empty_tensor.device
    ), "Device consistency issue detected"

    # Test that we can perform operations between them without device errors
    try:
        result = regular_tensor + empty_tensor
        assert result.device == device, "Result tensor device is inconsistent"
    except RuntimeError as e:
        if "device" in str(e).lower():
            pytest.fail(f"Device consistency error in tensor operations: {e}")
        else:
            raise  # Re-raise if it's a different error

    # Test stacking tensors from different sources
    image_tensor = torch.randn((64, 64, 64), device=device)

    # Get an empty tensor from the actual dataset method
    empty_tensor_2 = mocked_cellmap_dataset.get_empty_store(
        {"shape": (64, 64, 64)}, torch.device("cpu")
    )

    # Test that we can stack them (the key test that would fail before our fix)
    try:
        stacked = torch.stack([image_tensor, empty_tensor_2])
        assert stacked.shape == (2, 64, 64, 64)
        assert stacked.device.type == "cpu"
    except RuntimeError as e:
        if "device" in str(e).lower():
            pytest.fail(f"Device consistency fix failed: {e}")
        else:
            raise

    # Test concatenation as well
    try:
        concatenated = torch.cat(
            [image_tensor.unsqueeze(0), empty_tensor_2.unsqueeze(0)], dim=0
        )
        assert concatenated.shape == (2, 64, 64, 64)
        assert concatenated.device.type == "cpu"
    except RuntimeError as e:
        if "device" in str(e).lower():
            pytest.fail(f"Device consistency fix failed in concatenation: {e}")
        else:
            raise


def test_perf_dataloader_creation(mock_dataset):
    """Test that CellMapDataLoader can be created and configured correctly."""
    from cellmap_data import CellMapDataLoader

    # Create a data loader
    dataloader = CellMapDataLoader(mock_dataset, batch_size=2)

    # Verify basic properties
    assert dataloader is not None
    assert dataloader.batch_size == 2


def test_perf_optimization_integration(mocked_cellmap_dataset):
    """Test that performance optimizations work together correctly."""
    # Test that operations complete quickly (performance optimization impact)
    start_time = time.time()

    # Test multiple empty tensor creations (this should be fast)
    for i in range(10):
        empty_tensor = mocked_cellmap_dataset.get_empty_store(
            {"shape": (50, 50, 50)}, torch.device("cpu")
        )
        assert empty_tensor is not None

    end_time = time.time()
    creation_time = end_time - start_time

    # Should be very fast with optimizations
    assert creation_time < 1.0, f"Tensor creation took too long: {creation_time}s"


def test_perf_device_consistency_production_scenario(mocked_cellmap_dataset):
    """Test device consistency in the exact scenario that causes production RuntimeError."""
    # Move dataset to a device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mocked_cellmap_dataset.to(device)

    # Test that get_empty_store uses the correct device (should be dataset.device, not hardcoded CPU)
    empty_tensor = mocked_cellmap_dataset.get_empty_store(
        {"shape": (64, 64, 64)}, mocked_cellmap_dataset.device
    )

    # Verify the tensor is on the expected device (compare device types, not exact device objects)
    assert (
        empty_tensor.device.type == mocked_cellmap_dataset.device.type
    ), f"Empty tensor device type {empty_tensor.device.type} does not match dataset device type {mocked_cellmap_dataset.device.type}"

    # Create mock tensors that would come from class_arrays.values() in production
    # These should all be on the same device type as the empty_tensor
    mock_class_tensor_1 = torch.ones(
        (64, 64, 64), device=mocked_cellmap_dataset.device.type
    )
    mock_class_tensor_2 = torch.zeros(
        (64, 64, 64), device=mocked_cellmap_dataset.device.type
    )

    # This is the exact operation that was failing in production (line 610 in dataset.py)
    # torch.stack(list(class_arrays.values()))
    try:
        stacked_tensors = torch.stack(
            [mock_class_tensor_1, mock_class_tensor_2, empty_tensor]
        )

        # Verify the stacked result
        assert (
            stacked_tensors.device.type == mocked_cellmap_dataset.device.type
        ), "Stacked tensors should be on dataset device type"
        assert stacked_tensors.shape == (
            3,
            64,
            64,
            64,
        ), "Stacked shape should be correct"

    except RuntimeError as e:
        if "Expected all tensors to be on the same device" in str(e):
            pytest.fail(
                f"Device consistency fix failed - tensors are on different devices: {e}"
            )
        else:
            raise  # Re-raise if it's a different error
