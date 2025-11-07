#!/usr/bin/env python3
"""
Integration tests for the refactored CellMapDataLoader functionality.

These tests verify that the refactored implementation maintains full compatibility
while adding new PyTorch DataLoader parameter support.
"""

import pytest
import torch

from cellmap_data.dataloader import CellMapDataLoader


class MockDataset:
    """Test dataset that implements the minimal interface expected by CellMapDataLoader."""

    def __init__(self, size=20, return_cpu_tensors=False):
        self.size = size
        self.classes = ["class_a", "class_b", "class_c"]
        self.return_cpu_tensors = return_cpu_tensors
        self.class_counts = {"class_a": 7, "class_b": 7, "class_c": 6}
        self.class_weights = {"class_a": 0.33, "class_b": 0.33, "class_c": 0.34}
        self.validation_indices = list(range(size // 2))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if self.return_cpu_tensors:
            # Return CPU tensors for pin_memory testing
            device = "cpu"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        return {
            "input_data": torch.randn(4, 8, 8, device=device),
            "target": torch.tensor(idx % 3, device=device),
            "sample_id": torch.tensor(idx, device=device),
            "__metadata__": {"original_idx": idx, "filename": f"sample_{idx}.dat"},
        }

    def to(self, device, non_blocking=True):
        """Required by CellMapDataLoader interface."""
        pass


class TestRefactoredDataLoader:
    """Test suite for the refactored CellMapDataLoader functionality."""

    def test_backward_compatibility(self):
        """Test that existing code patterns still work after refactoring."""
        dataset = MockDataset(size=12)
        loader = CellMapDataLoader(dataset, batch_size=4, num_workers=0)

        # Original pattern: iter(loader.loader)
        batch = next(iter(loader.loader))
        assert isinstance(batch, dict), "Should return dictionary"
        assert "input_data" in batch, "Should contain input_data key"
        assert batch["input_data"].shape[0] == 4, "Should have correct batch size"

        # Original pattern: loader.refresh()
        loader.refresh()
        batch_after_refresh = next(iter(loader.loader))
        assert (
            batch_after_refresh["input_data"].shape[0] == 4
        ), "Should work after refresh"

        # Original pattern: loader[[0, 1]]
        direct_item = loader[[0, 1]]
        assert direct_item["input_data"].shape[0] == 2, "Direct access should work"

        print("✅ Backward compatibility test passed")

    def test_new_direct_iteration(self):
        """Test the new direct iteration feature."""
        dataset = MockDataset(size=10)
        loader = CellMapDataLoader(dataset, batch_size=3, num_workers=0)

        # New pattern: direct iteration
        batches = []
        for batch in loader:
            batches.append(batch)
            assert isinstance(batch, dict), "Should return dictionary"
            assert "input_data" in batch, "Should contain expected keys"

        expected_batches = (10 + 3 - 1) // 3  # Ceiling division
        assert (
            len(batches) == expected_batches
        ), f"Should generate {expected_batches} batches"

        # Last batch might be smaller
        assert (
            len(batches[-1]["input_data"]) == 1
        ), "Last batch should have 1 sample (10 % 3 = 1)"

        print("✅ New direct iteration test passed")

    def test_pytorch_parameter_integration(self):
        """Test that PyTorch DataLoader parameters work correctly together."""
        dataset = MockDataset(size=15, return_cpu_tensors=True)

        # Test comprehensive parameter combination
        device = "cuda" if torch.cuda.is_available() else "cpu"
        loader = CellMapDataLoader(
            dataset,
            batch_size=4,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
            num_workers=2,
            device=device,
            shuffle=True,
        )

        # Verify configuration (pin_memory only works on CUDA)
        if device == "cuda":
            assert loader._pin_memory, "pin_memory should be enabled on CUDA"
        else:
            assert not loader._pin_memory, "pin_memory should be False on CPU"
        assert loader._persistent_workers, "persistent_workers should be enabled"
        assert loader._drop_last, "drop_last should be enabled"
        assert loader.num_workers == 2, "Should have 2 workers"

        # Test batching behavior
        expected_batches = 15 // 4  # drop_last=True
        assert (
            len(loader) == expected_batches
        ), f"Should have {expected_batches} batches with drop_last=True"

        batches = list(loader)
        assert (
            len(batches) == expected_batches
        ), "Should generate expected number of batches"

        for i, batch in enumerate(batches):
            assert (
                len(batch["input_data"]) == 4
            ), f"Batch {i} should have exactly 4 samples"

            # Verify device transfer
            expected_device = "cuda" if torch.cuda.is_available() else "cpu"
            assert (
                batch["input_data"].device.type == expected_device
            ), f"Should be on {expected_device}"

            # Verify pin_memory (only relevant for CPU->GPU transfer)
            if expected_device == "cuda":
                # Tensors should be transferred to GPU (pin_memory helps with transfer speed)
                assert (
                    batch["input_data"].device.type == "cuda"
                ), "Should be transferred to GPU"

        print("✅ PyTorch parameter integration test passed")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_specific_features(self):
        """Test GPU-specific functionality."""
        dataset = MockDataset(size=8, return_cpu_tensors=True)

        # Test pin_memory with GPU transfer
        loader = CellMapDataLoader(
            dataset, batch_size=2, pin_memory=True, device="cuda", num_workers=0
        )

        batch = next(iter(loader))

        # Verify GPU transfer
        assert batch["input_data"].device.type == "cuda", "Should be on GPU"
        assert batch["target"].device.type == "cuda", "Should be on GPU"

        # Test that pin_memory flag is respected
        assert loader._pin_memory, "pin_memory flag should be True"

        print("✅ GPU-specific features test passed")

    def test_error_handling_and_edge_cases(self):
        """Test error handling and edge cases."""
        dataset = MockDataset(size=5)

        # Test with empty batches (edge case)
        loader = CellMapDataLoader(
            dataset, batch_size=10, drop_last=False, num_workers=0
        )
        batches = list(loader)
        assert (
            len(batches) == 1
        ), "Should generate 1 batch for 5 samples with batch_size=10"
        assert len(batches[0]["input_data"]) == 5, "Batch should contain all 5 samples"

        # Test with drop_last=True and incomplete batch
        loader_drop = CellMapDataLoader(
            dataset, batch_size=10, drop_last=True, num_workers=0
        )
        batches_drop = list(loader_drop)
        assert (
            len(batches_drop) == 0
        ), "Should generate 0 batches with drop_last=True and incomplete batch"

        # Test __len__ calculation
        loader_len = CellMapDataLoader(
            dataset, batch_size=3, drop_last=False, num_workers=0
        )
        expected_len = (5 + 3 - 1) // 3  # Ceiling division
        assert len(loader_len) == expected_len, f"__len__ should return {expected_len}"

        loader_len_drop = CellMapDataLoader(
            dataset, batch_size=3, drop_last=True, num_workers=0
        )
        expected_len_drop = 5 // 3  # Floor division
        assert (
            len(loader_len_drop) == expected_len_drop
        ), f"__len__ with drop_last should return {expected_len_drop}"

        print("✅ Error handling and edge cases test passed")

    def test_multiworker_functionality(self):
        """Test multiworker functionality with the refactored implementation."""
        dataset = MockDataset(size=12)

        # Test with multiple workers
        loader = CellMapDataLoader(
            dataset, batch_size=3, num_workers=3, persistent_workers=True
        )

        # Test that workers are initialized
        batch = next(iter(loader))
        assert batch["input_data"].shape[0] == 3, "Should work with multiple workers"

        # Test that PyTorch loader is initialized
        assert loader._pytorch_loader is not None, "PyTorch loader should exist"

        # Test multiple iterations
        batches = list(loader)
        assert len(batches) == 4, "Should generate 4 batches for 12 samples"

        # Verify PyTorch loader persistence (with persistent_workers enabled)
        assert loader._pytorch_loader is not None, "PyTorch loader should persist"

        print("✅ Multiworker functionality test passed")

    def test_compatibility_parameters(self):
        """Test that unsupported PyTorch parameters are handled gracefully."""
        dataset = MockDataset(size=6)

        # Test with various PyTorch DataLoader parameters (use num_workers=1 so prefetch_factor is applicable)
        loader = CellMapDataLoader(
            dataset,
            batch_size=2,
            timeout=30,  # Not implemented, stored for compatibility
            prefetch_factor=2,  # Stored when num_workers > 0
            worker_init_fn=None,  # Not implemented, stored for compatibility
            generator=None,  # Not implemented, stored for compatibility
            num_workers=1,  # Changed from 0 to 1 so prefetch_factor is stored
        )

        # Should not crash and should store parameters
        assert "timeout" in loader.default_kwargs, "Should store timeout parameter"
        assert (
            "prefetch_factor" in loader.default_kwargs
        ), "Should store prefetch_factor parameter when num_workers > 0"
        assert (
            loader.default_kwargs["timeout"] == 30
        ), "Should store correct timeout value"

        # Should still work normally
        batch = next(iter(loader))
        assert (
            batch["input_data"].shape[0] == 2
        ), "Should work with compatibility parameters"

        print("✅ Compatibility parameters test passed")


def test_integration_basic():
    """Basic integration test that can be run without pytest."""
    test_suite = TestRefactoredDataLoader()

    print("Running integration tests for refactored CellMapDataLoader...")
    print("=" * 60)

    test_suite.test_backward_compatibility()
    test_suite.test_new_direct_iteration()
    test_suite.test_pytorch_parameter_integration()

    if torch.cuda.is_available():
        test_suite.test_gpu_specific_features()
    else:
        print("⚠️  Skipping GPU tests (CUDA not available)")

    test_suite.test_error_handling_and_edge_cases()
    test_suite.test_multiworker_functionality()
    test_suite.test_compatibility_parameters()

    print("=" * 60)
    print("🎉 All integration tests passed!")
    print("\n📊 Summary:")
    print("  ✅ Backward compatibility maintained")
    print("  ✅ New direct iteration works")
    print("  ✅ PyTorch parameters properly implemented")
    print("  ✅ GPU features working (if available)")
    print("  ✅ Edge cases handled correctly")
    print("  ✅ Multiworker support functional")
    print("  ✅ Compatibility parameters stored")


if __name__ == "__main__":
    test_integration_basic()
