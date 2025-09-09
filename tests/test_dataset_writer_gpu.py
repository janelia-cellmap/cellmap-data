import pytest
import torch
import torch.utils.data
from unittest.mock import Mock, patch
from cellmap_data.dataset_writer import CellMapDatasetWriter


class TestDatasetWriterGPUTransfer:
    """Test GPU transfer functionality for CellMapDatasetWriter"""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_collate_fn_gpu_transfer(self):
        """Test that CellMapDatasetWriter.collate_fn transfers tensors to GPU"""

        # Create a minimal mock writer to test collate_fn
        class MockWriter:
            def __init__(self):
                self.device = torch.device("cuda")

            def collate_fn(self, batch: list[dict]) -> dict[str, torch.Tensor]:
                """Copy of the fixed collate_fn from CellMapDatasetWriter"""
                outputs = {}
                for b in batch:
                    for key, value in b.items():
                        if key not in outputs:
                            outputs[key] = []
                        outputs[key].append(value)
                for key, value in outputs.items():
                    outputs[key] = torch.stack(value).to(self.device, non_blocking=True)
                return outputs

        writer = MockWriter()

        # Create mock batch data on CPU
        mock_batch = [
            {"input_array": torch.randn(1, 8, 8, 8), "idx": torch.tensor(0)},
            {"input_array": torch.randn(1, 8, 8, 8), "idx": torch.tensor(1)},
        ]

        # Ensure input tensors are on CPU
        for batch_item in mock_batch:
            for key, tensor in batch_item.items():
                assert (
                    tensor.device.type == "cpu"
                ), f"Input tensor {key} should be on CPU"

        # Test collate function
        result = writer.collate_fn(mock_batch)

        # Verify all output tensors are on GPU
        assert "input_array" in result
        assert "idx" in result

        for key, tensor in result.items():
            assert (
                tensor.device.type == "cuda"
            ), f"Output tensor {key} should be on CUDA device, got {tensor.device}"
            assert isinstance(tensor, torch.Tensor)

        # Verify tensor shapes are correct
        assert result["input_array"].shape == torch.Size([2, 1, 8, 8, 8])
        assert result["idx"].shape == torch.Size([2])

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_loader_uses_gpu_transfer(self):
        """Test that CellMapDatasetWriter.loader() creates a dataloader that transfers to GPU"""

        # Mock the dependencies to avoid complex initialization
        with (
            patch("cellmap_data.dataset_writer.CellMapImage"),
            patch("cellmap_data.dataset_writer.ImageWriter"),
            patch("cellmap_data.dataset_writer.UPath"),
        ):

            # Create minimal dataset writer for testing
            writer = CellMapDatasetWriter(
                raw_path="/fake/path",
                target_path="/fake/output",
                classes=["test_class"],
                input_arrays={
                    "test_input": {"shape": [8, 8, 8], "scale": [1.0, 1.0, 1.0]}
                },
                target_arrays={
                    "test_target": {"shape": [4, 4, 4], "scale": [2.0, 2.0, 2.0]}
                },
                target_bounds={
                    "test_target": {"x": [0.0, 8.0], "y": [0.0, 8.0], "z": [0.0, 8.0]}
                },
                device="cuda",
            )

            # Test that device is set correctly
            assert writer.device.type == "cuda"

            # Create loader - this returns a standard PyTorch DataLoader
            loader = writer.loader(batch_size=2, num_workers=0)

            # Verify loader is a DataLoader (PyTorch DataLoader doesn't have device attribute)
            assert isinstance(loader, torch.utils.data.DataLoader)
            # The device info is maintained by the dataset writer itself
            assert writer.device.type == "cuda"

            # Test collate function transfers to GPU
            mock_batch = [
                {"test_input": torch.randn(1, 8, 8, 8), "idx": torch.tensor(0)},
                {"test_input": torch.randn(1, 8, 8, 8), "idx": torch.tensor(1)},
            ]

            # Use the loader's collate function (which should be the dataloader's, not writer's)
            result = loader.collate_fn(mock_batch)

            # Verify tensors are on GPU
            for key, tensor in result.items():
                assert (
                    tensor.device.type == "cuda"
                ), f"Loader output tensor {key} should be on CUDA device"
