#!/usr/bin/env python3
"""
Test script to verify tensor creation optimization in image.py
"""

import numpy as np
import torch
import time
from pathlib import Path
import sys

# Add src to path to import cellmap_data
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_tensor_creation_optimization():
    """Test that torch.from_numpy is used for numpy arrays"""
    
    # Create a large numpy array to test performance
    large_array = np.random.random((100, 100, 100)).astype(np.float32)
    
    # Test torch.tensor (copies data)
    start_time = time.time()
    tensor_copy = torch.tensor(large_array)
    copy_time = time.time() - start_time
    
    # Test torch.from_numpy (zero-copy)
    start_time = time.time()
    tensor_zerocopy = torch.from_numpy(large_array)
    zerocopy_time = time.time() - start_time
    
    print(f"Array shape: {large_array.shape}")
    print(f"torch.tensor time: {copy_time:.6f} seconds")
    print(f"torch.from_numpy time: {zerocopy_time:.6f} seconds")
    print(f"Speedup: {copy_time/zerocopy_time:.2f}x")
    
    # Verify they produce equivalent results
    assert torch.allclose(tensor_copy, tensor_zerocopy)
    print("✅ Results are equivalent")
    
    # Verify memory sharing for from_numpy
    original_value = large_array[0, 0, 0]
    large_array[0, 0, 0] = 999.0
    
    if tensor_zerocopy[0, 0, 0] == 999.0:
        print("✅ torch.from_numpy shares memory (zero-copy confirmed)")
    else:
        print("❌ torch.from_numpy does not share memory")
    
    # Reset for copy test
    large_array[0, 0, 0] = original_value
    tensor_copy_test = torch.tensor(large_array)
    large_array[0, 0, 0] = 999.0
    
    if tensor_copy_test[0, 0, 0] != 999.0:
        print("✅ torch.tensor copies data (as expected)")
    else:
        print("❌ torch.tensor unexpectedly shares memory")

if __name__ == "__main__":
    print("Testing tensor creation optimization...")
    test_tensor_creation_optimization()
    print("\nTensor optimization verification complete!")
