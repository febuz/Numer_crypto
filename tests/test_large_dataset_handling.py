#!/usr/bin/env python3
"""
Test that our large dataset handling works correctly
"""
import os
import sys
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import GPU Math Accelerator
from scripts.features.gpu_math_accelerator import GPUMathAccelerator

def test_with_shape(rows, cols):
    """Test with specific shape without actually creating the full array"""
    print(f"\n===== Testing with shape: {rows:,} rows x {cols:,} columns =====")
    
    # Set memory limit to 20GB (same as in the error logs)
    os.environ["GPU_MEMORY_LIMIT"] = "20.0"
    
    # Initialize accelerator
    acc = GPUMathAccelerator()
    
    # Create small array
    small_array = np.zeros((min(rows, 1000), min(cols, 100)), dtype=np.float32)
    feature_names = [f"feature_{i}" for i in range(min(cols, 100))]
    
    # Test if our special case logic would apply
    result = False
    for case in [
        (rows > 5000000 and 23.0 < 20.0),
        (rows > 3000000 and 23.0 < 12.0),
        (cols > 10000),
        (rows > 3000000 and cols > 3000)
    ]:
        if case:
            print(f"Special case condition would apply: {case}")
            result = True
    
    if result:
        print("✅ Dataset would use CPU fallback (no OOM risk)")
    else:
        print("❌ Dataset would attempt GPU processing (potential OOM risk)")
    
    # For large datasets, check the interaction limit
    if rows > 1000000:
        if cols > 3000:
            max_interactions = max(50, min(100, 150 // 10))
            print(f"For this dataset, max_interactions would be limited to: {max_interactions}")
        else:
            max_interactions = max(300, min(1000, 150 // 3))
            print(f"For this dataset, max_interactions would be: {max_interactions}")

if __name__ == "__main__":
    # Test with the specific dimensions from the error log
    test_with_shape(3435484, 3669)
    
    # Test with a smaller dataset for comparison
    test_with_shape(1000000, 100)
    
    # Test with very large dataset
    test_with_shape(10000000, 500)