#!/usr/bin/env python
"""
Minimal script to check NumPy installation.
"""
import platform
import sys

print(f"Python version: {platform.python_version()}")
print(f"Python executable: {sys.executable}")

try:
    import numpy as np
    print(f"NumPy version: {np.__version__}")
    print("NumPy test:")
    
    # Create a simple array
    arr = np.random.rand(5, 5)
    print(f"Random 5x5 array:\n{arr}")
    
    # Simple operation
    print(f"Sum of array: {np.sum(arr)}")
    print(f"Mean of array: {np.mean(arr)}")
    
    print("\nNumPy is working correctly!")
except ImportError:
    print("NumPy is not installed")
except Exception as e:
    print(f"Error testing NumPy: {e}")