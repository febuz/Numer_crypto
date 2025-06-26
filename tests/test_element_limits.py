#!/usr/bin/env python3
"""
Quick test of element limits for our large dataset
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

# Dataset dimensions from actual use case
rows = 3435484
cols = 3669
print(f"Dataset shape: {rows:,} rows x {cols:,} columns")

# Calculate total elements
total_elements = rows * cols
print(f"Total elements: {total_elements:,}")

# Initialize accelerator
os.environ["GPU_MEMORY_LIMIT"] = "20.0"  # Set memory limit to 20GB
acc = GPUMathAccelerator()

# Calculate max elements with our new formula
memory_limit_gb = 20.0
max_elements = int(min(memory_limit_gb * 100000000, 2500000000))  # Our new calculation
print(f"Max elements with new calculation: {max_elements:,}")
print(f"Is dataset under new limit? {total_elements < max_elements}")

# Print the estimated memory that would be needed
element_size_bytes = 4  # float32 is 4 bytes
estimated_memory_gb = (total_elements * element_size_bytes) / (1024**3)
print(f"Estimated memory needed for dataset: {estimated_memory_gb:.2f} GB")
print(f"With interactions (approximately 3x): {estimated_memory_gb * 3:.2f} GB")