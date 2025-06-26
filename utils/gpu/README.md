# GPU Acceleration Utilities

This directory contains utility modules for GPU acceleration, providing efficient implementations for data processing and feature engineering tasks.

## Modules

### `accelerator.py`

The main interface for GPU acceleration. This module provides a unified API for using GPU-accelerated functions, abstracting away the underlying libraries (CuPy, PyTorch).

```python
from utils.gpu.accelerator import accelerator

# Apply basic mathematical transformations
X_transformed, feature_names = accelerator.basic_transforms(X, feature_names)

# Apply polynomial transformations
X_poly, poly_names = accelerator.polynomial_transforms(X, feature_names, max_degree=2)
```

### `memory_utils.py`

Utilities for managing GPU memory, including estimation, measurement, and memory clearing functions.

```python
from utils.gpu.memory_utils import clear_gpu_memory, get_gpu_memory_usage

# Check current memory usage
used_gb, free_gb, total_gb = get_gpu_memory_usage()

# Clear GPU memory
clear_gpu_memory(force_full_clear=True)
```

### `data_conversion.py`

Utilities for converting data between CPU and GPU formats, supporting various GPU libraries.

```python
from utils.gpu.data_conversion import to_gpu, to_cpu, chunk_array_for_gpu

# Move data to GPU
gpu_data = to_gpu(numpy_data)

# Move data back to CPU
cpu_data = to_cpu(gpu_data)

# Split large arrays into chunks for GPU processing
chunks = chunk_array_for_gpu(large_data, max_chunk_size=1000000)
```

### `math_transforms.py`

Implementation of various mathematical transformations accelerated by GPU.

```python
from utils.gpu.math_transforms import basic_transforms, polynomial_transforms

# Apply basic transformations
X_basic, names_basic = basic_transforms(X, feature_names)

# Apply trigonometric transformations
X_trig, names_trig = trigonometric_transforms(X, feature_names)
```

## Usage Examples

### Basic Example

```python
import numpy as np
from utils.gpu.accelerator import accelerator

# Create sample data
X = np.random.random((10000, 20)).astype(np.float32)
feature_names = [f"feature_{i}" for i in range(X.shape[1])]

# Apply basic transformations
X_transformed, transformed_names = accelerator.basic_transforms(X, feature_names)

# Apply polynomial transformations
X_poly, poly_names = accelerator.polynomial_transforms(X, feature_names, max_degree=2)

# Clear GPU memory when done
accelerator.clear_memory()
```

### Processing Large Data in Chunks

```python
import numpy as np
from utils.gpu.accelerator import accelerator

# Large dataset
X = np.random.random((1000000, 50)).astype(np.float32)
feature_names = [f"feature_{i}" for i in range(X.shape[1])]

# Custom processing function
def process_chunk(chunk, names):
    # Apply some transformations
    return accelerator.basic_transforms(chunk, names)

# Process in chunks
X_processed, names = accelerator.process_in_chunks(
    X, 
    chunk_size=10000, 
    processing_func=process_chunk,
    names=feature_names
)
```

## GPU Compatibility

The utilities automatically detect and use available GPU libraries:

1. First tries CuPy (preferred for numerical operations)
2. Falls back to PyTorch if CuPy is not available
3. Uses CPU with NumPy if no GPU libraries are available

## Performance Tips

1. Convert data to `float32` instead of `float64` to reduce memory usage and improve performance
2. Use `accelerator.process_in_chunks()` for large datasets to avoid GPU memory issues
3. Call `accelerator.clear_memory()` after processing to free GPU memory
4. Monitor GPU memory usage with `get_memory_usage()` to avoid out-of-memory errors