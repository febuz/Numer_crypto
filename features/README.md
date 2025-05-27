# Feature Generation Modules

This directory contains modules for feature generation, with both CPU and GPU accelerated implementations.

## Available Feature Generators

- **polars_generator.py**: CPU-based feature generation using Polars for high-performance operations
- **polars_gpu_accelerator.py**: GPU-accelerated feature generation using Polars with GPU libraries
- **pyspark_generator.py**: Distributed feature generation using PySpark
- **gpu_accelerator.py**: General GPU acceleration utilities for feature generation

## GPU Acceleration

The `polars_gpu_accelerator.py` module provides GPU acceleration for feature generation operations. It can significantly improve performance for large datasets when a compatible GPU is available.

### Requirements

For GPU acceleration, you need:

1. An NVIDIA GPU with CUDA support
2. NVIDIA drivers installed
3. CUDA toolkit installed (version 10.x, 11.x, or 12.x)
4. One of the following Python GPU libraries:
   - CuPy (recommended for data processing)
   - PyTorch
   - TensorFlow

### Installation

The GPU acceleration libraries are automatically installed by the `setup_env.sh` script if a GPU is detected. To manually install:

```bash
# For CUDA 12.x
pip install cupy-cuda12x
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 11.x
pip install cupy-cuda11x
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 10.x
pip install cupy-cuda10x
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu102

# For all CUDA versions
pip install tensorflow
pip install nvidia-ml-py py3nvml GPUtil
```

### Checking GPU Support

To check if GPU acceleration is available, run:

```bash
python -m features.polars_gpu_accelerator --check-gpu
```

Or if you're using the project's virtual environment:

```bash
check-gpu
```

### Usage

```python
import polars as pl
from features.polars_gpu_accelerator import PolarsGPUAccelerator

# Create a DataFrame
df = pl.read_parquet('my_data.parquet')

# Initialize the GPU accelerator
accelerator = PolarsGPUAccelerator()

# Generate features
result_df = accelerator.generate_all_features(
    df,
    group_col='symbol',  # Column for grouping
    numeric_cols=['price', 'volume', 'open', 'high', 'low'],  # Columns to generate features from
    date_col='date'  # Date column for time-based features
)

# You can also use specific feature generation methods:
# Rolling window features (mean, std, max, min)
rolling_df = accelerator.generate_rolling_features(
    df, 
    group_col='symbol',
    numeric_cols=['price', 'volume'],
    windows=[3, 7, 14, 28]  # Window sizes
)

# Lag features
lag_df = accelerator.generate_lag_features(
    df,
    group_col='symbol',
    numeric_cols=['price', 'volume'],
    lag_periods=[1, 3, 7, 14]  # Lag periods
)

# Exponential weighted moving average features
ewm_df = accelerator.generate_ewm_features(
    df,
    group_col='symbol',
    numeric_cols=['price', 'volume'],
    spans=[5, 10, 20, 40]  # EWM spans
)
```

### Command Line Usage

You can also use the module from the command line:

```bash
python -m features.polars_gpu_accelerator --input-file data.parquet --output-file features.parquet --group-col symbol --date-col date --limit-cols 20
```

Add `--benchmark` flag to compare CPU vs GPU performance:

```bash
python -m features.polars_gpu_accelerator --input-file data.parquet --benchmark
```

## Fallback to CPU

If no GPU is available, the accelerator will automatically fallback to CPU processing using NumPy.