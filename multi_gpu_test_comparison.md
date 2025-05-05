# Multi-GPU Machine Learning Comparison

This document compares two approaches for utilizing multiple GPUs in parallel for machine learning tasks:

1. **H2O Sparkling Water** - Integration of H2O with Apache Spark for distributed machine learning
2. **PyTorch** - Native multi-GPU support for neural network training

## Test Scripts

1. **H2O Sparkling Water**: `test_multi_gpu_h2o.py`
   - Leverages H2O's XGBoost implementation
   - Creates a Spark context for each GPU
   - Uses CUDA_VISIBLE_DEVICES for GPU isolation

2. **PyTorch**: `multi_gpu_pytorch.py`
   - Utilizes PyTorch's native CUDA support
   - Implements a simple neural network
   - Uses separate CUDA devices for parallelization

## Setting Up the Environment

### H2O Sparkling Water Setup

For H2O Sparkling Water, you need to set up a special environment:

```bash
cd /home/knight2/repos/Numer_crypto
./scripts/setup/setup_h2o_sparkling_java17.sh
source h2o_sparkling_java17_env/bin/activate.sparkling
python tests/performance/test_multi_gpu_h2o.py
```

### PyTorch Setup

PyTorch only requires a standard installation with CUDA support:

```bash
pip install torch torchvision torchaudio
python multi_gpu_pytorch.py
```

## Implementation Details

### H2O Sparkling Water Approach

1. **Initialization**:
   - Creates a separate Spark session for each GPU
   - Initializes H2O context with unique ports
   - Sets CUDA_VISIBLE_DEVICES to isolate GPUs

2. **Data Management**:
   - Converts Pandas DataFrame to Spark DataFrame
   - Assembles features using Spark VectorAssembler
   - Converts to H2O Frame for model training

3. **Model Training**:
   - Uses H2OXGBoostEstimator with GPU support
   - Configures tree_method="gpu_hist" for GPU acceleration
   - Sets gpu_id=0 (since we've isolated each GPU)

### PyTorch Approach

1. **Initialization**:
   - Sets device using torch.cuda.set_device(gpu_id)
   - Creates a neural network model
   - Moves model to the selected device

2. **Data Management**:
   - Converts numpy arrays to PyTorch tensors
   - Creates DataLoader objects with batching
   - Moves data to the appropriate GPU device

3. **Model Training**:
   - Uses standard PyTorch training loop
   - Performs forward pass, loss calculation, and backward pass
   - Optimizes using Adam optimizer

## Key Advantages

### H2O Sparkling Water

- **Tree-based models**: Excels at XGBoost and other tree-based algorithms
- **Memory optimization**: Better memory management for large datasets
- **Distributed computing**: Leverages Spark for data parallelism
- **Production deployment**: Easy to deploy models to production

### PyTorch

- **Deep learning**: Native support for neural networks
- **Flexibility**: More flexibility in model architecture
- **Ecosystem**: Rich ecosystem for DL research and development
- **Dynamic computation**: Dynamic computation graph for complex models

## Performance Considerations

- **Training speed**: Both systems can utilize GPU acceleration effectively
- **Scaling**: Scales well with number of GPUs in parallel
- **Memory usage**: H2O may have more efficient memory management for tabular data
- **GPU utilization**: Both can achieve high GPU utilization

## Conclusion

Both approaches effectively utilize multiple GPUs in parallel, but they target different use cases:

- Use **H2O Sparkling Water** for gradient boosting models on tabular data
- Use **PyTorch** for deep learning models, especially with unstructured data

For maximum performance, consider using the right tool for your specific modeling needs and data characteristics.

## Running the Test Scripts

Run these scripts to see multi-GPU utilization in action:

```bash
# H2O Sparkling Water
cd /home/knight2/repos/Numer_crypto
./scripts/setup/setup_h2o_sparkling_java17.sh
source h2o_sparkling_java17_env/bin/activate.sparkling
python tests/performance/test_multi_gpu_h2o.py

# PyTorch
cd /home/knight2/repos/Numer_crypto
python multi_gpu_pytorch.py
```

Both scripts will produce visualizations and performance metrics in the reports directory.