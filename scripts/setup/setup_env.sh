#!/bin/bash
# 
# Numerai Crypto Environment Setup
#
# This script sets up a Conda environment for the Numerai Crypto project
# with GPU acceleration using RAPIDS and other GPU-accelerated libraries.
#

set -e  # Exit on error

# Environment name
ENV_NAME="numerai"

# CUDA version - change as needed for your system
CUDA_VERSION="11.8"

# Python version
PYTHON_VERSION="3.10"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    echo "Please install Anaconda or Miniconda first"
    exit 1
fi

# Check if NVIDIA GPUs are available
echo "Checking for NVIDIA GPUs..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPUs detected:"
    nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader)
    echo "Found $GPU_COUNT GPU(s)"
else
    echo "Warning: No NVIDIA GPUs detected or nvidia-smi not installed"
    read -p "Continue with CPU-only setup? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    GPU_COUNT=0
fi

# Check if environment already exists
echo "Checking for existing '$ENV_NAME' environment..."
if conda env list | grep -q "^$ENV_NAME "; then
    echo "Environment '$ENV_NAME' already exists"
    read -p "Do you want to update the existing environment? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Updating environment '$ENV_NAME'..."
    else
        echo "Exiting without changes"
        exit 0
    fi
else
    echo "Creating new environment '$ENV_NAME'..."
    conda create -y -n $ENV_NAME python=$PYTHON_VERSION
fi

# Activate environment
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# Install base packages
echo "Installing base packages..."
conda install -y -c conda-forge \
    numpy pandas pyarrow scikit-learn matplotlib \
    h2o pyspark python-dotenv psutil

# Install packages from requirements.txt
echo "Installing packages from requirements.txt..."
pip install -r requirements.txt

# Install GPU-accelerated packages if GPUs are available
if [ "$GPU_COUNT" -gt 0 ]; then
    echo "Installing GPU-accelerated packages..."
    
    # Install RAPIDS
    echo "Installing RAPIDS ecosystem..."
    conda install -y -c rapidsai -c conda-forge -c nvidia \
        rapids=23.12 python=$PYTHON_VERSION cuda-version=$CUDA_VERSION
    
    # Install GPU-accelerated XGBoost and LightGBM
    echo "Installing GPU-accelerated ML libraries..."
    conda install -y -c conda-forge \
        xgboost lightgbm cudatoolkit=$CUDA_VERSION
    
    # Install RAPIDS Spark integration
    echo "Installing RAPIDS integration for Spark..."
    pip install rapids-4-spark
    
    # Set up environment variables for GPU acceleration
    echo "Setting up environment variables for GPU acceleration..."
    
    # Create a file with environment variable settings that can be sourced
    ENV_SETUP_FILE="$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh"
    mkdir -p "$(dirname "$ENV_SETUP_FILE")"
    
    cat > "$ENV_SETUP_FILE" << EOF
#!/bin/bash
# Environment variables for GPU acceleration

# XGBoost GPU acceleration
export XGBOOST_GPU_SUPPORT=1

# LightGBM GPU acceleration
export LIGHTGBM_GPU_SUPPORT=1

# RAPIDS settings
export RAPIDS_CUDA_VERSION=$CUDA_VERSION
export RAPIDS_NO_INITIALIZE=1

# CUDA setup
export CUDA_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH

# Spark RAPIDS configuration
export SPARK_RAPIDS_DIR=$CONDA_PREFIX/lib/python$PYTHON_VERSION/site-packages/rapids/jars
EOF
    
    chmod +x "$ENV_SETUP_FILE"
    source "$ENV_SETUP_FILE"
    
    echo "GPU setup complete!"
else
    echo "Skipping GPU package installation (no GPUs detected)"
    
    # Install CPU versions of XGBoost and LightGBM
    echo "Installing CPU versions of ML libraries..."
    conda install -y -c conda-forge xgboost lightgbm
fi

# Install additional development tools
echo "Installing additional development tools..."
conda install -y -c conda-forge jupyterlab ipywidgets

# Final setup
echo "Setting up project for development..."
pip install -e .

# Create a script to easily activate the environment
ACTIVATE_SCRIPT="activate_numerai_env.sh"
cat > "$ACTIVATE_SCRIPT" << EOF
#!/bin/bash
# Script to activate the Numerai environment

eval "\$(conda shell.bash hook)"
conda activate $ENV_NAME

# Print environment info
echo "Numerai environment activated:"
echo "  Python: \$(python --version)"
echo "  RAPIDS: \$(pip list | grep -E 'cudf|cuml' || echo 'Not installed')"
echo "  XGBoost: \$(pip list | grep xgboost)"
echo "  LightGBM: \$(pip list | grep lightgbm)"
echo "  Spark: \$(pip list | grep pyspark)"

# Check for NVIDIA GPUs
if command -v nvidia-smi &> /dev/null; then
    echo "  GPUs Available: \$(nvidia-smi --query-gpu=count --format=csv,noheader)"
else
    echo "  GPUs Available: None detected"
fi
EOF

chmod +x "$ACTIVATE_SCRIPT"

echo
echo "====================================================="
echo "Numerai Crypto environment setup complete!"
echo "====================================================="
echo
echo "To activate the environment, run:"
echo "  source ./$ACTIVATE_SCRIPT"
echo
echo "To test GPU acceleration, run:"
echo "  python scripts/test_hardware.py --full"
echo