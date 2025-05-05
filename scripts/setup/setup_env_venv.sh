#!/bin/bash
# 
# Numerai Crypto Environment Setup using venv
#
# This script sets up a Python virtual environment for the Numerai Crypto project
# with Python venv and pip (no conda required)
#

set -e  # Exit on error

# Environment name
ENV_NAME="numerai"

# Python version verification
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Using Python $PYTHON_VERSION"

# Create virtual environment directory
VENV_DIR="${HOME}/venvs/${ENV_NAME}"
mkdir -p $(dirname "$VENV_DIR")

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating new virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists at $VENV_DIR"
    read -p "Do you want to update the existing environment? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting without changes"
        exit 0
    fi
fi

# Activate environment
source "$VENV_DIR/bin/activate"

# Install base packages
echo "Installing base packages..."
pip install --upgrade pip wheel setuptools
pip install numpy pandas pyarrow scikit-learn matplotlib h2o 
pip install "pyspark<4" python-dotenv psutil

# Install packages from requirements.txt (excluding RAPIDS)
echo "Installing packages from requirements.txt..."
grep -v "^#" requirements.txt | grep -v "cudf\|cuml\|cugraph\|cuspatial\|cupy\|dask-cuda\|rapids" > requirements_cpu.txt
pip install -r requirements_cpu.txt

# Check for NVIDIA GPUs
echo "Checking for NVIDIA GPUs..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPUs detected:"
    nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits)
    echo "Found $GPU_COUNT GPU(s)"
    
    # Install GPU-accelerated XGBoost and LightGBM via pip
    echo "Installing GPU-accelerated ML libraries..."
    pip install xgboost lightgbm --upgrade
    
    # Create GPU configuration file
    GPU_CONFIG_PY="$VENV_DIR/gpu_config.py"
    cat > "$GPU_CONFIG_PY" << EOF
"""GPU configuration for Numerai project."""
GPU_COUNT = $GPU_COUNT
GPU_ENABLED = True
EOF
    
else
    echo "No NVIDIA GPUs detected or nvidia-smi not available"
    # Create CPU-only configuration file
    GPU_CONFIG_PY="$VENV_DIR/gpu_config.py"
    cat > "$GPU_CONFIG_PY" << EOF
"""GPU configuration for Numerai project."""
GPU_COUNT = 0
GPU_ENABLED = False
EOF
    
    echo "Installing CPU versions of ML libraries..."
    pip install xgboost lightgbm --upgrade
fi

# Install development tools
echo "Installing additional development tools..."
pip install jupyterlab ipywidgets

# Final setup
echo "Setting up project for development..."
pip install -e .

# Create a script to easily activate the environment
ACTIVATE_SCRIPT="activate_numerai_env.sh"
cat > "$ACTIVATE_SCRIPT" << EOF
#!/bin/bash
# Script to activate the Numerai environment

source "$VENV_DIR/bin/activate"

# Print environment info
echo "Numerai environment activated:"
echo "  Python: \$(python --version)"
echo "  XGBoost: \$(pip freeze | grep xgboost)"
echo "  LightGBM: \$(pip freeze | grep lightgbm)"
echo "  Spark: \$(pip freeze | grep pyspark)"

# Check for NVIDIA GPUs
if command -v nvidia-smi &> /dev/null; then
    echo "  GPUs Available: \$(nvidia-smi --query-gpu=count --format=csv,noheader)"
    echo "  GPU Types: \$(nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq)"
else
    echo "  GPUs Available: None detected"
fi

# Add environment variables for GPU-enabled XGBoost/LightGBM if GPUs present
if [ -f "$VENV_DIR/gpu_config.py" ] && grep -q "GPU_ENABLED = True" "$VENV_DIR/gpu_config.py"; then
    export XGBOOST_GPU_SUPPORT=1
    export LIGHTGBM_GPU_SUPPORT=1
    echo "  GPU acceleration enabled for ML libraries"
fi

echo ""
echo "Ready to use Numerai environment!"
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
echo "To test hardware capabilities, run:"
echo "  python scripts/check_env.py"
echo
echo "Note: Full RAPIDS GPU acceleration is not available with pip installation."
echo "Only XGBoost and LightGBM will use GPU acceleration if GPUs are detected."
echo