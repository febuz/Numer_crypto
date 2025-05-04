#!/bin/bash
# 
# Numerai Crypto Environment Setup using venv
#
# This script sets up a Python virtual environment for the Numerai Crypto project
#

set -e  # Exit on error

# Environment name
ENV_NAME="numerai"

# Python version verification
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Using Python $PYTHON_VERSION"

# Create virtual environment directory
VENV_DIR="${HOME}/.venvs/${ENV_NAME}"
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
pip install --upgrade pip
pip install numpy pandas pyarrow scikit-learn matplotlib h2o pyspark 
pip install python-dotenv psutil

# Install packages from requirements.txt
echo "Installing packages from requirements.txt..."
pip install -r requirements.txt

# Check for NVIDIA GPUs
echo "Checking for NVIDIA GPUs..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPUs detected:"
    nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
    echo "Note: For GPU acceleration with RAPIDS, a conda-based installation is recommended"
    echo "as pip installations of RAPIDS are challenging."
    echo ""
    echo "You may still be able to use GPU acceleration for XGBoost and LightGBM:"
    pip install xgboost lightgbm --upgrade
else
    echo "No NVIDIA GPUs detected or nvidia-smi not available"
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
ACTIVATE_SCRIPT="activate_numerai_venv.sh"
cat > "$ACTIVATE_SCRIPT" << EOF
#!/bin/bash
# Script to activate the Numerai environment

source "$VENV_DIR/bin/activate"

# Print environment info
echo "Numerai environment activated:"
echo "  Python: \$(python --version)"
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
echo "To test hardware capabilities, run:"
echo "  python scripts/check_env.py"
echo