#!/bin/bash
# Install required packages for GPU-accelerated predictions
# This script creates a local virtual environment to avoid conflicts

set -e

echo "Creating Python virtual environment..."
VENV_DIR="./venv"
python3 -m venv $VENV_DIR

echo "Activating virtual environment..."
source $VENV_DIR/bin/activate

echo "Installing core dependencies..."
pip install --upgrade pip
pip install numpy pandas scikit-learn matplotlib seaborn plotly

echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "Installing H2O version 3.46.0.6 for Sparkling Water compatibility..."
pip install h2o==3.46.0.6

echo "Installing additional dependencies..."
pip install lightgbm xgboost optuna shap

echo "Checking installations..."
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import h2o; print(f'H2O: {h2o.__version__}')"
python -c "import lightgbm as lgb; print(f'LightGBM: {lgb.__version__}')"

echo "Setup complete! Activate the environment using:"
echo "source $VENV_DIR/bin/activate"