#!/bin/bash
# Model Comparison Script for Numerai Crypto Competition
# This script compares multiple models with extensive feature engineering
# to find the best performing model for crypto prediction

set -e

echo "===================================================="
echo "NUMERAI CRYPTO MODEL COMPARISON"
echo "===================================================="

# Check for GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi -L
    
    # Count available GPUs
    gpu_count=$(nvidia-smi -L | wc -l)
    echo "Found $gpu_count GPUs"
    
    # Create comma-separated list of GPU IDs
    gpu_ids=$(seq -s, 0 $((gpu_count-1)))
    echo "Using GPU IDs: $gpu_ids"
else
    echo "No GPU detected, using CPU only"
    gpu_ids=""
fi

# Set working directory
cd "$(dirname "$0")"

# Create required directories
mkdir -p data/submissions
mkdir -p models/comparison
mkdir -p reports/plots

# Parse command line arguments
ram_gb="500"
time_limit_per_model="720"
max_features="5000"
n_samples="100000"

while [[ $# -gt 0 ]]; do
    case $1 in
        --ram)
            ram_gb="$2"
            shift 2
            ;;
        --time-limit)
            time_limit_per_model="$2"
            shift 2
            ;;
        --features)
            max_features="$2"
            shift 2
            ;;
        --samples)
            n_samples="$2"
            shift 2
            ;;
        --no-gpu)
            gpu_ids=""
            shift
            ;;
        --gpu-ids)
            gpu_ids="$2"
            shift 2
            ;;
        *)
            # Unknown option
            echo "Unknown option: $1"
            echo "Usage: $0 [--ram GB] [--time-limit SECONDS] [--features COUNT] [--samples COUNT] [--no-gpu] [--gpu-ids IDS]"
            exit 1
            ;;
    esac
done

# Set timestamp for output files
timestamp=$(date +"%Y%m%d_%H%M%S")
output_dir="data/submissions"
results_file="${output_dir}/model_comparison_results_${timestamp}.json"

# Check if virtual environment exists, if not create it
if [ ! -d "./venv" ]; then
    echo "Setting up virtual environment and installing dependencies..."
    bash scripts/install_requirements.sh
    
    # Install additional dependencies for model comparison
    source ./venv/bin/activate
    pip install catboost pyarrow fastparquet psutil
    deactivate
else
    echo "Using existing virtual environment."
    
    # Ensure additional dependencies are installed
    source ./venv/bin/activate
    pip install -q catboost pyarrow fastparquet psutil
    deactivate
fi

# Activate virtual environment
source ./venv/bin/activate

echo "Starting model comparison at $(date)"
echo "Running with parameters:"
echo "  - RAM: ${ram_gb}GB"
echo "  - Time limit per model: ${time_limit_per_model} seconds"
echo "  - Max features: ${max_features}"
echo "  - Samples: ${n_samples}"
echo "  - GPU IDs: ${gpu_ids:-none (CPU mode)}"
echo ""

# Run the model comparison script
python scripts/run_model_comparison.py \
    --ram "${ram_gb}" \
    --gpus "${gpu_ids}" \
    --time-limit "${time_limit_per_model}" \
    --features "${max_features}" \
    --output-dir "${output_dir}"

exit_code=$?

echo ""
echo "===================================================="
if [[ $exit_code -eq 0 ]]; then
    echo "MODEL COMPARISON COMPLETE"
else
    echo "MODEL COMPARISON FAILED WITH EXIT CODE ${exit_code}"
fi
echo "===================================================="

if [[ $exit_code -eq 0 && -f "$results_file" ]]; then
    echo "Results summary:"
    cat "$results_file" | python -m json.tool
    
    echo ""
    echo "Prediction files generated in $output_dir directory"
    echo "Model comparison plots saved in reports/plots directory"
fi

echo ""
echo "Script completed at $(date)"

# Deactivate virtual environment
deactivate

exit $exit_code