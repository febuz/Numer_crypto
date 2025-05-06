#!/bin/bash
# High Memory GPU-Accelerated Crypto Model Runner
# This script sets up and runs the high memory crypto model with GPU acceleration

set -e

echo "===================================================="
echo "HIGH MEMORY GPU-ACCELERATED CRYPTO MODEL"
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
mkdir -p data/yiedl
mkdir -p models/yiedl
mkdir -p reports/plots

# External feature store directory
DEFAULT_FSTORE_DIR="/media/knight2/EDB/fstore"

# Parse command line arguments
ram_gb="500"
forecast_days="20"
time_limit="900"  # Default to 15 minutes (900 seconds)
use_nn=""
evaluate=""
output_file=""
test_only=""
skip_tests=""
fstore_dir="$DEFAULT_FSTORE_DIR"

while [[ $# -gt 0 ]]; do
    case $1 in
        --ram)
            ram_gb="$2"
            shift 2
            ;;
        --forecast-days)
            forecast_days="$2"
            shift 2
            ;;
        --time-limit)
            time_limit="$2"
            shift 2
            ;;
        --output)
            output_file="--output $2"
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
        --nn-model)
            use_nn="--nn-model"
            shift
            ;;
        --evaluate)
            evaluate="--evaluate"
            shift
            ;;
        --test)
            test_only="yes"
            shift
            ;;
        --skip-tests)
            skip_tests="yes"
            shift
            ;;
        --fstore-dir)
            fstore_dir="$2"
            shift 2
            ;;
        *)
            # Unknown option
            echo "Unknown option: $1"
            echo "Usage: $0 [--ram GB] [--forecast-days DAYS] [--time-limit SECONDS] [--output FILE] [--no-gpu] [--gpu-ids IDS] [--nn-model] [--evaluate] [--test] [--skip-tests] [--fstore-dir DIR]"
            exit 1
            ;;
    esac
done

# Set timestamp for output files
timestamp=$(date +"%Y%m%d_%H%M%S")
output_dir="data/submissions"
default_output="${output_dir}/crypto_submission_${timestamp}.csv"

# Check if virtual environment exists, if not create it
if [ ! -d "./venv" ]; then
    echo "Setting up virtual environment and installing dependencies..."
    bash scripts/install_requirements.sh
else
    echo "Using existing virtual environment."
fi

# Create external feature store directory if it doesn't exist
if [ ! -d "$fstore_dir" ]; then
    echo "Creating external feature store directory: $fstore_dir"
    mkdir -p "$fstore_dir"
fi

# Activate virtual environment
source ./venv/bin/activate

# Run test script if requested and not skipped
if [[ -n "$test_only" ]]; then
    echo "Running environment tests..."
    if [[ -z "$gpu_ids" ]]; then
        python scripts/test_high_mem_crypto_model.py --no-gpu
    else
        python scripts/test_high_mem_crypto_model.py
    fi
    
    test_exit=$?
    
    if [[ $test_exit -eq 0 ]]; then
        echo "Tests completed successfully. Environment is ready."
    else
        echo "Tests failed with exit code $test_exit. Check output for details."
    fi
    
    # Deactivate and exit
    deactivate
    exit $test_exit
fi

# Run tests if not skipped
if [[ -z "$skip_tests" && -z "$test_only" ]]; then
    echo "Running environment tests before starting model (use --skip-tests to bypass)..."
    if [[ -z "$gpu_ids" ]]; then
        python scripts/test_high_mem_crypto_model.py --no-gpu
    else
        python scripts/test_high_mem_crypto_model.py
    fi
    
    test_exit=$?
    
    if [[ $test_exit -ne 0 ]]; then
        echo "WARNING: Environment tests failed with exit code $test_exit."
        echo "You can still proceed, but there may be issues with the model run."
        echo "To proceed anyway, press Enter. To exit, press Ctrl+C."
        read -r
    else
        echo "Environment tests passed. Proceeding with model run."
    fi
else
    if [[ -n "$skip_tests" ]]; then
        echo "Skipping environment tests as requested."
    fi
fi

echo "Starting high memory crypto model at $(date)"
echo "Running with parameters:"
echo "  - RAM: ${ram_gb}GB"
echo "  - Forecast days: ${forecast_days}"
echo "  - Time limit: ${time_limit} seconds"
echo "  - GPU IDs: ${gpu_ids:-none (CPU mode)}"
echo "  - Neural network: ${use_nn:-disabled}"
echo "  - Evaluation: ${evaluate:-disabled}"
echo "  - Output file: ${output_file:-$default_output}"
echo "  - Feature store directory: ${fstore_dir}"
echo ""

# Run the model
python scripts/high_mem_crypto_model.py \
    --gpus "${gpu_ids}" \
    --ram "${ram_gb}" \
    --forecast-days "${forecast_days}" \
    --time-limit "${time_limit}" \
    --fstore-dir "${fstore_dir}" \
    ${use_nn} \
    ${evaluate} \
    ${output_file}

exit_code=$?

echo ""
echo "===================================================="
if [[ $exit_code -eq 0 ]]; then
    echo "MODEL RUN COMPLETE"
else
    echo "MODEL RUN FAILED WITH EXIT CODE ${exit_code}"
fi
echo "===================================================="

if [[ $exit_code -eq 0 ]]; then
    # If output file wasn't specified, use the default
    actual_output=${output_file:9}  # strip "--output " prefix
    if [ -z "$actual_output" ]; then
        actual_output="$default_output"
    fi
    
    echo "Submission created:"
    echo "${actual_output}"
    echo ""

    # Display submission statistics if available
    if [ -f "${actual_output}" ]; then
        echo "Submission statistics:"
        echo "  - File size: $(du -h "${actual_output}" | cut -f1)"
        echo "  - Row count: $(($(wc -l < "${actual_output}") - 1))"
    fi

    echo ""
    echo "To submit predictions to Numerai, run:"
    echo "python -c \"from numer_crypto.data.retrieval import NumeraiDataRetriever; NumeraiDataRetriever().submit_predictions('${actual_output}', 'crypto')\""
    
    # If evaluate was used, display where to find the reports
    if [[ -n "$evaluate" ]]; then
        echo ""
        echo "Evaluation reports saved to the reports/plots directory."
        echo "View model training history and performance metrics in these files."
    fi
    
    # Display feature store info
    echo ""
    echo "Feature store information:"
    echo "  - Location: ${fstore_dir}"
    ls -lh "${fstore_dir}" | grep -v "^total" | wc -l | xargs -I {} echo "  - Files: {}"
    du -sh "${fstore_dir}" | cut -f1 | xargs -I {} echo "  - Total size: {}"
fi

echo ""
echo "Script completed at $(date)"

# Deactivate virtual environment
deactivate

exit $exit_code