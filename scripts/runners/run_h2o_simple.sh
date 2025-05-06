#!/bin/bash
# Run H2O Simple Submission Script
# Creates a high-quality submission using H2O with ensembles

set -e

echo "===================================================="
echo "NUMERAI CRYPTO H2O SIMPLE SUBMISSION"
echo "===================================================="
echo "Creating submission with H2O ensembles and validation"
echo "===================================================="

# Check for GPU availability
GPU_IDS=""
if command -v nvidia-smi &> /dev/null; then
    echo "GPUs detected:"
    nvidia-smi -L
    
    # Get comma-separated list of GPU IDs
    GPU_COUNT=$(nvidia-smi -L | wc -l)
    for i in $(seq 0 $((GPU_COUNT-1))); do
        if [ -z "$GPU_IDS" ]; then
            GPU_IDS="$i"
        else
            GPU_IDS="$GPU_IDS,$i"
        fi
    done
    
    echo "Using GPUs: $GPU_IDS"
    gpu_flag="--gpus $GPU_IDS"
else
    echo "No GPUs detected, using CPU only"
    gpu_flag=""
fi

# Set working directory
cd "$(dirname "$0")"

# Create required directories
mkdir -p data/submissions
mkdir -p data/yiedl/tmp
mkdir -p reports

# Set timestamp for output files
timestamp=$(date +"%Y%m%d_%H%M%S")
output="data/submissions/h2o_simple_${timestamp}.csv"

# Get start time for timing
start_time=$(date +%s)

echo "Starting submission pipeline at $(date)"
echo "This will take a few minutes..."

# Check if we should validate previous submissions
if [ "$1" == "--validate" ]; then
    echo "Validating previous submissions..."
    validate_flag="--validate"
else
    validate_flag=""
fi

# Run the script with a time limit
python3 scripts/h2o_simple_submission.py \
    ${gpu_flag} \
    ${validate_flag} \
    --output "${output}" \
    --time-limit 600

# Check exit code
exit_code=$?

# Calculate elapsed time
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
minutes=$((elapsed_time / 60))
seconds=$((elapsed_time % 60))

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "===================================================="
    echo "SUBMISSION SUCCESSFUL"
    echo "===================================================="
    
    # Get the main and alternative submission files
    main_output="${output}"
    alt_output="${output%.*}_v2.csv"
    
    if [ -f "${main_output}" ]; then
        echo "Main submission file:"
        echo "  - ${main_output}"
        echo "  - File size: $(du -h "${main_output}" | cut -f1)"
        echo "  - Row count: $(($(wc -l < "${main_output}") - 1))"
        
        # Check if the alternative file exists
        if [ -f "${alt_output}" ]; then
            echo ""
            echo "Alternative submission file:"
            echo "  - ${alt_output}"
            echo "  - File size: $(du -h "${alt_output}" | cut -f1)"
            echo "  - Row count: $(($(wc -l < "${alt_output}") - 1))"
        fi
        
        echo ""
        echo "To submit predictions to Numerai, run:"
        echo "python -c \"from numer_crypto.data.retrieval import NumeraiDataRetriever; NumeraiDataRetriever().submit_predictions('${main_output}', 'crypto')\""
    else
        echo "Warning: Output file not found at ${main_output}"
    fi
else
    echo ""
    echo "===================================================="
    echo "SUBMISSION FAILED"
    echo "===================================================="
    echo "Check the log file for details"
fi

echo ""
echo "Total execution time: $minutes minutes and $seconds seconds"
echo "Completed at $(date)"