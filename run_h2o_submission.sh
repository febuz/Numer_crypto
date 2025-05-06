#!/bin/bash
# H2O Sparkling Water Submission Script
# This script creates a high-quality submission using H2O Sparkling Water

set -e

echo "===================================================="
echo "NUMERAI CRYPTO H2O SPARKLING WATER SUBMISSION"
echo "===================================================="
echo "Creating advanced submission with H2O and proper feature engineering"
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
output="data/submissions/h2o_submission_${timestamp}.csv"

# Get start time for timing
start_time=$(date +%s)

echo "Starting submission pipeline at $(date)"
echo "This will take a few minutes..."

# Run the script with a time limit
python3 scripts/h2o_yiedl_submission.py \
    ${gpu_flag} \
    --output "${output}" \
    --time-limit 1500

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
    
    # Create plots of the predictions
    if [ -f "${main_output}" ]; then
        echo ""
        echo "Creating distribution plots..."
        # Generate a simple histogram using Python
        python3 -c "
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the predictions
df = pd.read_csv('${main_output}')

# Create histogram
plt.figure(figsize=(10, 6))
plt.hist(df['prediction'], bins=50, alpha=0.75)
plt.title('Prediction Distribution')
plt.xlabel('Prediction Value')
plt.ylabel('Count')
plt.grid(True, alpha=0.3)

# Add statistics
stats = f'Mean: {df[\"prediction\"].mean():.4f}, Std: {df[\"prediction\"].std():.4f}\\nMin: {df[\"prediction\"].min():.4f}, Max: {df[\"prediction\"].max():.4f}'
plt.annotate(stats, xy=(0.5, 0.95), xycoords='axes fraction', ha='center', va='top', bbox=dict(boxstyle='round', alpha=0.1))

# Save the plot
plt.savefig('${main_output%.*}_distribution.png')
plt.close()

# If alternative file exists, also plot it
if '${alt_output}' and pd.io.common.file_exists('${alt_output}'):
    # Load the alternative predictions
    alt_df = pd.read_csv('${alt_output}')
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df['prediction'], bins=50, alpha=0.5, label='Main')
    plt.hist(alt_df['prediction'], bins=50, alpha=0.5, label='Alternative')
    plt.title('Prediction Distributions Comparison')
    plt.xlabel('Prediction Value')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the plot
    plt.savefig('${main_output%.*}_comparison.png')
    plt.close()
" 2>/dev/null || echo "Could not create distribution plots"
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