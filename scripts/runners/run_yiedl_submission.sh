#!/bin/bash
# Quick Yiedl Submission Script
# This script creates a submission for the Numerai Crypto competition using Yiedl data
# Optimized for a 30-minute time constraint

set -e

echo "===================================================="
echo "NUMERAI CRYPTO YIEDL SUBMISSION"
echo "===================================================="
echo "Creating a high-quality submission using Yiedl data"
echo "Optimized for speed (30-minute constraint)"
echo "===================================================="

# Check for GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi -L
    use_gpu="--gpu"
    echo "Using GPU acceleration"
else
    echo "No GPU detected, using CPU"
    use_gpu=""
fi

# Set working directory
cd "$(dirname "$0")"

# Create required directories
mkdir -p data/submissions
mkdir -p data/yiedl/tmp

# Set timestamp for output files
timestamp=$(date +"%Y%m%d_%H%M%S")
output="data/submissions/yiedl_submission_${timestamp}.csv"

echo "Starting submission pipeline at $(date)"
echo "This will take a few minutes..."

# Run the submission script
# Try different Python commands
if command -v python3 &> /dev/null; then
    python3 scripts/quick_yiedl_submission.py \
        ${use_gpu} \
        --output "${output}" \
        --random-seed 42
elif command -v python &> /dev/null; then
    python scripts/quick_yiedl_submission.py \
        ${use_gpu} \
        --output "${output}" \
        --random-seed 42
else
    echo "Error: Python command not found. Please ensure Python is installed."
    exit 1
fi

# Check if submission was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "===================================================="
    echo "SUBMISSION SUCCESSFUL"
    echo "===================================================="
    
    # Check output files
    main_output="${output}"
    alt_output="${output%.*}_v2.csv"
    
    if [ -f "${main_output}" ]; then
        echo "Main submission file created:"
        echo "  - ${main_output}"
        echo "  - File size: $(du -h "${main_output}" | cut -f1)"
        echo "  - Row count: $(($(wc -l < "${main_output}") - 1))"
    else
        echo "WARNING: Main submission file not found!"
    fi
    
    if [ -f "${alt_output}" ]; then
        echo ""
        echo "Alternative submission file created:"
        echo "  - ${alt_output}"
        echo "  - File size: $(du -h "${alt_output}" | cut -f1)"
        echo "  - Row count: $(($(wc -l < "${alt_output}") - 1))"
    fi
    
    echo ""
    echo "To submit predictions to Numerai, run:"
    echo "python -c \"from numer_crypto.data.retrieval import NumeraiDataRetriever; NumeraiDataRetriever().submit_predictions('${main_output}', 'crypto')\""
else
    echo ""
    echo "===================================================="
    echo "SUBMISSION FAILED"
    echo "===================================================="
    echo "Check the log output above for details"
fi

echo ""
echo "Script completed at $(date)"