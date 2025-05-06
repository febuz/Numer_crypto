#!/bin/bash
# Quick Yiedl Submission Script - 30 Minute Version
# This script creates a submission for the Numerai Crypto competition using Yiedl data
# Designed to run in under 30 minutes even in constrained environments

set -e

echo "===================================================="
echo "NUMERAI CRYPTO YIEDL QUICK SUBMISSION"
echo "===================================================="
echo "Creating a high-quality submission using Yiedl data"
echo "Optimized for speed (under 30-minute constraint)"
echo "===================================================="

# Get start time
start_time=$(date +%s)

# Set working directory
cd "$(dirname "$0")"

# Create required directories
mkdir -p data/submissions
mkdir -p data/yiedl/tmp

# Set timestamp for output files
timestamp=$(date +"%Y%m%d_%H%M%S")
output="data/submissions/yiedl_submission_${timestamp}.csv"

echo "Starting at $(date)"
echo "Will use the advanced script for better quality submissions"

# Run the script
python3 scripts/advanced_yiedl_submission.py

# Check if successful
if [ $? -eq 0 ]; then
    # Get the latest CSV files in the submissions directory
    latest_files=$(ls -t data/submissions/advanced_yiedl_*.csv | head -2)
    
    if [ -n "$latest_files" ]; then
        echo ""
        echo "===================================================="
        echo "SUBMISSION SUCCESSFUL"
        echo "===================================================="
        
        # Show file info
        echo "Submission files created:"
        for file in $latest_files; do
            echo "  - $file"
            echo "  - File size: $(du -h $file | cut -f1)"
            echo "  - Row count: $(($(wc -l < $file) - 1))"
            echo ""
        done
        
        # Find the first file for submission command
        submit_file=$(echo "$latest_files" | head -1)
        
        echo "To submit predictions to Numerai, run:"
        echo "python -c \"from numer_crypto.data.retrieval import NumeraiDataRetriever; NumeraiDataRetriever().submit_predictions('$submit_file', 'crypto')\""
    else
        echo "Submission files not found in the expected location"
    fi
    
else
    echo ""
    echo "===================================================="
    echo "SUBMISSION ATTEMPT FAILED"
    echo "===================================================="
    echo "Trying fallback method..."
    
    # Run the simple script as fallback
    python3 scripts/simple_yiedl_submission.py
    
    if [ $? -eq 0 ]; then
        echo "Fallback successful"
        latest_files=$(ls -t data/submissions/simple_yiedl_*.csv | head -2)
        
        if [ -n "$latest_files" ]; then
            echo "Submission files created:"
            for file in $latest_files; do
                echo "  - $file"
                echo "  - File size: $(du -h $file | cut -f1)"
                echo "  - Row count: $(($(wc -l < $file) - 1))"
                echo ""
            done
            
            # Find the first file for submission command
            submit_file=$(echo "$latest_files" | head -1)
            
            echo "To submit predictions to Numerai, run:"
            echo "python -c \"from numer_crypto.data.retrieval import NumeraiDataRetriever; NumeraiDataRetriever().submit_predictions('$submit_file', 'crypto')\""
        fi
    else
        echo "Both submission methods failed"
    fi
fi

# Calculate elapsed time
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
minutes=$((elapsed_time / 60))
seconds=$((elapsed_time % 60))

echo ""
echo "Total time: ${minutes} minutes, ${seconds} seconds"
echo "Completed at $(date)"