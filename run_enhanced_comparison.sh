#!/bin/bash
# Run enhanced model comparison with H2O Sparkling Water
# This script integrates multiple approaches from the repository
# and creates a comprehensive model comparison table

set -e
LOG_FILE="enhanced_comparison_$(date +%Y%m%d_%H%M%S).log"
echo "Starting enhanced model comparison. Logs will be written to $LOG_FILE"

# Create directories
mkdir -p data/processed data/submissions/enhanced models/enhanced

# Ensure Java is available
if ! command -v java &> /dev/null; then
    echo "Java is required but not found. Please install Java."
    exit 1
fi

# Check Java version
JAVA_VERSION=$(java -version 2>&1 | grep -i version | awk -F'"' '{print $2}' | cut -d'.' -f1)
if [[ "$JAVA_VERSION" = "1" ]]; then
    # For Java 1.x, use second number
    JAVA_VERSION=$(java -version 2>&1 | grep -i version | awk -F'"' '{print $2}' | cut -d'.' -f2)
fi
echo "Java version: $JAVA_VERSION"

# Set up environment based on Java version
if [[ "$JAVA_VERSION" = "11" ]]; then
    echo "Using Java 11 environment setup"
    if [ -f scripts/setup/setup_test_env_java11.sh ]; then
        source scripts/setup/setup_test_env_java11.sh
    elif [ -f scripts/setup_java11_env.sh ]; then
        source scripts/setup_java11_env.sh
    fi
elif [[ "$JAVA_VERSION" = "17" ]]; then
    echo "Using Java 17 environment setup"
    if [ -f scripts/setup/setup_test_env_java17.sh ]; then
        source scripts/setup/setup_test_env_java17.sh
    elif [ -f scripts/setup_java17_env.sh ]; then
        source scripts/setup_java17_env.sh
    fi
else
    echo "Warning: Unknown Java version. Proceeding without specific setup."
fi

# Step 1: Process Yiedl data if needed
if [ "$1" != "--skip-processing" ]; then
    echo "Processing Yiedl data..." | tee -a "$LOG_FILE"
    python3 scripts/process_yiedl_data.py | tee -a "$LOG_FILE"
else
    echo "Skipping data processing step" | tee -a "$LOG_FILE"
fi

# Step 2: Run enhanced model comparison
echo "Running enhanced model comparison..." | tee -a "$LOG_FILE"
RUNTIME=${2:-1800}  # Default 30 minutes or use second argument
echo "Using runtime of $RUNTIME seconds" | tee -a "$LOG_FILE"
python3 scripts/enhanced_model_comparison.py --runtime $RUNTIME | tee -a "$LOG_FILE"

# Step 3: Generate report
echo "Generating comprehensive report..." | tee -a "$LOG_FILE"

# Find the latest model comparison file
LATEST_COMPARISON=$(find data/submissions/enhanced -name "model_comparison_*.md" -type f -print0 | xargs -0 ls -t | head -n1)

if [ -n "$LATEST_COMPARISON" ]; then
    # Create enhanced summary
    OUTPUT_FILE="ENHANCED_MODEL_SUMMARY.md"
    echo "# Enhanced Model Comparison Summary" > $OUTPUT_FILE
    echo "" >> $OUTPUT_FILE
    echo "## Overview" >> $OUTPUT_FILE
    echo "" >> $OUTPUT_FILE
    echo "This report presents a comprehensive comparison of various models for Numerai Crypto prediction," >> $OUTPUT_FILE
    echo "integrating multiple approaches from the repository." >> $OUTPUT_FILE
    echo "" >> $OUTPUT_FILE
    echo "## Model Performance" >> $OUTPUT_FILE
    echo "" >> $OUTPUT_FILE
    tail -n +4 "$LATEST_COMPARISON" >> $OUTPUT_FILE
    echo "" >> $OUTPUT_FILE
    
    # Add validation metrics
    LATEST_VALIDATION=$(find data/submissions/enhanced -name "validation_results_*.json" -type f -print0 | xargs -0 ls -t | head -n1)
    if [ -n "$LATEST_VALIDATION" ]; then
        echo "## Validation Results" >> $OUTPUT_FILE
        echo "" >> $OUTPUT_FILE
        echo '```json' >> $OUTPUT_FILE
        cat "$LATEST_VALIDATION" >> $OUTPUT_FILE
        echo '```' >> $OUTPUT_FILE
    fi
    
    echo "" >> $OUTPUT_FILE
    echo "## Prediction Files" >> $OUTPUT_FILE
    echo "" >> $OUTPUT_FILE
    echo "Standard submission:" >> $OUTPUT_FILE
    find data/submissions/enhanced -name "numerai_standard_*.csv" -type f -print0 | xargs -0 ls -t | head -n1 >> $OUTPUT_FILE
    echo "" >> $OUTPUT_FILE
    echo "Unique submission:" >> $OUTPUT_FILE
    find data/submissions/enhanced -name "numerai_unique_*.csv" -type f -print0 | xargs -0 ls -t | head -n1 >> $OUTPUT_FILE
    
    echo "Comprehensive report generated: $OUTPUT_FILE" | tee -a "$LOG_FILE"
fi

# Summary
echo "================================" | tee -a "$LOG_FILE"
echo "ENHANCED MODEL COMPARISON COMPLETE" | tee -a "$LOG_FILE"
echo "================================" | tee -a "$LOG_FILE"

# Find RMSE values
echo "Best RMSE values:" | tee -a "$LOG_FILE"
if [ -n "$LATEST_COMPARISON" ]; then
    head -n 12 "$LATEST_COMPARISON" | grep -E "^\|" | sort -t'|' -k5,5n | head -n 3 | tee -a "$LOG_FILE"
fi

echo "================================" | tee -a "$LOG_FILE"
echo "Enhanced model comparison complete!" | tee -a "$LOG_FILE"
echo "See external submission directory: /media/knight2/EDB/cryptos/submission/enhanced/" | tee -a "$LOG_FILE"