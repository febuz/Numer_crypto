#!/bin/bash
# This script runs a comprehensive comparison of models for Numerai Crypto
# with the goal of achieving RMSE < 0.2 using real Yiedl data.

set -e
LOG_FILE="model_comparison_$(date +%Y%m%d_%H%M%S).log"
echo "Starting model comparison. Logs will be written to $LOG_FILE"

# Create directories
mkdir -p data/processed data/submissions models/comparison

# Step 1: Process Yiedl data if needed
if [ "$1" != "--skip-processing" ]; then
  echo "Processing Yiedl data..."
  python3 scripts/process_yiedl_data.py | tee -a "$LOG_FILE"
else
  echo "Skipping data processing step" | tee -a "$LOG_FILE"
fi

# Step 2: Run model comparison script
echo "Running model comparison with 20+ models..." | tee -a "$LOG_FILE"
python3 scripts/model_comparison.py | tee -a "$LOG_FILE"

# Step 3: Run H2O AutoML for 30 minutes
echo "Running H2O AutoML for 30 minutes..." | tee -a "$LOG_FILE"
python3 scripts/h2o_automl_crypto.py --max-runtime 1800 --max-models 50 | tee -a "$LOG_FILE"

# Step 4: Run standard model pipeline
echo "Running standard prediction pipeline..." | tee -a "$LOG_FILE"
python3 scripts/train_predict_crypto.py | tee -a "$LOG_FILE"

# Summary of results
echo "================================" | tee -a "$LOG_FILE"
echo "MODEL COMPARISON SUMMARY" | tee -a "$LOG_FILE"
echo "================================" | tee -a "$LOG_FILE"
echo "Submission files generated:" | tee -a "$LOG_FILE"
find data/submissions -type f -name "*submission*.csv" -mtime -1 | sort | tee -a "$LOG_FILE"
echo ""

# Find RMSE values in validation files
echo "Validation RMSE values:" | tee -a "$LOG_FILE"
find data/submissions -type f -name "*validation*.json" -mtime -1 -exec sh -c "echo {} && grep -o '\"rmse\": [0-9.]*' {} | head -1" \; | tee -a "$LOG_FILE"

echo "================================" | tee -a "$LOG_FILE"
echo "Model comparison complete!" | tee -a "$LOG_FILE"
echo "See external submission directory: /media/knight2/EDB/cryptos/submission/" | tee -a "$LOG_FILE"

# Check if any submission achieved RMSE < 0.2
BEST_RMSE=$(find data/submissions -type f -name "*validation*.json" -mtime -1 -exec grep -o '"rmse": [0-9.]*' {} \; | sort -n | head -1 | cut -d':' -f2 | tr -d ' ')
if (( $(echo "$BEST_RMSE < 0.2" | bc -l) )); then
  echo "SUCCESS! Achieved RMSE < 0.2 ($BEST_RMSE)" | tee -a "$LOG_FILE"
else
  echo "Target RMSE < 0.2 not yet achieved. Best RMSE: $BEST_RMSE" | tee -a "$LOG_FILE"
fi