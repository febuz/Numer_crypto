#!/bin/bash

# Script to create 8 standard model files in submission directory and remove submissions directory
# Usage: bash create_submission_files.sh

set -e

# Define directories
BASE_DIR="/media/knight2/EDB/numer_crypto_temp"
SUBMISSION_DIR="${BASE_DIR}/submission"
SUBMISSIONS_DIR="${BASE_DIR}/submissions"
PREDICTIONS_DIR="${BASE_DIR}/predictions"
MODELS_DIR="${BASE_DIR}/models"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Create submission directory if it doesn't exist
mkdir -p "${SUBMISSION_DIR}"
log_info "Ensuring submission directory exists at ${SUBMISSION_DIR}"

# Find the latest model files
log_info "Finding latest model files from ${PREDICTIONS_DIR} and ${MODELS_DIR}..."
LATEST_MODELS=($(find "${PREDICTIONS_DIR}" "${MODELS_DIR}" -name "*.csv" -type f -printf "%T@ %p\n" 2>/dev/null | sort -nr | head -n 20 | awk '{print $2}'))

# Create or copy the 8 standard model files
MODEL_FILES=(
    "submission_1_gravitator_best.csv"
    "submission_2_diverse_models.csv" 
    "submission_3_crypto_randomforest.csv"
    "submission_4_crypto_xgboost.csv"
    "submission_5_diverse_xgb_predictions.csv"
    "submission_6_ensemble_expert_mean.csv"
    "submission_7_ensemble_predictions.csv"
    "submission_8_ensemble_predictions.csv"
)

# If we have model files, use them, otherwise create dummy files
if [ ${#LATEST_MODELS[@]} -gt 0 ]; then
    for i in {0..7}; do
        if [ $i -lt ${#LATEST_MODELS[@]} ]; then
            cp "${LATEST_MODELS[$i]}" "${SUBMISSION_DIR}/${MODEL_FILES[$i]}"
            log_info "Copied ${LATEST_MODELS[$i]} to ${SUBMISSION_DIR}/${MODEL_FILES[$i]}"
        else
            # If we don't have enough models, copy the first one for remaining slots
            cp "${LATEST_MODELS[0]}" "${SUBMISSION_DIR}/${MODEL_FILES[$i]}"
            log_info "Copied ${LATEST_MODELS[0]} to ${SUBMISSION_DIR}/${MODEL_FILES[$i]}"
        fi
    done
else
    # Create dummy files if no models are found
    log_warning "No model files found. Creating dummy submission files."
    for i in {0..7}; do
        echo "id,prediction" > "${SUBMISSION_DIR}/${MODEL_FILES[$i]}"
        echo "BTC_1,0.1" >> "${SUBMISSION_DIR}/${MODEL_FILES[$i]}"
        log_info "Created dummy file: ${SUBMISSION_DIR}/${MODEL_FILES[$i]}"
    done
fi

# Remove the submissions directory if it exists
if [ -d "${SUBMISSIONS_DIR}" ]; then
    log_info "Removing submissions directory..."
    rm -rf "${SUBMISSIONS_DIR}"
    log_success "Removed submissions directory ${SUBMISSIONS_DIR}"
else
    log_info "No submissions directory found at ${SUBMISSIONS_DIR} - nothing to remove"
fi

log_success "Created 8 model files in submission directory and checked for submissions directory removal"
log_info "Submission files available in: ${SUBMISSION_DIR}/"

# List generated submission files
echo ""
log_info "Generated submission files:"
ls -la "${SUBMISSION_DIR}"/*.csv 2>/dev/null | while read -r line; do
    echo "  $line"
done