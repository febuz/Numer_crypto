#!/bin/bash
# Run Crypto Ensemble Submission v2
# This script executes the enhanced crypto ensemble submission pipeline

set -e

echo "===================================================="
echo "NUMERAI CRYPTO ENSEMBLE SUBMISSION V2"
echo "===================================================="

# Check for GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi -L
    use_gpu="--gpu"
else
    echo "No GPU detected, using CPU"
    use_gpu=""
fi

# Set working directory
cd "$(dirname "$0")"

# Create required directories
mkdir -p data/submissions
mkdir -p data/feature_store
mkdir -p models/checkpoints
mkdir -p reports

# Parse command line arguments
download_flag=""
test_first_flag=""
use_cache_flag=""
use_feature_set=""
load_model=""
best_model_flag=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --download)
            download_flag="--download"
            shift
            ;;
        --test-first)
            test_first_flag="--test-first"
            shift
            ;;
        --use-cache)
            use_cache_flag="--use-cache"
            shift
            ;;
        --feature-set)
            use_feature_set="--feature-set $2"
            shift 2
            ;;
        --load-model)
            load_model="--load-model $2"
            shift 2
            ;;
        --best-model)
            best_model_flag="--best-model"
            shift
            ;;
        *)
            # Unknown option
            echo "Unknown option: $1"
            echo "Usage: $0 [--download] [--test-first] [--use-cache] [--feature-set NAME] [--load-model NAME] [--best-model]"
            exit 1
            ;;
    esac
done

# Set timestamp for output files
timestamp=$(date +"%Y%m%d_%H%M%S")
output1="data/submissions/crypto_ensemble_${timestamp}.csv"
output2="data/submissions/crypto_ensemble_${timestamp}_v2.csv"

# Run functional tests first
if [[ -n "$test_first_flag" ]]; then
    echo "Running functional tests..."
    python scripts/run_functional_tests.py
    test_exit=$?
    if [[ $test_exit -ne 0 ]]; then
        echo "WARNING: Functional tests failed, but continuing with pipeline..."
    fi
fi

echo "Starting ensemble pipeline at $(date)"
echo "Running with parameters:"
echo "  - Feature count: 5000"
echo "  - Polynomial degree: 2"
echo "  - GPU: ${use_gpu:-off}"
echo "  - Feature Store Cache: ${use_cache_flag:-disabled}"
echo "  - Iterative pruning: enabled"
echo "  - Feature Set: ${use_feature_set:-auto}"
echo "  - Load Model: ${load_model:-none}"
echo "  - Best Model: ${best_model_flag:-disabled}"
echo "  - Output files: ${output1} and ${output2}"
echo ""

# Run the pipeline
python scripts/crypto_ensemble_submission_v2.py \
    ${download_flag} \
    ${test_first_flag} \
    ${use_gpu} \
    ${use_cache_flag} \
    ${use_feature_set} \
    ${load_model} \
    ${best_model_flag} \
    --feature-count 5000 \
    --poly-degree 2 \
    --ensemble-size 5 \
    --output "${output1}" \
    --output2 "${output2}" \
    --iterative-pruning \
    --prune-pct 0.5 \
    --random-seed 42

exit_code=$?

echo ""
echo "===================================================="
if [[ $exit_code -eq 0 ]]; then
    echo "SUBMISSION COMPLETE"
else
    echo "SUBMISSION FAILED WITH EXIT CODE ${exit_code}"
fi
echo "===================================================="

if [[ $exit_code -eq 0 ]]; then
    echo "Submissions created:"
    echo "1. ${output1}"
    echo "2. ${output2}"
    echo ""

    # Display submission statistics if available
    if [ -f "${output1}" ]; then
        echo "Submission 1 statistics:"
        echo "  - File size: $(du -h "${output1}" | cut -f1)"
        echo "  - Row count: $(($(wc -l < "${output1}") - 1))"
    fi

    if [ -f "${output2}" ]; then
        echo "Submission 2 statistics:"
        echo "  - File size: $(du -h "${output2}" | cut -f1)"
        echo "  - Row count: $(($(wc -l < "${output2}") - 1))"
    fi

    echo ""
    echo "To submit predictions to Numerai, run:"
    echo "python -c \"from numer_crypto.data.retrieval import NumeraiDataRetriever; NumeraiDataRetriever().submit_predictions('${output1}', 'crypto')\""

    # Run the analysis script on the results
    echo ""
    echo "Analyzing prediction results..."
    python scripts/analyze_crypto_predictions.py \
        --predictions "${output1}" \
        --compare "${output2}"
fi

echo ""
echo "Script completed at $(date)"
exit $exit_code