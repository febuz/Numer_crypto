#!/bin/bash
# Help and usage utilities for Numerai Crypto Pipeline

# Import utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd ../.. && pwd)"
source "$SCRIPT_DIR/scripts/utils/logging.sh"
source "$SCRIPT_DIR/scripts/utils/directory.sh"

# Usage information
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Numerai Crypto Pipeline - Self-Optimizing Pipeline with Advanced Validation"
    echo ""
    echo "Options:"
    echo "  --skip-env-setup         Skip environment setup"
    echo "  --skip-download          Skip data download"
    echo "  --skip-features          Skip feature generation"
    echo "  --skip-training          Skip model training"
    echo "  --skip-ensemble          Skip ensemble creation"
    echo "  --models-rerun           Force model retraining even if models exist from today"
    echo "  --max-features N         Maximum features to generate (default: 25000)"
    echo "  --h2o-time-limit N       H2O time limit in seconds (default: 7200)"
    echo "  --optimization-cycles N  Number of optimization cycles to run (default: 3)"
    echo "  --disable-optimization   Disable automatic parameter optimization"
    echo "  --parallel-jobs N        Number of parallel jobs to run (default: 1)"
    echo "  --models-to-train LIST   Comma-separated list of models to train"
    echo "  --help                   Show this help message"
    echo ""
    echo "Feature Generation Strategy (SELF-OPTIMIZING ITERATIVE EVOLUTION):"
    echo "  🚀 GPU-accelerated feature generation with stability metrics"
    echo "  🧬 Feature stability tracking across validation folds"
    echo "  🛡️ Conservative reduce with overfitting penalty"
    echo "  🔄 Automatic parameter optimization based on feedback"
    echo "  🚀 Multiple optimization cycles for progressive improvement"
    echo "  🔬 Time series validation to prevent future leakage"
    echo "  ✅ Feature importance analysis with stability weights"
    echo "  📊 Progressive refinement based on metrics database"
    echo "  💾 Memory monitoring with automatic cleanup"
    echo "  🎯 Final output: optimized feature set with minimal overfitting"
    echo ""
    echo "All methods use random baseline features and time series cross-validation."
    echo ""
    echo "Model Architecture:"
    echo "  1. Simple Strategy       - Fast baseline model"
    echo "  2. LightGBM             - GPU-accelerated gradient boosting with stability metrics"
    echo "  3. XGBoost              - GPU-accelerated gradient boosting with overfitting controls"
    echo "  4. CatBoost             - GPU-accelerated gradient boosting (optional)"
    echo "  5. PyTorch              - Deep learning model (optional)"
    echo "  6. H2O AutoML           - Only if Sparkling Water is available"
    echo ""
    echo "Output: Multiple submission files with different ensemble strategies"
    echo "Note: H2O models are only included if Sparkling Water is available"
}

# Parse command line arguments
parse_args() {
    local -n SKIP_ENV_SETUP_REF=$1
    local -n SKIP_DOWNLOAD_REF=$2
    local -n SKIP_FEATURES_REF=$3
    local -n SKIP_TRAINING_REF=$4
    local -n SKIP_ENSEMBLE_REF=$5
    local -n MODELS_RERUN_REF=$6
    local -n MAX_FEATURES_REF=$7
    local -n H2O_TIME_LIMIT_REF=$8
    local -n OPTIMIZATION_CYCLES_REF=$9
    local -n ENABLE_OPTIMIZATION_REF=${10}
    local -n PARALLEL_JOBS_REF=${11}
    local -n MODELS_TO_TRAIN_REF=${12}
    
    # Remove the first 12 arguments (the variable references)
    shift 12
    
    # Now parse the actual command line arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --skip-env-setup)
                SKIP_ENV_SETUP_REF=true
                shift
                ;;
            --skip-download)
                SKIP_DOWNLOAD_REF=true
                shift
                ;;
            --skip-features)
                SKIP_FEATURES_REF=true
                shift
                ;;
            --skip-training)
                SKIP_TRAINING_REF=true
                shift
                ;;
            --skip-ensemble)
                SKIP_ENSEMBLE_REF=true
                shift
                ;;
            --models-rerun)
                MODELS_RERUN_REF=true
                shift
                ;;
            --max-features)
                MAX_FEATURES_REF="$2"
                shift 2
                ;;
            --h2o-time-limit)
                H2O_TIME_LIMIT_REF="$2"
                shift 2
                ;;
            --optimization-cycles)
                OPTIMIZATION_CYCLES_REF="$2"
                shift 2
                ;;
            --disable-optimization)
                ENABLE_OPTIMIZATION_REF=false
                shift
                ;;
            --parallel-jobs)
                PARALLEL_JOBS_REF="$2"
                shift 2
                ;;
            --models-to-train)
                MODELS_TO_TRAIN_REF="$2"
                shift 2
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
}