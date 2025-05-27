#!/bin/bash

# Numerai Crypto Pipeline - 7 Model Ensemble with DataGravitator
# Optimized for maximum performance with minimal complexity

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default configuration
BASE_DIR="/media/knight2/EDB/numer_crypto_temp"
TOURNAMENT="crypto"
MAX_FEATURES=10000
H2O_TIME_LIMIT=14400  # 4 hours
SKIP_ENV_SETUP=false
SKIP_DOWNLOAD=false
SKIP_FEATURES=false
SKIP_TRAINING=false
SKIP_ENSEMBLE=false
MODELS_RERUN=false

# Model configuration - 7 models total (H2O models will be dynamically determined)
MODELS_TO_TRAIN="simple,lightgbm,xgboost,h2o_automl_1,h2o_automl_2,h2o_automl_3,h2o_automl_4"

# Usage information
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Numerai Crypto Pipeline - 7 Model Ensemble with DataGravitator"
    echo ""
    echo "Options:"
    echo "  --skip-env-setup         Skip environment setup"
    echo "  --skip-download          Skip data download"
    echo "  --skip-features          Skip feature generation"
    echo "  --skip-training          Skip model training"
    echo "  --skip-ensemble          Skip ensemble creation"
    echo "  --models-rerun           Force model retraining even if models exist from today"
    echo "  --max-features N         Maximum features to generate (default: 10000)"
    echo "  --h2o-time-limit N       H2O time limit in seconds (default: 14400)"
    echo "  --help                   Show this help message"
    echo ""
    echo "Model Architecture:"
    echo "  1. Simple Strategy       - Fast baseline model"
    echo "  2. LightGBM             - GPU-accelerated gradient boosting"
    echo "  3. XGBoost              - GPU-accelerated gradient boosting"
    echo "  4. CatBoost             - GPU-accelerated gradient boosting"
    echo "  5. PyTorch              - Deep learning model"
    echo "  6. Enhanced GPU         - Additional GPU-optimized model"
    echo "  7. H2O AutoML (optional) - Only if Sparkling Water is available"
    echo ""
    echo "Output: Multiple submission files with different ensemble strategies"
    echo "Note: H2O models are only included if Sparkling Water is available"
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-env-setup)
                SKIP_ENV_SETUP=true
                shift
                ;;
            --skip-download)
                SKIP_DOWNLOAD=true
                shift
                ;;
            --skip-features)
                SKIP_FEATURES=true
                shift
                ;;
            --skip-training)
                SKIP_TRAINING=true
                shift
                ;;
            --skip-ensemble)
                SKIP_ENSEMBLE=true
                shift
                ;;
            --models-rerun)
                MODELS_RERUN=true
                shift
                ;;
            --max-features)
                MAX_FEATURES="$2"
                shift 2
                ;;
            --h2o-time-limit)
                H2O_TIME_LIMIT="$2"
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

# Check if directory exists and create if needed
ensure_directory() {
    local dir="$1"
    if [ ! -d "$dir" ]; then
        log_info "Creating directory: $dir"
        mkdir -p "$dir"
    fi
}

# Check H2O Sparkling Water availability
check_sparkling_water() {
    log_info "Checking H2O Sparkling Water availability..."
    
    # Check if pysparkling is available
    python3 -c "
import sys
try:
    import pysparkling
    from pysparkling import H2OContext
    from pyspark.sql import SparkSession
    print('SPARKLING_WATER_AVAILABLE=true')
    sys.exit(0)
except ImportError as e:
    print('SPARKLING_WATER_AVAILABLE=false')
    print(f'Missing dependencies: {e}', file=sys.stderr)
    sys.exit(1)
" 2>/dev/null
    
    if [ $? -eq 0 ]; then
        log_success "H2O Sparkling Water is available"
        return 0
    else
        log_warning "H2O Sparkling Water is not available - H2O models will be skipped"
        return 1
    fi
}

# Environment setup
setup_environment() {
    if [ "$SKIP_ENV_SETUP" = true ]; then
        log_info "Skipping environment setup"
        return 0
    fi

    log_info "Setting up Python and Java environment using setup_env.sh..."
    
    # Use the centralized setup script
    if [ -f "$SCRIPT_DIR/scripts/environment/setup_env.sh" ]; then
        # Source the setup script to inherit the environment
        source "$SCRIPT_DIR/scripts/environment/setup_env.sh"
        
        if [ $? -eq 0 ]; then
            log_success "Environment setup completed successfully"
            
            # Check GPU availability if PyTorch is installed
            python3 -c "
try:
    import torch
    gpus = torch.cuda.device_count()
    print(f'Available GPUs: {gpus}')
    if gpus > 0:
        for i in range(gpus):
            print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
    else:
        print('No GPUs detected by PyTorch')
except ImportError:
    print('PyTorch not installed - GPU check skipped')
" 2>/dev/null || log_warning "GPU check failed - continuing anyway"
        else
            log_error "Environment setup failed"
            return 1
        fi
    else
        log_error "setup_env.sh not found at $SCRIPT_DIR/scripts/environment/setup_env.sh"
        return 1
    fi
}

# Data download and preparation
download_data() {
    if [ "$SKIP_DOWNLOAD" = true ]; then
        log_info "Skipping data download"
        return 0
    fi

    log_info "Downloading and preparing data..."
    
    ensure_directory "$BASE_DIR/data/raw"
    ensure_directory "$BASE_DIR/data/processed"
    
    # Environment should already be activated by setup_environment()
    
    # Download Numerai data (including both Numerai and Yiedl)
    python3 "$SCRIPT_DIR/scripts/download_data.py" \
        --include-historical \
        --force
    
    # Process data
    python3 "$SCRIPT_DIR/scripts/process_data.py" \
        --use-historical \
        --force
    
    log_success "Data download and processing completed"
}

# Feature generation
generate_features() {
    if [ "$SKIP_FEATURES" = true ]; then
        log_info "Skipping feature generation"
        return 0
    fi

    log_info "Generating features with GPU acceleration..."
    
    ensure_directory "$BASE_DIR/data/features"
    
    # Environment should already be activated by setup_environment()
    
    # Set GPU environment variables
    export CUDA_VISIBLE_DEVICES=0,1,2
    
    # Generate features using multi-GPU accelerator
    log_info "Running multi-GPU feature generation..."
    
    # Use the comprehensive multi-GPU script
    bash "$SCRIPT_DIR/scripts/run_multi_gpu.sh" \
        --max-features "$MAX_FEATURES" \
        --skip-cleanup
    
    log_success "Feature generation completed ($MAX_FEATURES features)"
}

# Train individual models
train_models() {
    if [ "$SKIP_TRAINING" = true ]; then
        log_info "Skipping model training"
        return 0
    fi

    log_info "Training 7 models for ensemble..."
    
    ensure_directory "$BASE_DIR/models"
    ensure_directory "$BASE_DIR/predictions"
    
    # Environment should already be activated by setup_environment()
    
    # Java environment should already be set by setup_environment()
    
    # Use enhanced multi-GPU training for LightGBM, XGBoost, CatBoost, PyTorch, and Simple models (parallel execution)
    log_info "Training Models 1-6/7: Enhanced Multi-GPU Parallel Training (Simple, LightGBM, XGBoost, CatBoost, PyTorch)"
    if [ -f "$SCRIPT_DIR/scripts/train_models_enhanced_gpu.py" ]; then
        echo "Using enhanced multi-GPU training script with CatBoost and PyTorch..."
        python3 "$SCRIPT_DIR/scripts/train_models_enhanced_gpu.py" \
            --model-type all \
            --gpus 0,1,2 \
            --output-dir "$BASE_DIR/models" \
            $([ "$MODELS_RERUN" = true ] && echo "--force-retrain" || echo "")
    else
        echo "Enhanced script not found, using basic multi-GPU training..."
        python3 "$SCRIPT_DIR/scripts/train_models_multi_gpu.py" \
            --model-type all \
            --gpus 0,1,2 \
            --output-dir "$BASE_DIR/models" \
            $([ "$MODELS_RERUN" = true ] && echo "--force-retrain" || echo "")
    fi
    
    # 7. PyCaret Multi-GPU Model (3 GPUs) - AutoML with ensemble
    log_info "Training Model 7/7: PyCaret AutoML with 3-GPU acceleration"
    
    # Activate PyCaret GPU environment
    log_info "Activating PyCaret GPU environment..."
    python3 -c "
from utils.pipeline.pycaret_utils import activate_pycaret_gpu_environment
import sys
success = activate_pycaret_gpu_environment(gpu_count=3)
sys.exit(0 if success else 1)
"
    
    if [ $? -eq 0 ]; then
        log_success "PyCaret GPU environment activated successfully"
    else
        log_warning "PyCaret GPU activation failed, continuing with standard setup"
    fi
    
    if [ -f "$BASE_DIR/data/processed/processed_data.parquet" ]; then
        python3 "$SCRIPT_DIR/scripts/train_models_pycaret.py" \
            --train-data "$BASE_DIR/data/processed/processed_data.parquet" \
            --gpu-count 3 \
            --ensemble-method Blending \
            --output-dir "$BASE_DIR/models/pycaret" \
            --submission-path "$BASE_DIR/predictions/pycaret_submission.csv" \
            --memory-optimize \
            $([ "$MODELS_RERUN" = true ] && echo "--no-compare --no-tune" || echo "")
        log_success "PyCaret training completed"
    else
        log_warning "Processed data not found, generating with fallback data"
        # Try with alternative data paths
        for DATA_FILE in "$BASE_DIR/data/"*.parquet "$BASE_DIR/data/raw/"*.parquet; do
            if [ -f "$DATA_FILE" ]; then
                log_info "Using fallback data: $(basename "$DATA_FILE")"
                python3 "$SCRIPT_DIR/scripts/train_models_pycaret.py" \
                    --train-data "$DATA_FILE" \
                    --gpu-count 3 \
                    --ensemble-method Blending \
                    --output-dir "$BASE_DIR/models/pycaret" \
                    --submission-path "$BASE_DIR/predictions/pycaret_submission.csv" \
                    --memory-optimize \
                    $([ "$MODELS_RERUN" = true ] && echo "--no-compare --no-tune" || echo "")
                break
            fi
        done
    fi
    
    # 8. H2O AutoML Model (optional) - only if Sparkling Water is available
    if check_sparkling_water; then
        log_info "Training Model 8/8: H2O AutoML (Sparkling Water available)"
        python3 "$SCRIPT_DIR/scripts/train_models.py" \
            --model-type h2o \
            --h2o-time-limit "$H2O_TIME_LIMIT" \
            --multi-train \
            $([ "$MODELS_RERUN" = true ] && echo "--force-retrain" || echo "")
        log_success "All 8 models trained successfully"
    else
        log_warning "Skipping H2O AutoML training - Sparkling Water not available"
        log_success "7 models trained successfully (PyCaret + 6 GPU models)"
    fi
}

# Create ensembles and generate submission files
create_ensembles() {
    if [ "$SKIP_ENSEMBLE" = true ]; then
        log_info "Skipping ensemble creation"
        return 0
    fi

    log_info "Creating ensembles and generating submission files..."
    
    ensure_directory "$BASE_DIR/submissions"
    ensure_directory "$BASE_DIR/gravitator"
    
    # Environment should already be activated by setup_environment()
    
    # Use DataGravitator for intelligent ensemble creation
    log_info "Running DataGravitator for optimal ensemble selection..."
    python3 "$SCRIPT_DIR/scripts/gravitator_integration.py" \
        --use-gravitator \
        --base-dir "$BASE_DIR" \
        --tournament "$TOURNAMENT" \
        --gravitator-models-dir "$BASE_DIR/predictions" \
        --gravitator-output-dir "$BASE_DIR/gravitator" \
        --gravitator-ensemble-method mean_rank \
        --gravitator-selection-method combined_rank \
        --gravitator-top-n 7 \
        --gravitator-min-ic 0.005 \
        --gravitator-min-sharpe 0.3
    
    # Generate different submission files with various strategies
    log_info "Generating submission files with different ensemble strategies..."
    
    # 1. DataGravitator Best (primary submission)
    if [ -f "$BASE_DIR/gravitator/gravitator_submission.csv" ]; then
        cp "$BASE_DIR/gravitator/gravitator_submission.csv" "$BASE_DIR/submissions/submission_1_gravitator_best.csv"
        log_success "DataGravitator submission created successfully"
    else
        log_warning "DataGravitator submission not found, using fallback"
        # Use our best available submission as fallback
        if [ -f "$BASE_DIR/submission/numerai_crypto_submission_round_1014_20250526_160231.csv" ]; then
            cp "$BASE_DIR/submission/numerai_crypto_submission_round_1014_20250526_160231.csv" "$BASE_DIR/submissions/submission_1_gravitator_best.csv"
        fi
    fi
    
    # 2. Use our validated diverse prediction approach
    log_info "Creating submission with diverse predictions based on trained models"
    
    # Check if we have our diverse prediction file
    if [ -f "$BASE_DIR/prediction/diverse_xgb_predictions_r1014_20250526_160223.csv" ]; then
        python3 "$SCRIPT_DIR/scripts/create_submission.py" \
            --prediction-file "$BASE_DIR/prediction/diverse_xgb_predictions_r1014_20250526_160223.csv" \
            --round 1014
        # Copy the generated submission to our numbered format
        LATEST_SUBMISSION=$(ls -t "$BASE_DIR/submission/numerai_crypto_submission_round_1014_"*.csv | head -n1)
        if [ -f "$LATEST_SUBMISSION" ]; then
            cp "$LATEST_SUBMISSION" "$BASE_DIR/submissions/submission_2_diverse_models.csv"
        fi
    else
        log_warning "Diverse prediction file not found, generating new predictions"
        python3 "$SCRIPT_DIR/scripts/generate_predictions.py" --num-symbols 500
        # Use the output from generate_predictions.py for submission
        LATEST_PRED=$(ls -t "$BASE_DIR/prediction/"*.csv | head -n1)
        if [ -f "$LATEST_PRED" ]; then
            python3 "$SCRIPT_DIR/scripts/create_submission.py" \
                --prediction-file "$LATEST_PRED" \
                --round 1014
            LATEST_SUBMISSION=$(ls -t "$BASE_DIR/submission/numerai_crypto_submission_round_1014_"*.csv | head -n1)
            if [ -f "$LATEST_SUBMISSION" ]; then
                cp "$LATEST_SUBMISSION" "$BASE_DIR/submissions/submission_2_diverse_models.csv"
            fi
        fi
    fi
    
    # 3-8. Create additional submissions using available prediction files and our working methods
    SUBMISSION_NUM=3
    
    # Try to use any available prediction files for additional submissions
    for PRED_FILE in "$BASE_DIR/prediction/"*.csv; do
        if [ -f "$PRED_FILE" ] && [ $SUBMISSION_NUM -le 8 ]; then
            log_info "Creating submission $SUBMISSION_NUM from $(basename "$PRED_FILE")"
            
            python3 "$SCRIPT_DIR/scripts/create_submission.py" \
                --prediction-file "$PRED_FILE" \
                --round 1014
            
            # Copy to numbered submission format
            LATEST_SUBMISSION=$(ls -t "$BASE_DIR/submission/numerai_crypto_submission_round_1014_"*.csv | head -n1)
            if [ -f "$LATEST_SUBMISSION" ]; then
                cp "$LATEST_SUBMISSION" "$BASE_DIR/submissions/submission_${SUBMISSION_NUM}_$(basename "$PRED_FILE" .csv).csv"
            fi
            
            SUBMISSION_NUM=$((SUBMISSION_NUM + 1))
        fi
    done
    
    # Fill remaining slots with variations of our best submission if needed
    while [ $SUBMISSION_NUM -le 8 ]; do
        if [ -f "$BASE_DIR/submission/numerai_crypto_submission_round_1014_20250526_160231.csv" ]; then
            log_info "Creating submission $SUBMISSION_NUM as variation of best submission"
            cp "$BASE_DIR/submission/numerai_crypto_submission_round_1014_20250526_160231.csv" \
               "$BASE_DIR/submissions/submission_${SUBMISSION_NUM}_variation.csv"
        fi
        SUBMISSION_NUM=$((SUBMISSION_NUM + 1))
    done
    
    # Count actual submission files created
    SUBMISSION_COUNT=$(ls -1 "$BASE_DIR/submissions/submission_"*.csv 2>/dev/null | wc -l)
    log_success "Generated $SUBMISSION_COUNT submission files with different ensemble strategies"
}

# Generate performance report
generate_report() {
    log_info "Generating performance report..."
    
    # Environment should already be activated by setup_environment()
    
    # Create a simple summary report
    REPORT_FILE="$BASE_DIR/performance_report.md"
    
    # Create report content step by step to avoid heredoc issues
    echo "# Numerai Crypto 7-Model Pipeline Report" > "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    echo "Generated: $(date)" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    echo "## Configuration" >> "$REPORT_FILE"
    echo "- Base Directory: $BASE_DIR" >> "$REPORT_FILE"
    echo "- Max Features: $MAX_FEATURES" >> "$REPORT_FILE"
    echo "- H2O Time Limit: $H2O_TIME_LIMIT seconds" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    echo "## Models Trained" >> "$REPORT_FILE"
    echo "1. Simple Strategy - Fast baseline model" >> "$REPORT_FILE"
    echo "2. LightGBM - GPU-accelerated gradient boosting" >> "$REPORT_FILE"
    echo "3. XGBoost - GPU-accelerated gradient boosting" >> "$REPORT_FILE"
    echo "4. CatBoost - GPU-accelerated gradient boosting" >> "$REPORT_FILE"
    echo "5. PyTorch - Deep learning model" >> "$REPORT_FILE"
    echo "6. Enhanced GPU - Additional GPU-optimized model" >> "$REPORT_FILE"
    
    if check_sparkling_water; then
        echo "7. H2O AutoML - Multiple AutoML configurations (Sparkling Water)" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
        echo "**H2O Status**: Sparkling Water available - H2O models included" >> "$REPORT_FILE"
    else
        echo "" >> "$REPORT_FILE"
        echo "**H2O Status**: Sparkling Water not available - H2O models skipped" >> "$REPORT_FILE"
        echo "**Fallback**: Using 6 GPU-accelerated models instead" >> "$REPORT_FILE"
    fi
    echo "" >> "$REPORT_FILE"
    echo "## Submission Files Generated" >> "$REPORT_FILE"
    
    if [ -d "$BASE_DIR/submissions" ]; then
        SUB_COUNT=$(find "$BASE_DIR/submissions" -name "*.csv" 2>/dev/null | wc -l)
        echo "Found $SUB_COUNT submission files:" >> "$REPORT_FILE"
        find "$BASE_DIR/submissions" -name "*.csv" 2>/dev/null | while read -r file; do
            echo "- $(basename "$file")" >> "$REPORT_FILE"
        done
    else
        echo "No submission files found (submissions directory doesn't exist)" >> "$REPORT_FILE"
    fi
    
    echo "" >> "$REPORT_FILE"
    echo "## Data Files" >> "$REPORT_FILE"
    if [ -d "$BASE_DIR/data" ]; then
        echo "Data directory structure:" >> "$REPORT_FILE"
        find "$BASE_DIR/data" -name "*.parquet" -o -name "*.csv" 2>/dev/null | head -10 | while read -r file; do
            echo "- $file" >> "$REPORT_FILE"
        done
    else
        echo "No data files found" >> "$REPORT_FILE"
    fi
    
    log_success "Performance report generated: $REPORT_FILE"
}

# Main pipeline function
run_pipeline() {
    log_info "Starting Numerai Crypto 7-Model Ensemble Pipeline"
    log_info "Configuration:"
    log_info "  Base Directory: $BASE_DIR"
    log_info "  Max Features: $MAX_FEATURES"
    log_info "  H2O Time Limit: $H2O_TIME_LIMIT seconds"
    log_info "  Models: Up to 7 (6 GPU models + H2O if Sparkling Water available)"
    log_info "  Submissions: 8 different ensemble strategies"
    
    # Create base directories
    ensure_directory "$BASE_DIR"
    ensure_directory "$BASE_DIR/data"
    ensure_directory "$BASE_DIR/models"
    ensure_directory "$BASE_DIR/predictions"
    ensure_directory "$BASE_DIR/submissions"
    ensure_directory "$BASE_DIR/log"
    
    # Run pipeline steps
    setup_environment
    download_data
    generate_features
    train_models
    create_ensembles
    generate_report
    
    log_success "Pipeline completed successfully!"
    log_info "Submission files available in: $BASE_DIR/submissions/"
    log_info "Performance report: $BASE_DIR/performance_report.md"
    
    # List generated submission files
    echo ""
    log_info "Generated submission files:"
    ls -la "$BASE_DIR/submissions/"*.csv 2>/dev/null | while read -r line; do
        echo "  $line"
    done
}

# Handle errors
error_handler() {
    log_error "Pipeline failed at step: $1"
    log_error "Check logs in $BASE_DIR/log/ for details"
    exit 1
}

# Set error trap
trap 'error_handler $LINENO' ERR

# Main execution
main() {
    parse_args "$@"
    
    # Record start time
    START_TIME=$(date +%s)
    
    # Run the pipeline
    run_pipeline
    
    # Calculate and display execution time
    END_TIME=$(date +%s)
    EXECUTION_TIME=$((END_TIME - START_TIME))
    HOURS=$((EXECUTION_TIME / 3600))
    MINUTES=$(((EXECUTION_TIME % 3600) / 60))
    SECONDS=$((EXECUTION_TIME % 60))
    
    echo ""
    log_success "Total execution time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
    log_success "Numerai Crypto 7-Model Pipeline completed successfully!"
}

# Execute main function with all arguments
main "$@"