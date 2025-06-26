#!/bin/bash
# Main pipeline orchestration script for Numerai Crypto

# Import utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd ../.. && pwd)"
source "$SCRIPT_DIR/scripts/bash_utils/logging.sh"
source "$SCRIPT_DIR/scripts/bash_utils/directory.sh"
source "$SCRIPT_DIR/scripts/bash_utils/environment.sh"

# Function to run the complete pipeline
run_pipeline() {
    local BASE_DIR=$1
    local SKIP_DOWNLOAD=$2
    local SKIP_YIEDL=$3
    local FORCE_REPROCESS=$4
    local FORCE_RETRAIN=$5
    local USE_GPU=$6
    local MAX_ITERATIONS=$7
    local FEATURES_PER_ITERATION=$8
    local GPU_MEMORY_LIMIT=$9
    local USE_AZURE_SYNAPSE=${10}
    local INCLUDE_HISTORICAL=${11}
    local USE_RANDOM_FEATURES=${12}

    # Record start time
    START_TIME=$(date +%s)
    
    # Create log file for this run
    LOG_DIR="$BASE_DIR/log"
    ensure_directory "$LOG_DIR"
    LOG_FILE="$LOG_DIR/pipeline_$(date +%Y%m%d_%H%M%S).log"
    
    # Export important variables for Python scripts
    export NUMERAI_DATA_DIR="$BASE_DIR/data/numerai"
    export YIEDL_DATA_DIR="$BASE_DIR/data/yiedl"
    export PROCESSED_DATA_DIR="$BASE_DIR/data/processed"
    export FEATURES_DIR="$BASE_DIR/data/features"
    export MODELS_DIR="$BASE_DIR/models"
    export PREDICTIONS_DIR="$BASE_DIR/predictions"
    export SUBMISSION_DIR="$BASE_DIR/submission"
    export DATA_DIR="$BASE_DIR/data"
    
    # Log all data directories for debugging
    log_info "Using the following data directories:"
    log_info "BASE_DIR=$BASE_DIR"
    log_info "DATA_DIR=$DATA_DIR"
    log_info "NUMERAI_DATA_DIR=$NUMERAI_DATA_DIR"
    log_info "YIEDL_DATA_DIR=$YIEDL_DATA_DIR"
    log_info "PROCESSED_DATA_DIR=$PROCESSED_DATA_DIR"
    log_info "FEATURES_DIR=$FEATURES_DIR"
    log_info "MODELS_DIR=$MODELS_DIR"
    log_info "PREDICTIONS_DIR=$PREDICTIONS_DIR"
    log_info "SUBMISSION_DIR=$SUBMISSION_DIR"
    
    # Fix permissions for all data directories (using aggressive mode)
    log_info "Ensuring proper permissions for all data directories (aggressive mode)..."
    fix_permissions "$BASE_DIR" "true"
    
    # Run each step and exit on failure
    {
        log_info "Starting Numerai Crypto Pipeline"
        log_info "=================================="
        log_info "Base Directory: $BASE_DIR"
        log_info "Using GPU: $USE_GPU"
        log_info "Force Reprocess: $FORCE_REPROCESS"
        log_info "Force Retrain: $FORCE_RETRAIN"
        log_info "Max Iterations: $MAX_ITERATIONS"
        log_info "Features Per Iteration: $FEATURES_PER_ITERATION"
        log_info "Skip Yiedl: $SKIP_YIEDL"
        log_info "Include Historical: $INCLUDE_HISTORICAL"
        log_info "Skip Download: $SKIP_DOWNLOAD"
        log_info "GPU Memory Limit: $GPU_MEMORY_LIMIT GB"
        log_info "Use Azure Synapse LightGBM: $USE_AZURE_SYNAPSE"
        echo ""
        
        # Step 1: Download data
        log_info "Sourcing download_data.sh..."
        source "$SCRIPT_DIR/scripts/pipeline/download_data.sh"
        log_info "Running download_data_pipeline..."
        if ! download_data_pipeline "$BASE_DIR" "$SKIP_DOWNLOAD" "$SKIP_YIEDL" "$INCLUDE_HISTORICAL"; then
            log_error "Pipeline failed at step 1: Download data"
            exit 1
        fi
        echo ""
        
        # Step 2: Process data
        log_info "Sourcing process_data.sh..."
        source "$SCRIPT_DIR/scripts/pipeline/process_data.sh"
        log_info "Running process_data_pipeline..."
        if ! process_data_pipeline "$BASE_DIR" "$FORCE_REPROCESS" "$INCLUDE_HISTORICAL"; then
            log_error "Pipeline failed at step 2: Process data"
            exit 1
        fi
        echo ""
        
        # Step 3: Generate features
        log_info "Sourcing generate_features.sh..."
        source "$SCRIPT_DIR/scripts/pipeline/generate_features.sh"
        log_info "Running generate_features_pipeline..."
        # Pass random features flag
        USE_RANDOM_FEATURES=${12}
        log_info "Random baseline features: $USE_RANDOM_FEATURES"
        
        if ! generate_features_pipeline "$BASE_DIR" "$USE_GPU" "$MAX_ITERATIONS" "$FEATURES_PER_ITERATION" "$GPU_MEMORY_LIMIT" "$USE_RANDOM_FEATURES"; then
            log_error "Pipeline failed at step 3: Generate features"
            exit 1
        fi
        echo ""
        
        # Step 4: Train models
        log_info "Sourcing train_models.sh..."
        source "$SCRIPT_DIR/scripts/pipeline/train_models.sh"
        log_info "Running train_models_pipeline..."
        if ! train_models_pipeline "$BASE_DIR" "$USE_GPU" "$FORCE_RETRAIN" "$GPU_MEMORY_LIMIT" "$USE_AZURE_SYNAPSE"; then
            log_error "Pipeline failed at step 4: Train models"
            exit 1
        fi
        echo ""
        
        # Step 5: Create predictions
        log_info "Sourcing create_predictions.sh..."
        source "$SCRIPT_DIR/scripts/pipeline/create_predictions.sh"
        log_info "Running create_predictions_pipeline..."
        if ! create_predictions_pipeline "$BASE_DIR"; then
            log_error "Pipeline failed at step 5: Create predictions"
            exit 1
        fi
        echo ""
        
        # Step 6: Create submission files
        log_info "Sourcing create_submission_files.sh..."
        source "$SCRIPT_DIR/scripts/pipeline/create_submission_files.sh"
        log_info "Running create_submission_files_pipeline..."
        if ! create_submission_files_pipeline "$BASE_DIR"; then
            log_error "Pipeline failed at step 6: Create submission files"
            exit 1
        fi
        
        # Calculate and display execution time
        END_TIME=$(date +%s)
        EXECUTION_TIME=$((END_TIME - START_TIME))
        HOURS=$((EXECUTION_TIME / 3600))
        MINUTES=$(((EXECUTION_TIME % 3600) / 60))
        SECONDS=$((EXECUTION_TIME % 60))
        
        # Count models and submissions
        MODEL_COUNT=$(find "$BASE_DIR/models" -name "*.pkl" | wc -l)
        SUBMISSION_COUNT=$(find "$BASE_DIR/submission" -name "*.csv" | wc -l)
        
        echo ""
        log_success "Pipeline completed successfully!"
        log_success "Total execution time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
        log_info "Models created: $MODEL_COUNT"
        log_info "Submissions created: $SUBMISSION_COUNT"
        log_info "Submission files available in: $BASE_DIR/submission/"
        
        # Record metrics if metrics module is available
        if [ -f "$SCRIPT_DIR/utils/metrics/metrics_db.py" ]; then
            python3 "$SCRIPT_DIR/scripts/python_utils/metrics_utils.py" \
                --add-run \
                --db-path "$BASE_DIR/metrics/metrics.db" \
                --description "Pipeline run with $MODEL_COUNT models and $SUBMISSION_COUNT submissions" \
                --execution-time $EXECUTION_TIME \
                --num-models $MODEL_COUNT \
                --num-submissions $SUBMISSION_COUNT
        fi
        
    } | tee -a "$LOG_FILE"
    
    # Track metrics if available
    if [ -f "$SCRIPT_DIR/utils/metrics/metrics_db.py" ]; then
        log_info "Tracking performance metrics using metrics database..."
        
        # Use metrics database to track performance
        # Ensure we're using the Python from the virtual environment
        if [ -d "/media/knight2/EDB/numer_crypto_temp/venv/numer_crypto_env" ] && [ -f "/media/knight2/EDB/numer_crypto_temp/venv/numer_crypto_env/bin/python3" ]; then
            PYTHON_CMD="/media/knight2/EDB/numer_crypto_temp/venv/numer_crypto_env/bin/python3"
        else
            PYTHON_CMD="python3"
        fi
        
        # Display recent runs from metrics database
        $PYTHON_CMD "$SCRIPT_DIR/scripts/python_utils/metrics_utils.py" \
            --display-runs \
            --db-path "$BASE_DIR/metrics/metrics.db"
    fi
}

# If script is run directly (not sourced), execute the function with provided arguments
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Default values
    BASE_DIR="/media/knight2/EDB/numer_crypto_temp"
    SKIP_DOWNLOAD=false
    SKIP_YIEDL=false
    FORCE_REPROCESS=false
    FORCE_RETRAIN=false
    USE_GPU=true
    MAX_ITERATIONS=1
    FEATURES_PER_ITERATION=7500
    GPU_MEMORY_LIMIT=45
    USE_AZURE_SYNAPSE=true
    INCLUDE_HISTORICAL=true
    USE_RANDOM_FEATURES=true
    
    # Debug - print all arguments
    log_info "Arguments received: $@"
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --base-dir)
                BASE_DIR="$2"
                log_info "Setting BASE_DIR=$BASE_DIR"
                shift 2
                ;;
            --skip-download)
                SKIP_DOWNLOAD=true
                log_info "Setting SKIP_DOWNLOAD=true"
                shift
                ;;
            --skip-yiedl)
                SKIP_YIEDL=true
                log_info "Setting SKIP_YIEDL=true"
                shift
                ;;
            --force-reprocess)
                FORCE_REPROCESS=true
                log_info "Setting FORCE_REPROCESS=true"
                shift
                ;;
            --force-retrain)
                FORCE_RETRAIN=true
                log_info "Setting FORCE_RETRAIN=true"
                shift
                ;;
            --no-gpu)
                USE_GPU=false
                log_info "Setting USE_GPU=false"
                shift
                ;;
            --max-iterations)
                if [[ -n "$2" && "$2" != --* ]]; then
                    MAX_ITERATIONS="$2"
                    log_info "Setting MAX_ITERATIONS=$MAX_ITERATIONS"
                    shift 2
                else
                    log_error "Missing value for --max-iterations"
                    shift
                fi
                ;;
            --features-per-iteration)
                if [[ -n "$2" && "$2" != --* ]]; then
                    FEATURES_PER_ITERATION="$2"
                    log_info "Setting FEATURES_PER_ITERATION=$FEATURES_PER_ITERATION"
                    shift 2
                else
                    log_error "Missing value for --features-per-iteration"
                    shift
                fi
                ;;
            --gpu-memory)
                if [[ -n "$2" && "$2" != --* ]]; then
                    GPU_MEMORY_LIMIT="$2"
                    log_info "Setting GPU_MEMORY_LIMIT=$GPU_MEMORY_LIMIT"
                    shift 2
                else
                    log_error "Missing value for --gpu-memory"
                    shift
                fi
                ;;
            --no-azure-synapse)
                USE_AZURE_SYNAPSE=false
                log_info "Setting USE_AZURE_SYNAPSE=false"
                shift
                ;;
            --skip-historical)
                INCLUDE_HISTORICAL=false
                log_info "Setting INCLUDE_HISTORICAL=false"
                shift
                ;;
            --no-random-baselines)
                USE_RANDOM_FEATURES=false
                log_info "Setting USE_RANDOM_FEATURES=false"
                shift
                ;;
            --use-random-baselines)
                USE_RANDOM_FEATURES=true
                log_info "Setting USE_RANDOM_FEATURES=true"
                shift
                ;;
            --*)
                log_warning "Unknown option: $1 (skipping)"
                shift
                ;;
            *)
                log_warning "Unknown argument: $1 (skipping)"
                shift
                ;;
        esac
    done
    
    log_info "Starting pipeline with following parameters:"
    log_info "BASE_DIR: $BASE_DIR"
    log_info "SKIP_DOWNLOAD: $SKIP_DOWNLOAD"
    log_info "SKIP_YIEDL: $SKIP_YIEDL"
    log_info "FORCE_REPROCESS: $FORCE_REPROCESS"
    log_info "FORCE_RETRAIN: $FORCE_RETRAIN"
    log_info "USE_GPU: $USE_GPU"
    log_info "MAX_ITERATIONS: $MAX_ITERATIONS"
    log_info "FEATURES_PER_ITERATION: $FEATURES_PER_ITERATION"
    log_info "GPU_MEMORY_LIMIT: $GPU_MEMORY_LIMIT"
    log_info "USE_AZURE_SYNAPSE: $USE_AZURE_SYNAPSE"
    log_info "INCLUDE_HISTORICAL: $INCLUDE_HISTORICAL"
    log_info "USE_RANDOM_FEATURES: $USE_RANDOM_FEATURES"
    
    # Run the pipeline with the specified parameters
    run_pipeline \
        "$BASE_DIR" \
        "$SKIP_DOWNLOAD" \
        "$SKIP_YIEDL" \
        "$FORCE_REPROCESS" \
        "$FORCE_RETRAIN" \
        "$USE_GPU" \
        "$MAX_ITERATIONS" \
        "$FEATURES_PER_ITERATION" \
        "$GPU_MEMORY_LIMIT" \
        "$USE_AZURE_SYNAPSE" \
        "$INCLUDE_HISTORICAL" \
        "$USE_RANDOM_FEATURES"
fi