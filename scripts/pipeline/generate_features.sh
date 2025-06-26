#!/bin/bash
# Pipeline script for generating features for Numerai Crypto

# Import utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd ../.. && pwd)"
source "$SCRIPT_DIR/scripts/bash_utils/logging.sh"
source "$SCRIPT_DIR/scripts/bash_utils/directory.sh"

# Function to generate evolved features in pipeline
generate_features_pipeline() {
    local BASE_DIR=$1
    local USE_GPU=$2
    local MAX_ITERATIONS=$3
    local FEATURES_PER_ITERATION=$4
    local GPU_MEMORY_LIMIT=$5
    local USE_RANDOM_FEATURES=$6
    
    log_info "Step 3: Generating evolved features..."
    
    # Check if the fast iterative evolution script is available
    if [ -f "$SCRIPT_DIR/scripts/run_fast_iterative_evolution.py" ]; then
        log_info "Using fast iterative feature evolution..."
        
        # Determine GPU configuration for feature generation
        if [ "$USE_GPU" = false ]; then
            log_warning "GPU usage is disabled for feature generation"
            GPU_FLAG="--no-gpu"
            GPU_DEVICE=""
        else
            # Determine available GPUs
            if command -v nvidia-smi &> /dev/null; then
                GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
                if [ "$GPU_COUNT" -gt 0 ]; then
                    log_info "Found $GPU_COUNT GPUs available for feature generation"
                    GPU_FLAG="--use-gpu"
                    # Use GPU 0 for feature generation
                    GPU_DEVICE="0"
                    export CUDA_VISIBLE_DEVICES="0"
                else
                    log_warning "No GPUs detected, using CPU only for feature generation"
                    GPU_FLAG="--no-gpu"
                    GPU_DEVICE=""
                fi
            else
                log_warning "nvidia-smi not found, using CPU only for feature generation"
                GPU_FLAG="--no-gpu"
                GPU_DEVICE=""
            fi
        fi
        
        log_info "GPU memory limit: $GPU_MEMORY_LIMIT GB"
        
        # Set CPU thread count for optimal performance
        CPU_CORES=$(nproc)
        THREADS=$((CPU_CORES / 2 > 4 ? CPU_CORES / 2 : 4))  # Use half of cores, minimum 4
        log_info "Using $THREADS CPU threads for feature processing"
        
        export OMP_NUM_THREADS=$THREADS
        export MKL_NUM_THREADS=$THREADS
        export OPENBLAS_NUM_THREADS=$THREADS
        export NUMEXPR_NUM_THREADS=$THREADS
        
        # Set small batch size to avoid OOM issues
        BATCH_SIZE=5000
        
        # Clear GPU memory cache before feature generation
        if [ "$USE_GPU" = true ]; then
            log_info "Clearing GPU memory cache before feature generation..."
            python3 "$SCRIPT_DIR/scripts/python_utils/gpu_utils.py" --clear-cache
        fi
        
        # Process random features flag
        if [ "$USE_RANDOM_FEATURES" = true ]; then
            RANDOM_FEATURES_FLAG="--use-random-baselines"
            log_info "Random baseline features enabled for comparison"
        else
            RANDOM_FEATURES_FLAG="--no-random-baselines"
            log_info "Random baseline features disabled"
        fi
        
        # Set the sample size to 0 to use the full dataset (never use sample)
        python3 "$SCRIPT_DIR/scripts/run_fast_iterative_evolution.py" \
            --input-file "$BASE_DIR/data/processed/crypto_train.parquet" \
            --output-dir "$BASE_DIR/data/features" \
            --max-iterations "$MAX_ITERATIONS" \
            --features-per-iteration "$FEATURES_PER_ITERATION" \
            --memory-limit-gb 500 \
            --sample-size 0 \
            $GPU_FLAG \
            $RANDOM_FEATURES_FLAG
        
        if [ $? -eq 0 ]; then
            log_success "Feature evolution completed successfully"
        else
            log_error "Feature evolution failed"
            return 1
        fi
    else
        log_warning "Fast iterative evolution script not found, skipping feature evolution"
    fi
    
    # Ensure features directory exists with proper permissions
    ensure_directory "$BASE_DIR/data/features"
    chmod -R 777 "$BASE_DIR/data/features" 2>/dev/null || true
    
    # Check if we have features
    feature_files=$(find "$BASE_DIR/data/features" -name "*.parquet" | wc -l)
    if [ "$feature_files" -eq 0 ]; then
        log_warning "No feature files found. Checking for processed data to use directly for training."
        if [ -f "$BASE_DIR/data/processed/crypto_train.parquet" ]; then
            log_info "Using processed data directly for training."
            # First ensure proper permissions
            chmod 666 "$BASE_DIR/data/processed/crypto_train.parquet" 2>/dev/null || true
            # Copy to features directory
            cp "$BASE_DIR/data/processed/crypto_train.parquet" "$BASE_DIR/data/features/default_features.parquet"
            # Ensure proper permissions on the copied file
            chmod 666 "$BASE_DIR/data/features/default_features.parquet" 2>/dev/null || true
            log_success "Successfully copied processed data to features directory"
        else
            log_error "No processed data file found at $BASE_DIR/data/processed/crypto_train.parquet"
            log_error "Cannot continue without either feature files or processed training data."
            return 1
        fi
    fi
    
    return 0
}

# If script is run directly (not sourced), execute the function with provided arguments
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Process command line arguments
    BASE_DIR=${1:-"/media/knight2/EDB/numer_crypto_temp"}
    USE_GPU=${2:-true}
    MAX_ITERATIONS=${3:-1}
    FEATURES_PER_ITERATION=${4:-7500}
    GPU_MEMORY_LIMIT=${5:-45}
    
    generate_features_pipeline "$BASE_DIR" "$USE_GPU" "$MAX_ITERATIONS" "$FEATURES_PER_ITERATION" "$GPU_MEMORY_LIMIT"
fi