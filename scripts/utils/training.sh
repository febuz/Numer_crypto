#!/bin/bash
# Training and model utilities for Numerai Crypto Pipeline

# Import utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd ../.. && pwd)"
source "$SCRIPT_DIR/scripts/utils/logging.sh"
source "$SCRIPT_DIR/scripts/utils/directory.sh"

# Run self-optimizing pipeline
run_auto_optimizer() {
    local SKIP_FEATURES=$1
    local SKIP_TRAINING=$2
    local SKIP_ENSEMBLE=$3
    local ENABLE_OPTIMIZATION=$4
    local BASE_DIR=$5
    local ENV_DIR=${6:-"/media/knight2/EDB/repos/Numer_crypto/numer_crypto_env"}
    
    if [ "$SKIP_FEATURES" = true ] && [ "$SKIP_TRAINING" = true ] && [ "$SKIP_ENSEMBLE" = true ]; then
        log_info "Skipping all optimization steps"
        return 0
    fi

    log_info "Running self-optimizing pipeline..."
    
    # Ensure matplotlib and required modules are installed in the virtual environment
    if [ -d "$ENV_DIR" ] && [ -f "$ENV_DIR/bin/pip" ]; then
        log_info "Ensuring required packages are installed in the virtual environment..."
        "$ENV_DIR/bin/pip" install matplotlib numpy pandas polars torch lightgbm xgboost
    elif [ -d "/media/knight2/EDB/repos/Numer_crypto/numer_crypto_env" ] && [ -f "/media/knight2/EDB/repos/Numer_crypto/numer_crypto_env/bin/pip" ]; then
        log_info "Ensuring required packages are installed in numer_crypto_env..."
        "/media/knight2/EDB/repos/Numer_crypto/numer_crypto_env/bin/pip" install matplotlib numpy pandas polars torch lightgbm xgboost
    else
        log_info "Installing required packages with current pip..."
        pip install matplotlib numpy pandas polars torch lightgbm xgboost
    fi
    
    ensure_directories \
        "$BASE_DIR/data/features" \
        "$BASE_DIR/features" \
        "$BASE_DIR/metrics" \
        "$BASE_DIR/models" \
        "$BASE_DIR/predictions" \
        "$BASE_DIR/submission"
    
    # Use the enhanced_gpu training script instead of auto_optimizer
    COMMAND="python3 \"$SCRIPT_DIR/scripts/train_models_enhanced_gpu.py\" \
        --model-type all \
        --gpus 0 \
        --output-dir \"$BASE_DIR/models\" \
        --force-retrain"
        
    # Also run fast_iterative_evolution script if it exists
    if [ -f "$SCRIPT_DIR/scripts/run_fast_iterative_evolution.py" ]; then
        ITERATIVE_COMMAND="python3 \"$SCRIPT_DIR/scripts/run_fast_iterative_evolution.py\" \
            --input-file \"$BASE_DIR/data/processed/crypto_train.parquet\" \
            --output-dir \"$BASE_DIR/data/features\" \
            --max-iterations 2 \
            --features-per-iteration 5000 \
            --use-gpu"
    fi
    
    # Add optimization flag
    if [ "$ENABLE_OPTIMIZATION" = false ]; then
        COMMAND="$COMMAND --disable-optimization"
    fi
    
    # Handle skip flags
    if [ "$SKIP_FEATURES" = true ]; then
        log_info "Skipping feature generation..."
        # Skip running the iterative evolution
        ITERATIVE_COMMAND=""
    fi
    
    if [ "$SKIP_TRAINING" = true ]; then
        log_info "Skipping model training..."
        # Skip running the enhanced GPU training
        COMMAND=""
    fi
    
    if [ "$SKIP_ENSEMBLE" = true ]; then
        log_info "Skipping ensemble creation..."
    fi
    
    # Make sure we're not skipping everything
    if [ "$SKIP_FEATURES" = true ] && [ "$SKIP_TRAINING" = true ] && [ "$SKIP_ENSEMBLE" = true ]; then
        log_warning "All main steps (features, training, ensemble) are being skipped!"
        log_warning "Please enable at least one step or remove the skip flags."
    fi
    
    # Run the enhanced GPU training command if not skipped
    if [ -n "$COMMAND" ]; then
        log_info "Running enhanced GPU training command..."
        log_info "Command: $COMMAND"
        
        # First check if there are any existing models
        EXISTING_MODELS=$(find "$BASE_DIR/models" -name "*.pkl" -type f | wc -l)
        if [ $EXISTING_MODELS -gt 0 ]; then
            log_info "Found $EXISTING_MODELS existing model files before training"
        fi
        
        # Check if processed data exists
        if [ ! -f "$BASE_DIR/data/processed/crypto_train.parquet" ]; then
            log_warning "Training data file crypto_train.parquet not found! Training may fail."
        else
            log_info "Found training data file. Training should work."
            # Get file size
            DATA_FILE_SIZE=$(du -h "$BASE_DIR/data/processed/crypto_train.parquet" | cut -f1)
            log_info "Training data file size: $DATA_FILE_SIZE"
        fi
        
        # Execute the command
        eval "$COMMAND"
        
        # Check the result
        TRAINING_RESULT=$?
        if [ $TRAINING_RESULT -eq 0 ]; then
            log_success "Training command completed successfully"
            
            # Count new models
            NEW_MODELS=$(find "$BASE_DIR/models" -name "*.pkl" -type f | wc -l)
            if [ $NEW_MODELS -gt $EXISTING_MODELS ]; then
                MODELS_CREATED=$((NEW_MODELS - EXISTING_MODELS))
                log_success "Created $MODELS_CREATED new model files during training"
            else
                log_warning "No new model files detected after training"
            fi
        else
            log_error "Training command failed with exit code $TRAINING_RESULT"
        fi
    else
        log_info "Skipping enhanced GPU training command (was set to empty)"
    fi
    
    # Run the iterative evolution command if available and not skipped
    if [ -n "$ITERATIVE_COMMAND" ]; then
        log_info "Running fast iterative evolution command..."
        log_info "Command: $ITERATIVE_COMMAND"
        
        # First check if there are any existing feature files
        EXISTING_FEATURES=$(find "$BASE_DIR/data/features" -name "*.parquet" -type f | wc -l)
        if [ $EXISTING_FEATURES -gt 0 ]; then
            log_info "Found $EXISTING_FEATURES existing feature files before evolution"
        fi
        
        # Execute the command
        eval "$ITERATIVE_COMMAND"
        
        # Check the result
        EVOLUTION_RESULT=$?
        if [ $EVOLUTION_RESULT -eq 0 ]; then
            log_success "Feature evolution command completed successfully"
            
            # Count new feature files
            NEW_FEATURES=$(find "$BASE_DIR/data/features" -name "*.parquet" -type f | wc -l)
            if [ $NEW_FEATURES -gt $EXISTING_FEATURES ]; then
                FEATURES_CREATED=$((NEW_FEATURES - EXISTING_FEATURES))
                log_success "Created $FEATURES_CREATED new feature files during evolution"
            else
                log_warning "No new feature files detected after evolution"
            fi
        else
            log_error "Feature evolution command failed with exit code $EVOLUTION_RESULT"
        fi
    else
        log_info "Skipping fast iterative evolution (not available or skipped)"
    fi
}