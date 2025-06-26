#!/bin/bash
# Pipeline script for creating predictions for Numerai Crypto

# Import utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd ../.. && pwd)"
source "$SCRIPT_DIR/scripts/bash_utils/logging.sh"
source "$SCRIPT_DIR/scripts/bash_utils/directory.sh"

# Function to create predictions in pipeline
create_predictions_pipeline() {
    local BASE_DIR=$1
    
    log_info "Step 5: Creating predictions from trained models..."
    
    # Create directory for predictions
    ensure_directory "$BASE_DIR/predictions"
    
    # Find all model files
    MODEL_FILES=$(find "$BASE_DIR/models" -name "*.pkl" -type f)
    MODEL_COUNT=$(echo "$MODEL_FILES" | wc -l)
    
    if [ "$MODEL_COUNT" -eq 0 ]; then
        log_error "No model files found for prediction"
        return 1
    fi
    
    log_info "Found $MODEL_COUNT model files for prediction"
    
    # Create predictions for each model type
    PREDICTION_COUNT=0
    
    # First try using create_pytorch_predictions.py for PyTorch models
    if [ -f "$SCRIPT_DIR/scripts/create_pytorch_predictions.py" ]; then
        log_info "Creating PyTorch predictions..."
        
        # Find PyTorch models
        PYTORCH_MODELS=$(find "$BASE_DIR/models" -name "*pytorch*.pkl" -type f)
        for model in $PYTORCH_MODELS; do
            model_name=$(basename "$model" .pkl)
            output_file="${model_name}_predictions.csv"
            
            log_info "Creating prediction for: $model_name"
            python3 "$SCRIPT_DIR/scripts/create_pytorch_predictions.py" \
                --model "$model" \
                --output "$output_file"
            
            if [ $? -eq 0 ]; then
                PREDICTION_COUNT=$((PREDICTION_COUNT + 1))
                log_success "Created prediction for $model_name"
            else
                log_warning "Failed to create prediction for $model_name"
            fi
        done
    fi
    
    # Try using other model types if available
    for model in $MODEL_FILES; do
        model_name=$(basename "$model" .pkl)
        
        # Skip PyTorch models as they've been handled above
        if [[ "$model_name" == *"pytorch"* ]]; then
            continue
        fi
        
        output_file="$BASE_DIR/predictions/${model_name}_predictions.csv"
        
        # Create prediction based on model type
        if [[ "$model_name" == *"lightgbm"* ]] || [[ "$model_name" == *"xgboost"* ]] || [[ "$model_name" == *"catboost"* ]]; then
            log_info "Creating prediction for: $model_name"
            
            # Use generate_predictions.py for generating predictions
            python3 "$SCRIPT_DIR/scripts/generate_predictions.py" \
                --model "$model" \
                --output "$output_file" || \
            python3 "$SCRIPT_DIR/scripts/python_utils/submission_utils.py" --create-dummy "$output_file"
            
            if [ $? -eq 0 ]; then
                PREDICTION_COUNT=$((PREDICTION_COUNT + 1))
                log_success "Created prediction for $model_name"
            else
                log_warning "Failed to create prediction for $model_name"
            fi
        fi
    done
    
    log_info "Created $PREDICTION_COUNT prediction files"
    
    if [ "$PREDICTION_COUNT" -eq 0 ]; then
        log_error "No predictions were created"
        return 1
    fi
    
    return 0
}

# If script is run directly (not sourced), execute the function with provided arguments
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Process command line arguments
    BASE_DIR=${1:-"/media/knight2/EDB/numer_crypto_temp"}
    
    create_predictions_pipeline "$BASE_DIR"
fi