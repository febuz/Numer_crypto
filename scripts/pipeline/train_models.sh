#!/bin/bash
# Pipeline script for training models for Numerai Crypto

# Import utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd ../.. && pwd)"
source "$SCRIPT_DIR/scripts/bash_utils/logging.sh"
source "$SCRIPT_DIR/scripts/bash_utils/directory.sh"

# Function to train models in pipeline
train_models_pipeline() {
    local BASE_DIR=$1
    local USE_GPU=$2
    local FORCE_RETRAIN=$3
    local GPU_MEMORY_LIMIT=$4
    local USE_AZURE_SYNAPSE=$5
    
    log_info "Step 4: Training models with GPU acceleration..."
    
    # Determine available GPUs and configure for parallel training
    if [ "$USE_GPU" = false ]; then
        log_warning "GPU usage is disabled, models will train more slowly"
        GPU_FLAG="0"
        PARALLEL_JOBS=1
    else
        # Check how many GPUs are available
        if command -v nvidia-smi &> /dev/null; then
            GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
            if [ "$GPU_COUNT" -gt 0 ]; then
                log_info "Found $GPU_COUNT GPUs available for training"
                if [ "$GPU_COUNT" -ge 3 ]; then
                    GPU_FLAG="0,1,2"
                    PARALLEL_JOBS=3
                elif [ "$GPU_COUNT" -eq 2 ]; then
                    GPU_FLAG="0,1"
                    PARALLEL_JOBS=2
                else
                    GPU_FLAG="0"
                    PARALLEL_JOBS=1
                fi
                
                # Set CPU core count for optimal parallel processing
                CPU_CORES=$(nproc)
                log_info "Detected $CPU_CORES CPU cores"
                
                # Set optimal thread count per GPU process
                if [ "$CPU_CORES" -ge $((PARALLEL_JOBS * 4)) ]; then
                    THREADS_PER_GPU=$((CPU_CORES / PARALLEL_JOBS))
                else
                    THREADS_PER_GPU=4  # Minimum 4 threads per GPU
                fi
                
                log_info "Using $PARALLEL_JOBS parallel jobs with $THREADS_PER_GPU threads per job"
                
                # Set thread limits to avoid oversubscription
                export OMP_NUM_THREADS=$THREADS_PER_GPU
                export MKL_NUM_THREADS=$THREADS_PER_GPU
                export OPENBLAS_NUM_THREADS=$THREADS_PER_GPU
                export NUMEXPR_NUM_THREADS=$THREADS_PER_GPU
                
            else
                log_warning "No GPUs detected, using CPU only"
                USE_GPU=false
                GPU_FLAG="0"
                PARALLEL_JOBS=1
            fi
        else
            log_warning "nvidia-smi not found, using CPU only"
            USE_GPU=false
            GPU_FLAG="0"
            PARALLEL_JOBS=1
        fi
    fi
    
    # Use enhanced GPU training script
    # Process features by removing random baseline features
    FEATURES_DIR="$BASE_DIR/data/features"
    log_info "Preparing features for model training..."
    
    # Create temporary directory for processed features
    temp_features_dir="${BASE_DIR}/temp_features_for_training"
    ensure_directory "${temp_features_dir}"
    
    # Create links to the feature files (to avoid copying large files)
    find "$FEATURES_DIR" -name "*.parquet" -type f | while read -r feature_file; do
        filename=$(basename "$feature_file")
        ln -sf "$feature_file" "${temp_features_dir}/${filename}"
    done
    
    log_info "Feature files linked to temporary location: ${temp_features_dir}"
    
    # Remove random baseline features before model training
    log_info "Removing random baseline features before model training..."
    
    # Use Python to remove random features
    python3 - <<EOF
import sys
import os
import glob
import logging
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('remove_random_features')

# Find all feature files
feature_dir = '${temp_features_dir}'
feature_files = glob.glob(os.path.join(feature_dir, '*.parquet'))

if not feature_files:
    logger.warning(f"No feature files found in {feature_dir}")
    sys.exit(0)

# Process each feature file
for feature_file in feature_files:
    try:
        logger.info(f"Processing file: {feature_file}")
        
        # Check if the file has random baseline features
        parquet_file = pq.ParquetFile(feature_file)
        schema = parquet_file.schema.to_arrow_schema()
        column_names = schema.names
        
        # Find random baseline columns
        random_columns = [col for col in column_names if col.startswith('random_baseline_')]
        
        if not random_columns:
            logger.info(f"No random baseline features found in {feature_file}")
            continue
            
        logger.info(f"Found {len(random_columns)} random baseline features to remove: {', '.join(random_columns)}")
        
        # Read the parquet file (try to use memory-efficient methods)
        try:
            table = pq.read_table(feature_file)
            # Remove random columns
            remaining_columns = [col for col in column_names if col not in random_columns]
            table = table.select(remaining_columns)
            
            # Write back to the same file
            pq.write_table(table, feature_file)
            logger.info(f"Successfully removed random baseline features from {feature_file}")
        except Exception as e:
            logger.error(f"Error processing file {feature_file}: {e}")
            logger.info("Attempting with pandas as fallback")
            
            # Fallback to pandas
            df = pd.read_parquet(feature_file)
            # Drop random columns
            df = df.drop(columns=random_columns)
            # Write back to the same file
            df.to_parquet(feature_file, index=False)
            logger.info(f"Successfully removed random baseline features using pandas fallback")
            
    except Exception as e:
        logger.error(f"Failed to process {feature_file}: {e}")

logger.info("Random baseline feature removal complete")
EOF

    # Check if the Python script executed successfully
    if [ $? -eq 0 ]; then
        log_success "Random baseline features removed successfully"
    else
        log_warning "Failed to remove random baseline features, continuing with training"
    fi

    # Continue with model training
    if [ -f "$SCRIPT_DIR/scripts/train_models_enhanced_gpu.py" ]; then
        FORCE_FLAG=""
        if [ "$FORCE_RETRAIN" = true ]; then
            FORCE_FLAG="--force-retrain"
        fi
        
        # Set environment variables to help with GPU memory usage
        # Don't set CUDA_VISIBLE_DEVICES as it will be managed by the Python script
        export XGB_GPU_MEMORY_LIMIT=$((GPU_MEMORY_LIMIT * 1024 * 1024 * 1024))  # Convert to bytes
        export LIGHTGBM_GPU_MEMORY_MB=$((GPU_MEMORY_LIMIT * 1024))  # Convert to MB
        
        # Azure Synapse LightGBM settings
        if [ "$USE_AZURE_SYNAPSE" = true ]; then
            log_info "Using Azure Synapse LightGBM exclusively for maximum speed"
            export USE_AZURE_SYNAPSE_LIGHTGBM=1
            export LIGHTGBM_SYNAPSE_MODE=1
            export LIGHTGBM_USE_SYNAPSE=1
            SYNAPSE_FLAG="--use-azure-synapse"
        else
            log_info "Using standard LightGBM"
            export USE_AZURE_SYNAPSE_LIGHTGBM=0
            export LIGHTGBM_SYNAPSE_MODE=0
            export LIGHTGBM_USE_SYNAPSE=0
            SYNAPSE_FLAG=""
        fi
        
        log_info "Training models with GPU memory limit: $GPU_MEMORY_LIMIT GB"
        log_info "Using $PARALLEL_JOBS parallel jobs across GPUs: $GPU_FLAG"
        
        # Clear any GPU memory cache before training
        if [ "$USE_GPU" = true ]; then
            log_info "Clearing GPU memory cache before training..."
            python3 "$SCRIPT_DIR/scripts/python_utils/gpu_utils.py" --clear-cache
        fi
        
        python3 "$SCRIPT_DIR/scripts/train_models_enhanced_gpu.py" \
            --model-type all \
            --gpus "$GPU_FLAG" \
            --output-dir "$BASE_DIR/models" \
            --features-dir "$temp_features_dir" \
            --parallel-jobs $PARALLEL_JOBS \
            $FORCE_FLAG \
            $SYNAPSE_FLAG
        
        if [ $? -eq 0 ]; then
            log_success "Model training completed successfully"
        else
            log_error "Enhanced GPU model training failed"
            return 1
        fi
    else
        log_error "Enhanced GPU training script not found"
        return 1
    fi
    
    # Count trained models
    model_count=$(find "$BASE_DIR/models" -name "*.pkl" | wc -l)
    log_info "Total models trained: $model_count"
    
    return 0
}

# If script is run directly (not sourced), execute the function with provided arguments
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Process command line arguments
    BASE_DIR=${1:-"/media/knight2/EDB/numer_crypto_temp"}
    USE_GPU=${2:-true}
    FORCE_RETRAIN=${3:-false}
    GPU_MEMORY_LIMIT=${4:-45}
    USE_AZURE_SYNAPSE=${5:-true}
    
    train_models_pipeline "$BASE_DIR" "$USE_GPU" "$FORCE_RETRAIN" "$GPU_MEMORY_LIMIT" "$USE_AZURE_SYNAPSE"
fi