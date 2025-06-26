#!/bin/bash
# Data management utilities for Numerai Crypto Pipeline

# Import utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd ../.. && pwd)"
source "$SCRIPT_DIR/scripts/bash_utils/logging.sh"
source "$SCRIPT_DIR/scripts/bash_utils/directory.sh"

# Function to download data only
download_data() {
    local SKIP_DOWNLOAD=$1
    local BASE_DIR=$2
    local INCLUDE_HISTORICAL=${3:-true}
    local SKIP_YIEDL=${4:-false}
    
    if [ "$SKIP_DOWNLOAD" = true ]; then
        log_info "Skipping data download"
        return 0
    fi

    log_info "Downloading data..."
    
    # Create necessary directories with proper permissions
    ensure_directory "$BASE_DIR/data/raw"
    ensure_directory "$BASE_DIR/data/numerai"
    ensure_directory "$BASE_DIR/data/yiedl"
    
    # Fix permissions explicitly for data directories
    log_info "Ensuring proper permissions for data directories..."
    chmod -R 777 "$BASE_DIR/data/numerai" 2>/dev/null || true
    chmod -R 777 "$BASE_DIR/data/yiedl" 2>/dev/null || true
    chmod -R 777 "$BASE_DIR/data/raw" 2>/dev/null || true
    
    # Prepare download flags
    local DOWNLOAD_FLAGS="--force"
    if [ "$INCLUDE_HISTORICAL" = true ]; then
        DOWNLOAD_FLAGS="$DOWNLOAD_FLAGS --include-historical"
    else
        DOWNLOAD_FLAGS="$DOWNLOAD_FLAGS --skip-historical"
    fi
    
    if [ "$SKIP_YIEDL" = true ]; then
        DOWNLOAD_FLAGS="$DOWNLOAD_FLAGS --numerai-only"
    fi
    
    # Try main download script first - check possible locations
    if [ -f "$SCRIPT_DIR/scripts/data/download_data.py" ]; then
        DOWNLOAD_SCRIPT="$SCRIPT_DIR/scripts/data/download_data.py"
    elif [ -f "$SCRIPT_DIR/scripts/download_data.py" ]; then
        DOWNLOAD_SCRIPT="$SCRIPT_DIR/scripts/download_data.py"
    else
        DOWNLOAD_SCRIPT=$(find "$SCRIPT_DIR" -name "download_data.py" | head -1)
    fi
    
    # Make sure the base directories exist with correct permissions
    mkdir -p "$BASE_DIR/data/yiedl" 2>/dev/null
    mkdir -p "$BASE_DIR/data/numerai" 2>/dev/null
    chmod -R 777 "$BASE_DIR/data" 2>/dev/null || true
    
    # Set environment variables to ensure correct paths are used
    export NUMERAI_DATA_DIR="$BASE_DIR/data/numerai"
    export YIEDL_DATA_DIR="$BASE_DIR/data/yiedl"
    
    log_info "Running download_data.py script ($DOWNLOAD_SCRIPT) with flags: $DOWNLOAD_FLAGS"
    log_info "Using data directories: NUMERAI_DATA_DIR=$NUMERAI_DATA_DIR, YIEDL_DATA_DIR=$YIEDL_DATA_DIR"
    
    if [ -f "$DOWNLOAD_SCRIPT" ]; then
        if python3 "$DOWNLOAD_SCRIPT" $DOWNLOAD_FLAGS; then
            log_success "Successfully downloaded data"
            return 0
        else
            log_error "Failed to download data with main script"
        fi
    else
        log_error "Could not find download_data.py script"
        
        # Try fallback downloaders with correct environment variables
        export NUMERAI_DATA_DIR="$BASE_DIR/data/numerai"
        export YIEDL_DATA_DIR="$BASE_DIR/data/yiedl"
        export DATA_DIR="$BASE_DIR/data"
        
        # Find fallback downloaders
        YIEDL_DOWNLOADER=""
        NUMERAI_DOWNLOADER=""
        
        if [ -f "$SCRIPT_DIR/utils/data/download_yiedl_modified.py" ]; then
            YIEDL_DOWNLOADER="$SCRIPT_DIR/utils/data/download_yiedl_modified.py"
        else
            YIEDL_DOWNLOADER=$(find "$SCRIPT_DIR" -name "download_yiedl*.py" | head -1)
        fi
        
        if [ -f "$SCRIPT_DIR/utils/data/download_numerai.py" ]; then
            NUMERAI_DOWNLOADER="$SCRIPT_DIR/utils/data/download_numerai.py"
        else
            NUMERAI_DOWNLOADER=$(find "$SCRIPT_DIR" -name "download_numerai*.py" | head -1)
        fi
        
        log_info "Found Yiedl downloader: $YIEDL_DOWNLOADER"
        log_info "Found Numerai downloader: $NUMERAI_DOWNLOADER"
        
        if [ -n "$YIEDL_DOWNLOADER" ] && [ "$SKIP_YIEDL" = false ]; then
            log_info "Trying Yiedl downloader..."
            log_info "Using data directories: YIEDL_DATA_DIR=$YIEDL_DATA_DIR"
            if [ "$INCLUDE_HISTORICAL" = true ]; then
                python3 "$YIEDL_DOWNLOADER" --force --include-historical
            else
                python3 "$YIEDL_DOWNLOADER" --force
            fi
        fi
        
        if [ -n "$NUMERAI_DOWNLOADER" ]; then
            log_info "Trying Numerai downloader..."
            log_info "Using data directories: NUMERAI_DATA_DIR=$NUMERAI_DATA_DIR"
            if [ "$INCLUDE_HISTORICAL" = true ]; then
                python3 "$NUMERAI_DOWNLOADER" --include-historical
            else
                python3 "$NUMERAI_DOWNLOADER"
            fi
        fi
        
        # Check if we at least have the numerai data
        if [ -f "$BASE_DIR/data/raw/numerai_live.parquet" ] || [ -f "$BASE_DIR/data/raw/numerai_live.csv" ]; then
            log_warning "Downloaded some data using fallback methods"
            return 0
        else
            log_error "Failed to download required data even with fallback methods"
            return 1
        fi
    fi
}

# Function to process data only
process_data() {
    local BASE_DIR=$1
    local FORCE_REPROCESS=$2
    local USE_HISTORICAL=${3:-true}
    
    log_info "Processing data..."
    
    # Create necessary directory with proper permissions
    ensure_directory "$BASE_DIR/data/processed"
    
    # Fix permissions explicitly for processed data directory
    log_info "Ensuring proper permissions for processed data directory..."
    chmod -R 777 "$BASE_DIR/data/processed" 2>/dev/null || true
    
    # Set environment variables to ensure correct paths are used
    export NUMERAI_DATA_DIR="$BASE_DIR/data/numerai"
    export YIEDL_DATA_DIR="$BASE_DIR/data/yiedl"
    export PROCESSED_DATA_DIR="$BASE_DIR/data/processed"
    export DATA_DIR="$BASE_DIR/data"
    
    log_info "Using data directories:"
    log_info "NUMERAI_DATA_DIR=$NUMERAI_DATA_DIR"
    log_info "YIEDL_DATA_DIR=$YIEDL_DATA_DIR"
    log_info "PROCESSED_DATA_DIR=$PROCESSED_DATA_DIR"
    
    # Prepare processing flags
    local PROCESS_FLAGS=""
    if [ "$FORCE_REPROCESS" = true ]; then
        PROCESS_FLAGS="$PROCESS_FLAGS --force"
    fi
    
    if [ "$USE_HISTORICAL" = true ]; then
        PROCESS_FLAGS="$PROCESS_FLAGS --use-historical"
    fi
    
    # Find the process data scripts
    PROCESS_POLARS_SCRIPT=""
    PROCESS_STANDARD_SCRIPT=""
    
    if [ -f "$SCRIPT_DIR/scripts/data/process_data_polars.py" ]; then
        PROCESS_POLARS_SCRIPT="$SCRIPT_DIR/scripts/data/process_data_polars.py"
    else
        PROCESS_POLARS_SCRIPT=$(find "$SCRIPT_DIR" -name "process_data_polars.py" | head -1)
    fi
    
    if [ -f "$SCRIPT_DIR/scripts/data/process_data.py" ]; then
        PROCESS_STANDARD_SCRIPT="$SCRIPT_DIR/scripts/data/process_data.py"
    else
        PROCESS_STANDARD_SCRIPT=$(find "$SCRIPT_DIR" -name "process_data.py" | head -1)
    fi
    
    log_info "Found process_data_polars.py: $PROCESS_POLARS_SCRIPT"
    log_info "Found process_data.py: $PROCESS_STANDARD_SCRIPT"
    
    # Use process_data_polars.py which is more memory efficient
    if [ -n "$PROCESS_POLARS_SCRIPT" ]; then
        log_info "Using process_data_polars.py for data processing with flags: $PROCESS_FLAGS"
        python3 "$PROCESS_POLARS_SCRIPT" $PROCESS_FLAGS
        
        if [ $? -eq 0 ]; then
            log_success "Data processing completed successfully with Polars"
        else
            log_error "Polars data processing failed. Trying fallback processor..."
            
            # Try fallback to the original processor if available
            if [ -n "$PROCESS_STANDARD_SCRIPT" ]; then
                log_info "Using process_data.py for data processing with flags: $PROCESS_FLAGS"
                python3 "$PROCESS_STANDARD_SCRIPT" $PROCESS_FLAGS
                
                if [ $? -eq 0 ]; then
                    log_success "Fallback data processing completed successfully"
                else
                    log_error "All data processing attempts failed"
                    return 1
                fi
            else
                log_error "No fallback processor available"
                return 1
            fi
        fi
    elif [ -n "$PROCESS_STANDARD_SCRIPT" ]; then
        # Fallback to standard process_data.py
        log_info "Using standard process_data.py script with flags: $PROCESS_FLAGS"
        python3 "$PROCESS_STANDARD_SCRIPT" $PROCESS_FLAGS
        
        if [ $? -eq 0 ]; then
            log_success "Standard data processing completed successfully"
        else
            log_error "Data processing failed"
            return 1
        fi
    else
        log_error "No data processing scripts found"
        return 1
    fi
    
    # Verify processed data
    if [ ! -f "$BASE_DIR/data/processed/crypto_train.parquet" ]; then
        log_error "Processed training data not found at $BASE_DIR/data/processed/crypto_train.parquet"
        return 1
    fi
    
    log_info "Data processing statistics:"
    log_info "$(du -h "$BASE_DIR/data/processed/crypto_train.parquet")"
    
    return 0
}