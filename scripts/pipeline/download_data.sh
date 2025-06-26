#!/bin/bash
# Pipeline script for downloading data for Numerai Crypto

# Import utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd ../.. && pwd)"
source "$SCRIPT_DIR/scripts/bash_utils/logging.sh"
source "$SCRIPT_DIR/scripts/bash_utils/directory.sh"
source "$SCRIPT_DIR/scripts/bash_utils/data_manager.sh"

# Function to download data in pipeline
download_data_pipeline() {
    local BASE_DIR=$1
    local SKIP_DOWNLOAD=$2
    local SKIP_YIEDL=$3
    local INCLUDE_HISTORICAL=${4:-true}

    log_info "Step 1: Downloading Numerai and Yiedl data..."
    
    # Use the download_data function from data_manager.sh, passing correct parameters
    download_data "$SKIP_DOWNLOAD" "$BASE_DIR" "$INCLUDE_HISTORICAL" "$SKIP_YIEDL"
    DOWNLOAD_RESULT=$?
    
    if [ $DOWNLOAD_RESULT -ne 0 ]; then
        log_error "Data download failed"
        return 1
    fi
    
    # Verify Numerai download - this is always required
    if [ ! -f "$BASE_DIR/data/raw/numerai_live.parquet" ] && [ ! -f "$BASE_DIR/data/raw/numerai_live.csv" ]; then
        log_error "Required Numerai data files not found after download"
        return 1
    fi
    
    # Verify Yiedl download only if not skipped
    if [ "$SKIP_YIEDL" = false ] && [ ! -f "$BASE_DIR/data/raw/yiedl_latest.parquet" ] && [ ! -f "$BASE_DIR/data/yiedl/yiedl_latest_"*.parquet ]; then
        log_warning "Yiedl data files not found, but were requested. Processing will continue with only Numerai data."
        log_warning "This may affect model quality. Consider rerunning with --fix-permissions flag."
    fi
    
    return 0
}

# If script is run directly (not sourced), execute the function with provided arguments
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Process command line arguments
    BASE_DIR=${1:-"/media/knight2/EDB/numer_crypto_temp"}
    SKIP_DOWNLOAD=${2:-false}
    SKIP_YIEDL=${3:-false}
    INCLUDE_HISTORICAL=${4:-true}
    
    download_data_pipeline "$BASE_DIR" "$SKIP_DOWNLOAD" "$SKIP_YIEDL" "$INCLUDE_HISTORICAL"
fi