#!/bin/bash
# Pipeline script for processing data for Numerai Crypto

# Import utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd ../.. && pwd)"
source "$SCRIPT_DIR/scripts/bash_utils/logging.sh"
source "$SCRIPT_DIR/scripts/bash_utils/directory.sh"
source "$SCRIPT_DIR/scripts/bash_utils/data_manager.sh"

# Function to process data in pipeline
process_data_pipeline() {
    local BASE_DIR=$1
    local FORCE_REPROCESS=$2
    local USE_HISTORICAL=${3:-true}
    
    log_info "Step 2: Processing data..."
    
    # Use the shared process_data function from data_manager.sh
    process_data "$BASE_DIR" "$FORCE_REPROCESS" "$USE_HISTORICAL"
    PROCESS_RESULT=$?
    
    if [ $PROCESS_RESULT -ne 0 ]; then
        log_error "Data processing failed"
        return 1
    fi
    
    return 0
}

# If script is run directly (not sourced), execute the function with provided arguments
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Process command line arguments
    BASE_DIR=${1:-"/media/knight2/EDB/numer_crypto_temp"}
    FORCE_REPROCESS=${2:-false}
    USE_HISTORICAL=${3:-true}
    
    process_data_pipeline "$BASE_DIR" "$FORCE_REPROCESS" "$USE_HISTORICAL"
fi