#!/bin/bash
# Pipeline script for creating submission files for Numerai Crypto

# Import utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd ../.. && pwd)"
source "$SCRIPT_DIR/scripts/bash_utils/logging.sh"
source "$SCRIPT_DIR/scripts/bash_utils/directory.sh"
source "$SCRIPT_DIR/scripts/bash_utils/submission.sh"

# Function to create submission files in pipeline
create_submission_files_pipeline() {
    local BASE_DIR=$1
    
    log_info "Step 6: Creating final submission files..."
    
    # Create directory for submissions
    ensure_directory "$BASE_DIR/submission"
    
    # Use create_submission_files.sh if available
    if [ -f "$SCRIPT_DIR/scripts/create_submission_files.sh" ]; then
        bash "$SCRIPT_DIR/scripts/create_submission_files.sh"
        
        if [ $? -eq 0 ]; then
            log_success "Created submission files successfully"
        else
            log_error "Failed to create submission files"
            return 1
        fi
    else
        log_warning "create_submission_files.sh not found, creating submission files manually"
        
        # Use ensure_submission_files from submission.sh utils
        ensure_submission_files "$BASE_DIR"
    fi
    
    # Count submission files
    submission_count=$(find "$BASE_DIR/submission" -name "*.csv" | wc -l)
    log_info "Total submission files created: $submission_count"
    
    # List generated submission files
    echo ""
    log_info "Generated submission files:"
    find "$BASE_DIR/submission" -name "*.csv" -type f | while read -r file; do
        echo "  $(basename "$file")"
    done
    
    return 0
}

# If script is run directly (not sourced), execute the function with provided arguments
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Process command line arguments
    BASE_DIR=${1:-"/media/knight2/EDB/numer_crypto_temp"}
    
    create_submission_files_pipeline "$BASE_DIR"
fi