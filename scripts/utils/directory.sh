#!/bin/bash
# Directory management utilities for Numerai Crypto Pipeline

# Import logging utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd ../.. && pwd)"
source "$SCRIPT_DIR/scripts/utils/logging.sh"

# Create directory if it doesn't exist and handle permissions gracefully
ensure_directory() {
    local dir="$1"
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        # Try to set permissions but don't fail if it doesn't work
        chmod 777 "$dir" 2>/dev/null || true
        log_info "Created directory: $dir"
    elif [ ! -w "$dir" ]; then
        # Try to fix permissions but don't fail if it doesn't work
        chmod 777 "$dir" 2>/dev/null || true
        log_info "Directory exists but may not be writable: $dir"
    fi
}

# Ensure multiple directories exist
ensure_directories() {
    for dir in "$@"; do
        ensure_directory "$dir"
    done
}

# Check if directory is empty
is_directory_empty() {
    local dir="$1"
    if [ ! -d "$dir" ]; then
        # Directory doesn't exist, so consider it empty
        return 0
    elif [ -z "$(ls -A "$dir")" ]; then
        # Directory exists but is empty
        return 0
    else
        # Directory exists and has contents
        return 1
    fi
}

# Create standard directory structure for Numerai Crypto pipeline
create_standard_directories() {
    local BASE_DIR="$1"
    
    # Create base directories with proper permissions
    ensure_directory "$BASE_DIR"
    ensure_directory "$BASE_DIR/data"
    ensure_directory "$BASE_DIR/data/raw"
    ensure_directory "$BASE_DIR/data/processed"
    ensure_directory "$BASE_DIR/data/features"
    ensure_directory "$BASE_DIR/data/yiedl"
    ensure_directory "$BASE_DIR/data/numerai"
    ensure_directory "$BASE_DIR/models"
    ensure_directory "$BASE_DIR/metrics"
    ensure_directory "$BASE_DIR/predictions"
    ensure_directory "$BASE_DIR/submission"
    ensure_directory "$BASE_DIR/log"
    
    log_success "Created standard directory structure in $BASE_DIR"
}

# Fix permissions recursively for all data directories
fix_permissions() {
    local BASE_DIR="$1"
    
    log_info "Fixing permissions for all data directories..."
    chmod -R 777 "$BASE_DIR/data" 2>/dev/null || true
    chmod -R 777 "$BASE_DIR/models" 2>/dev/null || true
    chmod -R 777 "$BASE_DIR/metrics" 2>/dev/null || true
    chmod -R 777 "$BASE_DIR/predictions" 2>/dev/null || true
    chmod -R 777 "$BASE_DIR/submission" 2>/dev/null || true
    chmod -R 777 "$BASE_DIR/log" 2>/dev/null || true
    
    log_success "Permissions fixed for directories in $BASE_DIR"
}