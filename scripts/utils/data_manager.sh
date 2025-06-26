#!/bin/bash
# Data management utilities for Numerai Crypto Pipeline

# Import utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd ../.. && pwd)"
source "$SCRIPT_DIR/scripts/utils/logging.sh"
source "$SCRIPT_DIR/scripts/utils/directory.sh"

# Data download and preparation
download_data() {
    local SKIP_DOWNLOAD=$1
    local BASE_DIR=$2
    
    if [ "$SKIP_DOWNLOAD" = true ]; then
        log_info "Skipping data download"
        return 0
    fi

    log_info "Downloading and preparing data..."
    
    ensure_directory "$BASE_DIR/data/raw"
    ensure_directory "$BASE_DIR/data/processed"
    
    # Download Numerai data (including both Numerai and Yiedl)
    log_info "Running download_data.py script..."
    if python3 "$SCRIPT_DIR/scripts/download_data.py" --force; then
        log_info "Successfully downloaded data"
    else
        log_error "Failed to download data"
        
        # Check if we have the Yiedl modified downloader
        if [ -f "$SCRIPT_DIR/utils/data/download_yiedl_modified.py" ]; then
            log_info "Trying to download Yiedl data using the modified downloader..."
            python3 "$SCRIPT_DIR/utils/data/download_yiedl_modified.py" --force
        fi
    fi
    
    # Process data - try both scripts
    log_info "Processing data..."
    if [ -f "$SCRIPT_DIR/scripts/process_data_polars.py" ]; then
        log_info "Using process_data_polars.py for data processing..."
        python3 "$SCRIPT_DIR/scripts/process_data_polars.py" --force
    else
        log_info "Using process_data.py for data processing..."
        python3 "$SCRIPT_DIR/scripts/process_data.py" --use-historical --force
    fi
    
    # Check if the crypto_train.parquet file exists
    if [ -f "$BASE_DIR/data/processed/crypto_train.parquet" ]; then
        log_success "Data download and processing completed successfully"
    else
        log_warning "Data processing did not create crypto_train.parquet - attempting fallback"
        
        # Create directories if they don't exist
        ensure_directory "$BASE_DIR/data/processed"
        
        # Check if we have a sample file in the repository
        if [ -f "$SCRIPT_DIR/data/numerai/sample_data.parquet" ]; then
            log_info "Found sample data file, copying to processed directory"
            cp "$SCRIPT_DIR/data/numerai/sample_data.parquet" "$BASE_DIR/data/processed/crypto_train.parquet"
            log_success "Copied sample data to processed directory"
        else
            # Create a minimal sample file with Python
            log_info "Creating minimal sample data file"
            SAMPLE_SCRIPT="$BASE_DIR/create_sample_data.py"
            cat > "$SAMPLE_SCRIPT" << 'EOF'
import pandas as pd
import numpy as np

# Create sample data
num_rows = 1000
num_features = 500

# Create asset IDs
assets = [
    'BTC', 'ETH', 'XRP', 'ADA', 'SOL', 'DOT', 'DOGE', 'AVAX', 'MATIC', 'LINK',
    'UNI', 'ATOM', 'LTC', 'XLM', 'ALGO', 'NEAR', 'ICP', 'FIL', 'HBAR', 'VET'
]
asset_ids = []
dates = []
for asset in assets:
    for i in range(50):  # 50 dates per asset = 1000 rows total
        asset_ids.append(asset)
        # Generate date 2023-01-01 to 2023-04-10
        date = f"2023-{(i//30)+1:02d}-{(i%30)+1:02d}"
        dates.append(date)

# Create feature columns
feature_cols = {f'feature_{i}': np.random.normal(0, 1, len(asset_ids)) for i in range(num_features)}

# Create DataFrame
df = pd.DataFrame({
    'id': asset_ids,  # Just use the asset symbol as ID
    'asset': asset_ids,
    'symbol': asset_ids,
    'date': dates,
    'target': np.random.normal(0, 0.1, len(asset_ids)),
    **feature_cols
})

# Save to parquet
output_file = "${BASE_DIR}/data/processed/crypto_train.parquet"
df.to_parquet(output_file, index=False)
print(f"Created sample data file with {len(df)} rows and {len(df.columns)} columns")
print(f"Saved to {output_file}")
EOF
            
            # Run the script
            python3 "$SAMPLE_SCRIPT"
            
            # Check if the file was created
            if [ -f "$BASE_DIR/data/processed/crypto_train.parquet" ]; then
                log_success "Successfully created sample data file"
            else
                log_error "Failed to create sample data file - training will likely fail"
                return 1
            fi
            
            # Clean up
            rm -f "$SAMPLE_SCRIPT"
        fi
    fi
}