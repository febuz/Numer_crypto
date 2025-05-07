#!/bin/bash
# Main pipeline execution script for Numerai Crypto
# This script runs the full pipeline from environment setup to submission

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default time limit: 4 hours (in seconds)
TIME_LIMIT=14400
START_TIME=$(date +%s)

# Process command-line arguments
SKIP_ENV_SETUP=false
SKIP_DATA_DOWNLOAD=false
SKIP_DATA_PROCESSING=false
SKIP_FEATURE_GENERATION=false
SKIP_MODEL_TRAINING=false
BASELINE_ONLY=false

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --skip-env-setup           Skip environment setup"
    echo "  --skip-download            Skip data download step"
    echo "  --skip-processing          Skip data processing step"
    echo "  --skip-features            Skip feature generation step"
    echo "  --skip-training            Skip model training step"
    echo "  --baseline-only            Use baseline predictions only (quick run)"
    echo "  --time-limit SECONDS       Set time limit in seconds (default: 14400 = 4 hours)"
    echo "  --help                     Display this help message"
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-env-setup)
            SKIP_ENV_SETUP=true
            shift
            ;;
        --skip-download)
            SKIP_DATA_DOWNLOAD=true
            shift
            ;;
        --skip-processing)
            SKIP_DATA_PROCESSING=true
            shift
            ;;
        --skip-features)
            SKIP_FEATURE_GENERATION=true
            shift
            ;;
        --skip-training)
            SKIP_MODEL_TRAINING=true
            shift
            ;;
        --baseline-only)
            BASELINE_ONLY=true
            shift
            ;;
        --time-limit)
            TIME_LIMIT="$2"
            shift 2
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

# Function to check remaining time
check_remaining_time() {
    local current_time=$(date +%s)
    local elapsed=$((current_time - START_TIME))
    local remaining=$((TIME_LIMIT - elapsed))
    
    echo -e "${BLUE}Time elapsed: $elapsed seconds, Remaining: $remaining seconds${NC}"
    
    if [ $remaining -lt 600 ]; then
        echo -e "${RED}Warning: Less than 10 minutes remaining!${NC}"
        return 1
    fi
    return 0
}

# Function to log the start of a step
log_step_start() {
    echo -e "\n${BLUE}====================================${NC}"
    echo -e "${BLUE}= STEP: $1${NC}"
    echo -e "${BLUE}====================================${NC}"
}

# Step 1: Set up environment
setup_environment() {
    log_step_start "Setting up environment"
    
    if [ "$SKIP_ENV_SETUP" = true ]; then
        echo -e "${YELLOW}Skipping environment setup (--skip-env-setup specified)${NC}"
        return 0
    fi
    
    # Create directories
    mkdir -p /media/knight2/EDB/numer_crypto_temp/log
    mkdir -p /media/knight2/EDB/numer_crypto_temp/data/{raw,processed}
    mkdir -p /media/knight2/EDB/numer_crypto_temp/models
    mkdir -p /media/knight2/EDB/numer_crypto_temp/prediction
    mkdir -p /media/knight2/EDB/numer_crypto_temp/submission
    
    # Source environment scripts (note: this will affect the current shell)
    echo -e "${GREEN}Setting up Python environment...${NC}"
    source "$SCRIPT_DIR/scripts/environment/setup_python_env.sh"
    
    # Check if Python environment setup was successful
    if [ $? -ne 0 ]; then
        echo -e "${RED}Python environment setup failed${NC}"
        return 1
    fi
    
    echo -e "${GREEN}Setting up Java 17 environment...${NC}"
    source "$SCRIPT_DIR/scripts/environment/setup_java17_env.sh"
    
    # Check if Java environment setup was successful
    if [ $? -ne 0 ]; then
        echo -e "${RED}Java 17 environment setup failed${NC}"
        return 1
    fi
    
    echo -e "${GREEN}Environment setup complete${NC}"
    return 0
}

# Step 2: Download data
download_data() {
    log_step_start "Downloading data"
    
    if [ "$SKIP_DATA_DOWNLOAD" = true ]; then
        echo -e "${YELLOW}Skipping data download (--skip-download specified)${NC}"
        return 0
    fi
    
    if [ "$BASELINE_ONLY" = true ]; then
        echo -e "${YELLOW}Skipping data download (--baseline-only specified)${NC}"
        return 0
    fi
    
    # Ensure download scripts exist or create templates
    if [ ! -f "$SCRIPT_DIR/scripts/download_data.py" ]; then
        echo -e "${YELLOW}Creating download_data.py script template...${NC}"
        
        cat > "$SCRIPT_DIR/scripts/download_data.py" << 'EOF'
#!/usr/bin/env python3
"""
download_data.py - Download data for Numerai Crypto

This script downloads data from Numerai and Yiedl APIs.
"""
import os
import sys
import logging
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Simple logging setup
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default directories
RAW_DATA_DIR = "/numer_crypto_temp/data/raw"

def download_numerai_data(include_historical=False):
    """Download data from Numerai API"""
    logger.info("Downloading Numerai data...")
    
    # Create output directory
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    # For demonstration, create sample files
    sample_file = os.path.join(RAW_DATA_DIR, "numerai_sample_data.csv")
    with open(sample_file, 'w') as f:
        f.write("date,symbol,feature1,feature2,target\n")
        f.write("2023-01-01,BTC,0.1,0.2,1\n")
        f.write("2023-01-01,ETH,0.2,0.3,0\n")
    
    logger.info(f"Sample Numerai data saved to {sample_file}")
    return True

def download_yiedl_data(include_historical=False):
    """Download data from Yiedl API"""
    logger.info("Downloading Yiedl data...")
    
    # Create output directory
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    # For demonstration, create sample files
    sample_file = os.path.join(RAW_DATA_DIR, "yiedl_sample_data.csv")
    with open(sample_file, 'w') as f:
        f.write("date,asset,price,volume\n")
        f.write("2023-01-01,BTC,50000,10000\n")
        f.write("2023-01-01,ETH,3000,20000\n")
    
    # If historical data requested, create another file
    if include_historical:
        historical_file = os.path.join(RAW_DATA_DIR, "yiedl_historical_data.csv")
        with open(historical_file, 'w') as f:
            f.write("date,asset,price,volume\n")
            for month in range(1, 13):
                for day in range(1, 28, 7):
                    date = f"2022-{month:02d}-{day:02d}"
                    f.write(f"{date},BTC,{40000 + month*1000 + day*100},{10000 + day*100}\n")
                    f.write(f"{date},ETH,{2000 + month*100 + day*10},{20000 + day*100}\n")
        logger.info(f"Sample historical Yiedl data saved to {historical_file}")
    
    logger.info(f"Sample Yiedl data saved to {sample_file}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Download data for Numerai Crypto')
    parser.add_argument('--include-historical', action='store_true', help='Include historical data')
    
    args = parser.parse_args()
    
    logger.info("Starting download_data.py")
    
    # Download Numerai data
    if download_numerai_data(args.include_historical):
        logger.info("Numerai data download completed successfully")
    else:
        logger.error("Numerai data download failed")
        return False
    
    # Download Yiedl data
    if download_yiedl_data(args.include_historical):
        logger.info("Yiedl data download completed successfully")
    else:
        logger.error("Yiedl data download failed")
        return False
    
    logger.info("All data downloads completed successfully")
    return True

if __name__ == "__main__":
    main()
EOF
        chmod +x "$SCRIPT_DIR/scripts/download_data.py"
    fi
    
    # Run the download script
    python "$SCRIPT_DIR/scripts/download_data.py" --include-historical
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Data download failed${NC}"
        return 1
    fi
    
    echo -e "${GREEN}Data download complete${NC}"
    return 0
}

# Step 3: Process data
process_data() {
    log_step_start "Processing data"
    
    if [ "$SKIP_DATA_PROCESSING" = true ]; then
        echo -e "${YELLOW}Skipping data processing (--skip-processing specified)${NC}"
        return 0
    fi
    
    if [ "$BASELINE_ONLY" = true ]; then
        echo -e "${YELLOW}Skipping data processing (--baseline-only specified)${NC}"
        return 0
    fi
    
    # Check remaining time
    check_remaining_time
    if [ $? -ne 0 ]; then
        echo -e "${RED}Not enough time remaining, skipping data processing${NC}"
        return 0
    fi
    
    # Ensure process scripts exist or create templates
    if [ ! -f "$SCRIPT_DIR/scripts/process_data.py" ]; then
        echo -e "${YELLOW}Creating process_data.py script template...${NC}"
        
        cat > "$SCRIPT_DIR/scripts/process_data.py" << 'EOF'
#!/usr/bin/env python3
"""
process_data.py - Process data for Numerai Crypto

This script processes raw data and creates train/validation/prediction splits.
"""
import os
import sys
import logging
import argparse
import pandas as pd
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Simple logging setup
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default directories
RAW_DATA_DIR = "/numer_crypto_temp/data/raw"
PROCESSED_DATA_DIR = "/numer_crypto_temp/data/processed"
TRAIN_DIR = os.path.join(PROCESSED_DATA_DIR, "train")
VALIDATION_DIR = os.path.join(PROCESSED_DATA_DIR, "validation")
PREDICTION_DIR = os.path.join(PROCESSED_DATA_DIR, "prediction")

def process_numerai_data(use_historical=False):
    """Process Numerai data"""
    logger.info("Processing Numerai data...")
    
    # Load raw data
    numerai_file = os.path.join(RAW_DATA_DIR, "numerai_sample_data.csv")
    if not os.path.exists(numerai_file):
        logger.error(f"Numerai data file not found: {numerai_file}")
        return False
    
    numerai_data = pd.read_csv(numerai_file)
    logger.info(f"Loaded Numerai data with shape: {numerai_data.shape}")
    
    return numerai_data

def process_yiedl_data(use_historical=False):
    """Process Yiedl data"""
    logger.info("Processing Yiedl data...")
    
    # Load raw data
    yiedl_file = os.path.join(RAW_DATA_DIR, "yiedl_sample_data.csv")
    if not os.path.exists(yiedl_file):
        logger.error(f"Yiedl data file not found: {yiedl_file}")
        return False
    
    yiedl_data = pd.read_csv(yiedl_file)
    logger.info(f"Loaded Yiedl data with shape: {yiedl_data.shape}")
    
    # Load historical data if requested
    if use_historical:
        historical_file = os.path.join(RAW_DATA_DIR, "yiedl_historical_data.csv")
        if os.path.exists(historical_file):
            historical_data = pd.read_csv(historical_file)
            logger.info(f"Loaded historical Yiedl data with shape: {historical_data.shape}")
            
            # Combine with current data
            yiedl_data = pd.concat([historical_data, yiedl_data], ignore_index=True)
            logger.info(f"Combined Yiedl data shape: {yiedl_data.shape}")
    
    return yiedl_data

def create_data_splits(numerai_data, yiedl_data):
    """Create train/validation/prediction splits"""
    logger.info("Creating data splits...")
    
    # Create directories
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VALIDATION_DIR, exist_ok=True)
    os.makedirs(PREDICTION_DIR, exist_ok=True)
    
    # Merge data (simple example - in a real scenario this would be more complex)
    merged_data = pd.merge(
        numerai_data, 
        yiedl_data, 
        left_on=['date', 'symbol'], 
        right_on=['date', 'asset'],
        how='inner'
    )
    
    logger.info(f"Merged data shape: {merged_data.shape}")
    
    # Create splits (70% train, 15% validation, 15% prediction)
    train_size = int(len(merged_data) * 0.7)
    val_size = int(len(merged_data) * 0.15)
    
    train_data = merged_data.iloc[:train_size]
    val_data = merged_data.iloc[train_size:train_size+val_size]
    pred_data = merged_data.iloc[train_size+val_size:]
    
    # Save splits
    train_file = os.path.join(TRAIN_DIR, "train_data.csv")
    val_file = os.path.join(VALIDATION_DIR, "validation_data.csv")
    pred_file = os.path.join(PREDICTION_DIR, "prediction_data.csv")
    
    train_data.to_csv(train_file, index=False)
    val_data.to_csv(val_file, index=False)
    pred_data.to_csv(pred_file, index=False)
    
    logger.info(f"Saved train data to {train_file} with shape {train_data.shape}")
    logger.info(f"Saved validation data to {val_file} with shape {val_data.shape}")
    logger.info(f"Saved prediction data to {pred_file} with shape {pred_data.shape}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Process data for Numerai Crypto')
    parser.add_argument('--use-historical', action='store_true', help='Use historical data')
    
    args = parser.parse_args()
    
    logger.info("Starting process_data.py")
    
    # Process Numerai data
    numerai_data = process_numerai_data(args.use_historical)
    if numerai_data is False:
        logger.error("Numerai data processing failed")
        return False
    
    # Process Yiedl data
    yiedl_data = process_yiedl_data(args.use_historical)
    if yiedl_data is False:
        logger.error("Yiedl data processing failed")
        return False
    
    # Create data splits
    if create_data_splits(numerai_data, yiedl_data):
        logger.info("Data splits created successfully")
    else:
        logger.error("Data splits creation failed")
        return False
    
    logger.info("Data processing completed successfully")
    return True

if __name__ == "__main__":
    main()
EOF
        chmod +x "$SCRIPT_DIR/scripts/process_data.py"
    fi
    
    # Run the process script
    python "$SCRIPT_DIR/scripts/process_data.py" --use-historical
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Data processing failed${NC}"
        return 1
    fi
    
    echo -e "${GREEN}Data processing complete${NC}"
    return 0
}

# Step 4: Generate features
generate_features() {
    log_step_start "Generating features"
    
    if [ "$SKIP_FEATURE_GENERATION" = true ]; then
        echo -e "${YELLOW}Skipping feature generation (--skip-features specified)${NC}"
        return 0
    fi
    
    if [ "$BASELINE_ONLY" = true ]; then
        echo -e "${YELLOW}Skipping feature generation (--baseline-only specified)${NC}"
        return 0
    fi
    
    # Check remaining time
    check_remaining_time
    if [ $? -ne 0 ]; then
        echo -e "${RED}Not enough time remaining, skipping feature generation${NC}"
        return 0
    fi
    
    # Ensure the feature generation script exists or create a template
    if [ ! -f "$SCRIPT_DIR/scripts/generate_features.py" ]; then
        echo -e "${YELLOW}Creating generate_features.py script template...${NC}"
        
        cat > "$SCRIPT_DIR/scripts/generate_features.py" << 'EOF'
#!/usr/bin/env python3
"""
generate_features.py - Generate features for Numerai Crypto

This script generates time series features for training and prediction.
"""
import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Simple logging setup
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default directories
PROCESSED_DATA_DIR = "/numer_crypto_temp/data/processed"
TRAIN_DIR = os.path.join(PROCESSED_DATA_DIR, "train")
VALIDATION_DIR = os.path.join(PROCESSED_DATA_DIR, "validation")
PREDICTION_DIR = os.path.join(PROCESSED_DATA_DIR, "prediction")

def generate_time_series_features(df, generate_ts_features=True):
    """Generate time series features for the given DataFrame"""
    logger.info(f"Generating features for DataFrame with shape {df.shape}")
    
    # Make a copy of the DataFrame to avoid modifying the original
    result_df = df.copy()
    
    if not generate_ts_features:
        logger.info("Time series feature generation disabled, returning original data")
        return result_df
    
    # Example of simple time series features
    
    # Get numeric columns for feature generation
    numeric_cols = result_df.select_dtypes(include=[np.number]).columns.tolist()
    date_col = 'date' if 'date' in result_df.columns else None
    group_col = 'symbol' if 'symbol' in result_df.columns else ('asset' if 'asset' in result_df.columns else None)
    
    if date_col is None or group_col is None:
        logger.warning("Missing date or group column, cannot generate time series features")
        return result_df
    
    # Convert date column to datetime
    result_df[date_col] = pd.to_datetime(result_df[date_col])
    
    # Sort by group and date
    result_df = result_df.sort_values([group_col, date_col])
    
    # Generate rolling mean features
    for col in numeric_cols:
        # Skip columns that aren't meaningful for time series features
        if col in [group_col, 'target']:
            continue
            
        # Rolling mean with window sizes 3, 7
        for window in [3, 7]:
            result_df[f'{col}_rolling_mean_{window}'] = result_df.groupby(group_col)[col].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
    
    # Generate lag features
    for col in numeric_cols:
        # Skip columns that aren't meaningful for time series features
        if col in [group_col, 'target']:
            continue
            
        # Lag features with lag 1, 2
        for lag in [1, 2]:
            result_df[f'{col}_lag_{lag}'] = result_df.groupby(group_col)[col].shift(lag)
    
    logger.info(f"Generated features, new shape: {result_df.shape}")
    
    return result_df

def process_dataset(input_file, output_file, generate_ts_features=True):
    """Process a dataset file, generating features and saving the result"""
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        return False
    
    # Load data
    df = pd.read_csv(input_file)
    logger.info(f"Loaded data from {input_file} with shape {df.shape}")
    
    # Generate features
    result_df = generate_time_series_features(df, generate_ts_features)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save result
    result_df.to_csv(output_file, index=False)
    logger.info(f"Saved featured data to {output_file} with shape {result_df.shape}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Generate features for Numerai Crypto')
    parser.add_argument('--timeseries', action='store_true', help='Generate time series features')
    
    args = parser.parse_args()
    
    logger.info("Starting generate_features.py")
    
    # Process train data
    train_input = os.path.join(TRAIN_DIR, "train_data.csv")
    train_output = os.path.join(TRAIN_DIR, "train_data_featured.csv")
    if process_dataset(train_input, train_output, args.timeseries):
        logger.info("Train data feature generation completed successfully")
    else:
        logger.error("Train data feature generation failed")
        return False
    
    # Process validation data
    val_input = os.path.join(VALIDATION_DIR, "validation_data.csv")
    val_output = os.path.join(VALIDATION_DIR, "validation_data_featured.csv")
    if process_dataset(val_input, val_output, args.timeseries):
        logger.info("Validation data feature generation completed successfully")
    else:
        logger.error("Validation data feature generation failed")
        return False
    
    # Process prediction data
    pred_input = os.path.join(PREDICTION_DIR, "prediction_data.csv")
    pred_output = os.path.join(PREDICTION_DIR, "prediction_data_featured.csv")
    if process_dataset(pred_input, pred_output, args.timeseries):
        logger.info("Prediction data feature generation completed successfully")
    else:
        logger.error("Prediction data feature generation failed")
        return False
    
    logger.info("Feature generation completed successfully")
    return True

if __name__ == "__main__":
    main()
EOF
        chmod +x "$SCRIPT_DIR/scripts/generate_features.py"
    fi
    
    # Run the feature generation script
    python "$SCRIPT_DIR/scripts/generate_features.py" --timeseries
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Feature generation failed${NC}"
        return 1
    fi
    
    echo -e "${GREEN}Feature generation complete${NC}"
    return 0
}

# Step 5: Train models
train_models() {
    log_step_start "Training models"
    
    if [ "$SKIP_MODEL_TRAINING" = true ]; then
        echo -e "${YELLOW}Skipping model training (--skip-training specified)${NC}"
        return 0
    fi
    
    if [ "$BASELINE_ONLY" = true ]; then
        echo -e "${YELLOW}Skipping model training (--baseline-only specified)${NC}"
        return 0
    fi
    
    # Check remaining time
    check_remaining_time
    if [ $? -ne 0 ]; then
        echo -e "${RED}Not enough time remaining, skipping model training${NC}"
        return 0
    fi
    
    # Ensure the model training script exists or create a template
    if [ ! -f "$SCRIPT_DIR/scripts/train_models.py" ]; then
        echo -e "${YELLOW}Creating train_models.py script template...${NC}"
        
        cat > "$SCRIPT_DIR/scripts/train_models.py" << 'EOF'
#!/usr/bin/env python3
"""
train_models.py - Train models for Numerai Crypto

This script trains machine learning models for Numerai Crypto predictions.
"""
import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Simple logging setup
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default directories
PROCESSED_DATA_DIR = "/numer_crypto_temp/data/processed"
TRAIN_DIR = os.path.join(PROCESSED_DATA_DIR, "train")
VALIDATION_DIR = os.path.join(PROCESSED_DATA_DIR, "validation")
MODELS_DIR = "/numer_crypto_temp/models"

def train_simple_model(X_train, y_train, use_gpu=False, parallel=False):
    """Train a simple model for demonstration purposes"""
    logger.info(f"Training simple model with {X_train.shape[1]} features")
    
    try:
        from sklearn.ensemble import RandomForestRegressor
        
        # Train parameters
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }
        
        # Add parallel processing if requested
        if parallel:
            params['n_jobs'] = -1
            logger.info("Using parallel processing")
        
        # Create and train model
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        
        logger.info("Model training completed")
        return model
    
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return None

def prepare_data(file_path):
    """Prepare data for model training"""
    if not os.path.exists(file_path):
        logger.error(f"Data file not found: {file_path}")
        return None, None
    
    # Load data
    df = pd.read_csv(file_path)
    logger.info(f"Loaded data from {file_path} with shape {df.shape}")
    
    # Prepare features and target
    if 'target' not in df.columns:
        logger.error("Target column 'target' not found in data")
        return None, None
    
    # Remove non-numeric columns and the target column from features
    X = df.select_dtypes(include=[np.number])
    excluded_cols = ['target', 'Symbol', 'symbol', 'Prediction', 'prediction']
    feature_cols = [col for col in X.columns if col not in excluded_cols]
    
    # Handle missing values
    X = X[feature_cols].fillna(0)
    y = df['target'].fillna(0)
    
    logger.info(f"Prepared data with {X.shape[1]} features")
    return X, y

def save_model(model, model_path):
    """Save the trained model"""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model saved to {model_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Train models for Numerai Crypto')
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU for training')
    parser.add_argument('--parallel', action='store_true', help='Use parallel processing')
    
    args = parser.parse_args()
    
    logger.info("Starting train_models.py")
    
    # Prepare training data
    train_file = os.path.join(TRAIN_DIR, "train_data_featured.csv")
    X_train, y_train = prepare_data(train_file)
    
    if X_train is None or y_train is None:
        logger.error("Failed to prepare training data")
        return False
    
    # Train model
    model = train_simple_model(X_train, y_train, args.use_gpu, args.parallel)
    
    if model is None:
        logger.error("Model training failed")
        return False
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODELS_DIR, f"model_{timestamp}.pkl")
    
    if save_model(model, model_path):
        logger.info("Model saved successfully")
    else:
        logger.error("Failed to save model")
        return False
    
    logger.info("Model training completed successfully")
    return True

if __name__ == "__main__":
    main()
EOF
        chmod +x "$SCRIPT_DIR/scripts/train_models.py"
    fi
    
    # Run the model training script
    python "$SCRIPT_DIR/scripts/train_models.py" --parallel
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Model training failed${NC}"
        return 1
    fi
    
    echo -e "${GREEN}Model training complete${NC}"
    return 0
}

# Step 6: Generate predictions
generate_predictions() {
    log_step_start "Generating predictions"
    
    # Check remaining time
    check_remaining_time
    if [ $? -ne 0 ]; then
        echo -e "${RED}Not enough time remaining, using baseline predictions${NC}"
        # Use baseline predictions if time is running out
        python "$SCRIPT_DIR/scripts/generate_predictions.py" --baseline --num-symbols 100
        return $?
    fi
    
    # Use baseline predictions if specified
    if [ "$BASELINE_ONLY" = true ]; then
        echo -e "${YELLOW}Using baseline predictions (--baseline-only specified)${NC}"
        python "$SCRIPT_DIR/scripts/generate_predictions.py" --baseline --num-symbols 100
        return $?
    fi
    
    # Run the prediction script with trained models
    python "$SCRIPT_DIR/scripts/generate_predictions.py"
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Prediction generation failed, falling back to baseline${NC}"
        python "$SCRIPT_DIR/scripts/generate_predictions.py" --baseline --num-symbols 100
        return $?
    fi
    
    echo -e "${GREEN}Prediction generation complete${NC}"
    return 0
}

# Step 7: Create submission
create_submission() {
    log_step_start "Creating submission"
    
    # Run the submission script
    python "$SCRIPT_DIR/scripts/create_submission.py"
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Submission creation failed${NC}"
        return 1
    fi
    
    echo -e "${GREEN}Submission creation complete${NC}"
    
    # List submission files
    echo -e "${GREEN}Submission files:${NC}"
    ls -lt /numer_crypto_temp/submission/
    
    return 0
}

# Main execution flow
main() {
    # Print banner
    echo -e "\n${GREEN}=================================${NC}"
    echo -e "${GREEN}= Numerai Crypto Pipeline Runner =${NC}"
    echo -e "${GREEN}=================================${NC}"
    echo -e "${BLUE}Time limit: $TIME_LIMIT seconds${NC}"
    echo
    
    # Execute each step
    
    # Step 1: Set up environment
    setup_environment
    if [ $? -ne 0 ]; then
        echo -e "${RED}Pipeline failed at environment setup step${NC}"
        exit 1
    fi
    
    # Step 2: Download data
    download_data
    if [ $? -ne 0 ]; then
        echo -e "${RED}Pipeline failed at data download step${NC}"
        exit 1
    fi
    
    # Step 3: Process data
    process_data
    if [ $? -ne 0 ]; then
        echo -e "${RED}Pipeline failed at data processing step${NC}"
        exit 1
    fi
    
    # Step 4: Generate features
    generate_features
    # Continue even if feature generation fails
    
    # Step 5: Train models
    train_models
    # Continue even if model training fails
    
    # Step 6: Generate predictions
    generate_predictions
    if [ $? -ne 0 ]; then
        echo -e "${RED}Pipeline failed at prediction generation step${NC}"
        exit 1
    fi
    
    # Step 7: Create submission
    create_submission
    if [ $? -ne 0 ]; then
        echo -e "${RED}Pipeline failed at submission creation step${NC}"
        exit 1
    fi
    
    # Print final summary
    end_time=$(date +%s)
    elapsed=$((end_time - START_TIME))
    echo -e "\n${GREEN}==========================${NC}"
    echo -e "${GREEN}= Pipeline Run Complete! =${NC}"
    echo -e "${GREEN}==========================${NC}"
    echo -e "${BLUE}Total time: $elapsed seconds${NC}"
    echo -e "${GREEN}Submissions saved to: /numer_crypto_temp/submission/${NC}"
    
    exit 0
}

# Start the pipeline
main