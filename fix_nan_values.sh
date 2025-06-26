#!/bin/bash
# Script to fix NaN values by replacing with column means instead of zeros

echo "===== NaN Value Replacement Fix ====="
echo "This script will replace NaN values with column means instead of zeros."
echo "This improves statistical properties and prevents code from hanging during processing."
echo ""

# Check if input file was provided
if [ "$1" != "" ]; then
  INPUT_FILE="$1"
  echo "Using provided input file: $INPUT_FILE"
else
  # Look for existing data files
  if [ -f "/media/knight2/EDB/numer_crypto_temp/data/processed/crypto_train.parquet" ]; then
    INPUT_FILE="/media/knight2/EDB/numer_crypto_temp/data/processed/crypto_train.parquet"
    echo "Using existing processed data: $INPUT_FILE"
  elif [ -f "/media/knight2/EDB/numer_crypto_temp/data/raw/numerai_latest.parquet" ]; then
    INPUT_FILE="/media/knight2/EDB/numer_crypto_temp/data/raw/numerai_latest.parquet"
    echo "Using raw data: $INPUT_FILE"
  else
    echo "Error: No input file provided and no default files found."
    echo "Usage: ./fix_nan_values.sh [input_file]"
    exit 1
  fi
fi

# Run the NaN fixing script
echo ""
echo "Running NaN replacement with column means..."
python -m scripts.data.process_data_nan_fix --input "$INPUT_FILE"

# Check if the script succeeded
if [ $? -eq 0 ]; then
  echo ""
  echo "✅ NaN values have been successfully replaced with column means."
  echo "You can now proceed with feature generation and model training."
  echo ""
  echo "To run the full pipeline with improved NaN handling:"
  echo "python -m scripts.run_with_nan_means"
else
  echo ""
  echo "❌ Error: NaN fixing failed. Please check the logs for details."
  exit 1
fi