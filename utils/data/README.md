# Numerai Crypto Data Utilities

This directory contains utility functions for downloading, loading, and processing data for the Numerai Crypto tournament.

## Module Overview

- **`download_numerai.py`**: Functions for downloading data from the Numerai API
- **`download_yiedl.py`**: Functions for downloading data from the Yiedl API
- **`load_numerai.py`**: Functions for loading and processing Numerai data
- **`load_yiedl.py`**: Functions for loading and processing Yiedl data
- **`create_merged_dataset.py`**: Functions for creating merged datasets from Numerai and Yiedl data
- **`report_merge_summary.py`**: Functions for generating reports about merged datasets

## Function Documentation

### download_numerai.py

#### `download_numerai_crypto_data(numerai_dir, api_key=None, api_secret=None)`

Downloads Numerai crypto tournament data using the numerapi package.

- **Parameters:**
  - `numerai_dir`: Directory to save downloaded data
  - `api_key`: Optional Numerai API key
  - `api_secret`: Optional Numerai API secret

- **Returns:**
  - Dictionary with paths to downloaded files:
    - `train_targets`: Path to training targets file
    - `live_universe`: Path to live universe file (contains eligible symbols for submission)
    - `train_data`: Path to training data file
    - `current_round`: Current tournament round number

### download_yiedl.py

#### `download_yiedl_data(yiedl_dir)`

Downloads Yiedl data, which provides additional features for cryptocurrency prediction.

- **Parameters:**
  - `yiedl_dir`: Directory to save downloaded data

- **Returns:**
  - Dictionary with paths to downloaded files:
    - `latest`: Path to latest data file
    - `historical`: Path to historical data file

### load_numerai.py

#### `load_numerai_data(numerai_files)`

Loads Numerai data from downloaded files.

- **Parameters:**
  - `numerai_files`: Dictionary with paths to Numerai files

- **Returns:**
  - Dictionary with loaded data:
    - `train_data`: Training data DataFrame
    - `live_universe`: Live universe DataFrame
    - `current_round`: Current tournament round number

#### `get_eligible_crypto_symbols(numerai_data)`

Extracts eligible cryptocurrency symbols from Numerai data.

- **Parameters:**
  - `numerai_data`: Dictionary with loaded Numerai data

- **Returns:**
  - List of eligible cryptocurrency symbols

### load_yiedl.py

#### `load_yiedl_data(yiedl_files)`

Loads Yiedl data from downloaded files.

- **Parameters:**
  - `yiedl_files`: Dictionary with paths to Yiedl files

- **Returns:**
  - Dictionary with loaded data:
    - `latest`: Latest data DataFrame
    - `historical`: Historical data DataFrame

#### `load_yiedl_data_with_polars(yiedl_files)`

Loads Yiedl data using the polars package for improved performance.
Falls back to pandas if polars is not available.

- **Parameters:**
  - `yiedl_files`: Dictionary with paths to Yiedl files

- **Returns:**
  - Dictionary with loaded data (same as `load_yiedl_data`)

#### `get_yiedl_crypto_symbols(yiedl_data)`

Extracts cryptocurrency symbols from Yiedl data.

- **Parameters:**
  - `yiedl_data`: Dictionary with loaded Yiedl data

- **Returns:**
  - List of cryptocurrency symbols in Yiedl data

### create_merged_dataset.py

#### `get_overlapping_symbols(numerai_data, yiedl_data)`

Finds cryptocurrency symbols that appear in both Numerai and Yiedl data.

- **Parameters:**
  - `numerai_data`: Dictionary with loaded Numerai data
  - `yiedl_data`: Dictionary with loaded Yiedl data

- **Returns:**
  - List of overlapping cryptocurrency symbols

#### `create_merged_dataset(numerai_data, yiedl_data)`

Creates a merged dataset by combining Numerai and Yiedl data.

- **Parameters:**
  - `numerai_data`: Dictionary with loaded Numerai data
  - `yiedl_data`: Dictionary with loaded Yiedl data

- **Returns:**
  - DataFrame containing the merged dataset

### report_merge_summary.py

#### `report_merge_summary(numerai_data, yiedl_data, merged_data, overlapping_symbols)`

Generates a summary report about the merged dataset.

- **Parameters:**
  - `numerai_data`: Dictionary with loaded Numerai data
  - `yiedl_data`: Dictionary with loaded Yiedl data
  - `merged_data`: DataFrame containing the merged dataset
  - `overlapping_symbols`: List of overlapping cryptocurrency symbols

- **Returns:**
  - Dictionary with summary information

#### `save_merge_report(report, file_path)`

Saves a merge report to a JSON file.

- **Parameters:**
  - `report`: Dictionary with summary information
  - `file_path`: Path to save the report

## Usage Example

```python
# Download data
from utils.data.download_numerai import download_numerai_crypto_data
from utils.data.download_yiedl import download_yiedl_data

numerai_files = download_numerai_crypto_data('data/numerai')
yiedl_files = download_yiedl_data('data/yiedl')

# Load data
from utils.data.load_numerai import load_numerai_data
from utils.data.load_yiedl import load_yiedl_data

numerai_data = load_numerai_data(numerai_files)
yiedl_data = load_yiedl_data(yiedl_files)

# Get overlapping symbols
from utils.data.create_merged_dataset import get_overlapping_symbols

overlapping_symbols = get_overlapping_symbols(numerai_data, yiedl_data)
print(f"Found {len(overlapping_symbols)} overlapping symbols: {', '.join(overlapping_symbols[:5])}...")

# Create merged dataset
from utils.data.create_merged_dataset import create_merged_dataset

merged_data = create_merged_dataset(numerai_data, yiedl_data)
print(f"Merged dataset shape: {merged_data.shape}")

# Generate report
from utils.data.report_merge_summary import report_merge_summary, save_merge_report

report = report_merge_summary(numerai_data, yiedl_data, merged_data, overlapping_symbols)
save_merge_report(report, 'reports/merge_report.json')
```