# Feature Tracking System

This directory contains utilities for tracking features that have been processed from Yiedl and Numerai datasets. The system ensures that only new features are processed when new data files are received, avoiding redundant processing and optimizing performance.

## Components

- **FeatureRegistry**: A class that manages a database of features seen in processed data files.
- **Enhanced Downloaders**: Modified versions of the Yiedl and Numerai data downloaders that incorporate feature tracking.
- **Initialization Script**: A script to initialize the feature registry with existing data.

## How It Works

1. The system maintains a SQLite database that tracks:
   - Data sources (Yiedl, Numerai)
   - Processed files (with hashes to identify them)
   - Features seen in each file (with metadata)
   - Relationships between files and features

2. When a new data file is downloaded:
   - The file is registered in the database
   - Its features are compared against previously seen features
   - Only new features (plus essential columns) are returned for processing
   - The filtered data can be saved to a separate file

3. This approach significantly reduces processing time and memory usage when:
   - The same features appear in multiple files over time
   - Only a small percentage of features are new in each update

## Usage

### Initialization

Before using the system for the first time, initialize the registry with existing data:

```bash
python scripts/initialize_feature_registry.py --data-dir /path/to/data
```

### Downloading Data with Feature Tracking

Instead of using the standard downloaders, use the tracked versions:

```bash
# For Yiedl data
python utils/data/download_yiedl_tracked.py --output-dir /path/to/data/yiedl

# For Numerai data
python utils/data/download_numerai_tracked.py --output-dir /path/to/data/numerai
```

### Programmatic Usage

```python
from utils.feature.feature_registry import FeatureRegistry

# Initialize registry
registry = FeatureRegistry("/path/to/data")

# Register a file and get new features
df = pl.read_parquet("my_data_file.parquet")
new_df, common_cols = registry.get_new_features("yiedl", "my_data_file.parquet", df)

# Process only new features
# ...

# Get feature statistics
counts = registry.get_feature_count()
print(f"Total features by source: {counts}")
```

## Database Schema

The feature registry uses a SQLite database with the following tables:

- **sources**: Data sources (Yiedl, Numerai)
- **files**: Processed files with metadata
- **features**: Individual features with tracking information
- **feature_files**: Many-to-many relationship between features and files

## Benefits

- **Performance**: Reduces processing time by focusing only on new features
- **Memory Efficiency**: Prevents memory bloat from repeatedly processing the same features
- **Tracking**: Provides visibility into feature evolution over time
- **Incremental Processing**: Enables incremental updates to the feature set

## Integration with Existing Pipeline

The feature tracking system is designed to integrate seamlessly with the existing Numerai Crypto pipeline. It operates transparently in the download phase, ensuring that only new features flow through to the processing and model training phases.