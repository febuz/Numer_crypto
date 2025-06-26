# Temporal Feature Tracking System

This directory contains utilities for tracking features and their values over time using a data warehousing approach. The system tracks both which features exist and when their values change, storing temporal validity information to enable point-in-time analysis.

## Key Concepts

- **Temporal Validity**: Each value is stored with `valid_from` and `valid_to` dates, creating a complete history of changes.
- **Point-in-Time Analysis**: Query data as it existed at any specific point in time.
- **Delta Detection**: Only store new values when they actually change, saving storage space.
- **Data Lineage**: Track which files contributed to each value.

## Components

- **TemporalFeatureRegistry**: A class that manages a database of features and their values over time.
- **Enhanced Downloaders**: Modified versions of the data downloaders that incorporate temporal tracking.
- **Point-in-Time Tools**: Scripts for creating point-in-time datasets and time series.

## How It Works

1. The system maintains a SQLite database that tracks:
   - Data sources (Yiedl, Numerai)
   - Processed files
   - Features
   - Data values with temporal validity periods
   - Entity identifiers

2. When a new data file is processed:
   - The file is registered in the database
   - Its features are registered
   - For each entity and feature:
     - Calculate a hash of the value
     - Check if the value has changed since the last observation
     - If changed, close the previous value record and open a new one
     - Set `valid_from` date based on the file date
     - Previous values get a `valid_to` date, current values have `valid_to` as NULL

3. This approach enables:
   - Efficient storage (only store changes)
   - Point-in-time querying
   - Delta analysis (what changed between dates)
   - Time series construction without look-ahead bias

## Usage

### Processing Data with Temporal Tracking

```bash
# For Yiedl data
python utils/data/download_yiedl_temporal.py --output-dir /path/to/data/yiedl

# Show value history for a specific entity and feature
python utils/data/download_yiedl_temporal.py --entity BTC --feature price
```

### Creating Point-in-Time Datasets

```bash
# Create a single point-in-time snapshot
python scripts/create_point_in_time_dataset.py snapshot \
  --source yiedl \
  --entities BTC,ETH,SOL \
  --features price,volume,market_cap \
  --date 2025-01-15 \
  --output /path/to/output/snapshot.parquet

# Create a time series dataset with multiple snapshots
python scripts/create_point_in_time_dataset.py timeseries \
  --source yiedl \
  --entities BTC,ETH,SOL \
  --features price,volume,market_cap \
  --start-date 2025-01-01 \
  --end-date 2025-01-31 \
  --interval 1 \
  --output /path/to/output/timeseries.parquet
```

### Programmatic Usage

```python
from utils.feature.feature_registry_temporal import TemporalFeatureRegistry

# Initialize registry
registry = TemporalFeatureRegistry("/path/to/data")

# Process a file with temporal tracking
result = registry.process_file_with_tracking(
    'yiedl', 'my_data_file.parquet', 'symbol', 'date')

# Get value history for a specific entity and feature
history = registry.get_value_history('price', 'BTC', 'yiedl')
for record in history:
    print(f"{record['valid_from']} to {record['valid_to'] or 'Present'}: {record['value']}")

# Get a point-in-time snapshot for an entity
snapshot = registry.get_entity_snapshot('BTC', '2025-01-15', 'yiedl')
print(f"BTC price on 2025-01-15: {snapshot.get('price')}")

# Get a dataset of entities and features that changed between two dates
delta_df = registry.get_delta_dataset('2025-01-01', '2025-01-15', 'yiedl')
```

## Database Schema

The temporal feature registry uses a SQLite database with the following tables:

- **sources**: Data sources (Yiedl, Numerai)
- **files**: Processed files with metadata
- **features**: Feature definitions
- **feature_files**: Many-to-many relationship between features and files
- **data_values**: Actual values with temporal validity periods (`valid_from`, `valid_to`)

## Benefits

- **Efficient Storage**: Only store values when they change
- **Historical Analysis**: Ability to reconstruct datasets as they existed at any point in time
- **Change Detection**: Easily identify what changed and when
- **Backtesting**: Avoid look-ahead bias in historical analyses
- **Data Lineage**: Track which files contributed to each value

## Use Cases

1. **Backtesting Trading Strategies**:
   - Ensure you only use data that was available at the time of each trade
   - Avoid look-ahead bias in historical simulations

2. **Reproducing Historical Analyses**:
   - Generate the exact dataset as it existed on a specific date
   - Ensure consistent results when rerunning historical analyses

3. **Delta Processing**:
   - Only process data for entities and features that have changed
   - Optimize downstream pipeline to skip unchanged values

4. **Time Series Analysis**:
   - Create time series data with proper temporal alignment
   - Generate fixed-interval snapshots for consistent analysis