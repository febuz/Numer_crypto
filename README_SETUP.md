# Numerai Crypto Pipeline Setup Guide

This guide explains how to set up and run the Numerai Crypto pipeline with the optimized environment.

## Environment Setup

### 1. Create a Python Virtual Environment

```bash
# Create a virtual environment
python3 -m venv numer_crypto_venv

# Activate the environment
source numer_crypto_venv/bin/activate
```

### 2. Install Required Packages

```bash
# Install core data processing packages
pip install requests polars pyarrow fastparquet pandas numpy scipy

# Install ML packages
pip install scikit-learn xgboost lightgbm psutil numerapi

# Install GPU acceleration packages (if needed)
pip install torch torchvision cupy-cuda12x catboost nvidia-ml-py
```

## Running the Pipeline

### Basic Pipeline

Run the basic pipeline for data downloading and processing:

```bash
bash pipeline_basic.sh
```

This will:
1. Download Numerai data
2. Process the data with Polars
3. Generate features with the high memory limit (500GB)

### Full Pipeline with Submission

To run the full pipeline with submission to Numerai:

```bash
bash pipeline_basic.sh --submit --model-id "YOUR_MODEL_ID" --public-id "YOUR_API_PUBLIC_ID" --secret-key "YOUR_API_SECRET_KEY"
```

### Submission Only

If you want to submit predictions without running the full pipeline:

```bash
python simple_crypto_submit.py --model-id "YOUR_MODEL_ID" --public-id "YOUR_API_PUBLIC_ID" --secret-key "YOUR_API_SECRET_KEY"
```

## Environment Variables

You can set these environment variables to avoid passing API credentials in command line:

```bash
export NUMERAI_PUBLIC_ID="YOUR_API_PUBLIC_ID"
export NUMERAI_SECRET_KEY="YOUR_API_SECRET_KEY"
```

## Troubleshooting

### GPU Memory Issues

If you encounter GPU memory errors, you can adjust the memory limit in the pipeline scripts:

```bash
python scripts/run_fast_iterative_evolution.py --memory-limit-gb 500
```

### API Authentication Issues

If you encounter authentication issues with the Numerai API, ensure:

1. Your API credentials are correct
2. You have a valid model ID for submissions
3. Your prediction file follows the required format (id, prediction columns)

## Production Use

For production use, consider:

1. Setting up a cron job to run the pipeline regularly
2. Monitoring GPU memory usage
3. Using a dedicated server with sufficient resources
4. Implementing error reporting

## Directory Structure

### Repository Structure

The repository contains only source code, scripts, and documentation:

- `config/`: Configuration settings
- `data/`: Data management code (not actual data)
- `scripts/`: Pipeline and utility scripts
  - `features/`: Feature engineering and selection code
  - `models/`: ML model implementations
  - `utils/`: Utility scripts (including scripts formerly in root directory)
- `tests/`: Test scripts and validation tools
- `utils/`: Core utility functions (GPU, Memory, Threading, etc.)
- `pipeline.sh`: Main pipeline script
- `go_pipeline.sh`: Advanced pipeline script

### External Data Directory Structure

All generated data, models, and outputs are stored outside the repository in `/media/knight2/EDB/numer_crypto_temp/` with the following structure:

```
/media/knight2/EDB/numer_crypto_temp/
├── data/               # Data files
│   ├── numerai/        # Numerai competition data
│   ├── yiedl/          # Yiedl data files
│   ├── raw/            # Raw downloaded data
│   └── processed/      # Processed data files
├── models/             # Trained model files
│   └── checkpoints/    # Model checkpoints
├── metrics/            # Performance metrics and analytics
├── predictions/        # Generated predictions
├── submission/         # Formatted submission files
├── log/                # Log files
├── feature_importance/ # Feature importance analysis
└── venv/               # Virtual environments
```

This separation has several benefits:
1. Keeps the repository clean and focused on source code
2. Prevents accidental commits of large files to git
3. Allows for easier backup of just the code vs. generated data
4. Supports multiple data versions without cluttering the repo
5. Makes it easier to share just the code without sharing data

The pipeline scripts automatically create these directories as needed, but you can also create them manually:

```bash
mkdir -p /media/knight2/EDB/numer_crypto_temp/{data,models,metrics,predictions,submission,log,feature_importance}
mkdir -p /media/knight2/EDB/numer_crypto_temp/data/{raw,processed,features}
mkdir -p /media/knight2/EDB/numer_crypto_temp/models/checkpoints
```