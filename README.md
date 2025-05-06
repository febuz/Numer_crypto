# Numer_crypto

Numerai/Numerai Crypto competition prediction models using LightGBM, XGBoost, H2O, Sparkling Water, PySpark, and PyTorch.

## Overview

This project provides tools and models for participating in the Numerai Crypto competition. It leverages various machine learning frameworks, including LightGBM, XGBoost, H2O, PyTorch LSTM networks, and Sparkling Water for distributed training and inference.

## Features

- Data retrieval from the Numerai API
- Model training using multiple frameworks (LightGBM, XGBoost, H2O, PyTorch)
- GPU acceleration for improved performance
- High memory utilization (up to 600GB)
- Distributed processing with PySpark and H2O Sparkling Water
- Feature engineering and selection
- Ensemble model predictions
- 20-day ahead forecasting
- RMSE and hit rate evaluation metrics
- Prediction generation and submission

## New Yiedl-Based Crypto Pipeline

The project now includes a new pipeline specifically designed for using Yiedl data for Numerai Crypto competition:

- Processes real Yiedl crypto data (`yiedl_latest.parquet`)
- Maps Yiedl symbols to Numerai Crypto symbols
- Applies advanced feature engineering to prevent overfitting
- Creates stable ensemble models with RMSE < 0.2
- Generates compliant submission files for the competition

Run the full pipeline with:
```bash
./run_crypto_pipeline.py
```

For customization options:
```bash
./run_crypto_pipeline.py --skip-processing  # Skip data processing
./run_crypto_pipeline.py --skip-training    # Skip model training
```

## High Memory GPU-Accelerated Model

The project also includes a high-memory GPU-accelerated model specifically designed for crypto predictions with large datasets. Key features:

- Full utilization of hardware resources (memory and GPUs)
- Multiple model ensemble (LightGBM, XGBoost, PyTorch LSTM)
- Feature engineering optimized for crypto time series
- 20-day ahead forecasting
- Easy execution via shell script

Run with:
```bash
./run_high_mem_crypto_model.sh
```

For detailed documentation and parameters, see [HIGH_MEM_CRYPTO_MODEL.md](HIGH_MEM_CRYPTO_MODEL.md).

## Project Structure

```
numer_crypto/
├── config/              # Configuration settings
├── data/                # Data retrieval and processing
│   ├── yiedl/           # Yiedl crypto datasets
│   ├── processed/       # Processed feature data
│   └── submissions/     # Model predictions and submissions
├── models/              # Model implementations and saved models
├── scripts/             # Command-line scripts
│   ├── process_yiedl_data.py          # Yiedl data processor
│   ├── train_predict_crypto.py        # Model training and prediction 
│   ├── high_mem_crypto_model.py       # High memory GPU model
│   └── install_requirements.sh        # Dependencies installer
├── utils/               # Utility functions
├── run_crypto_pipeline.py             # New Yiedl-based pipeline
└── run_high_mem_crypto_model.sh       # Script for high memory model
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/numer_crypto.git
   cd numer_crypto
   ```

2. Run the installation script to create a virtual environment with all dependencies:
   ```bash
   bash scripts/install_requirements.sh
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your Numerai API credentials
   ```

## Prerequisites

- Python 3.8+
- Java 8+ (required for H2O and Spark)
- CUDA-compatible GPUs (for GPU acceleration)
- 16GB+ RAM (600GB+ recommended for high memory model)

## Usage

### Standard Models

#### Download Data

```bash
python -m numer_crypto.scripts.run --download --tournament crypto
```

#### Train a Model

```bash
python -m numer_crypto.scripts.run --train --tournament crypto
```

#### Generate Predictions

```bash
python -m numer_crypto.scripts.run --predict --tournament crypto
```

#### Submit Predictions

```bash
python -m numer_crypto.scripts.run --predict --submit --tournament crypto
```

#### Complete Pipeline

```bash
python -m numer_crypto.scripts.run --download --train --predict --submit --tournament crypto
```

### New Yiedl-Based Crypto Pipeline

```bash
# Run the full pipeline
./run_crypto_pipeline.py

# Skip the data processing step if already processed
./run_crypto_pipeline.py --skip-processing

# Only generate predictions using existing models
./run_crypto_pipeline.py --skip-processing --skip-training
```

### High Memory GPU Model

#### Run with Default Settings

```bash
./run_high_mem_crypto_model.sh
```

#### Run with Custom Settings

```bash
./run_high_mem_crypto_model.sh --ram 400 --forecast-days 20 --nn-model --evaluate
```

For all available options, see the [detailed documentation](HIGH_MEM_CRYPTO_MODEL.md).

## Configuration

Configure the application by editing the `.env` file:

```
# Numerai API credentials
NUMERAI_PUBLIC_ID=your_public_id_here
NUMERAI_SECRET_KEY=your_secret_key_here

# External storage configuration
EXTERNAL_STORAGE_PATH=/path/to/external/storage
```

## Advanced Usage with H2O Sparkling Water

For H2O Sparkling Water integration, ensure Java 11+ is installed and run:

```bash
./scripts/setup/setup_h2o_sparkling_java17.sh
python scripts/test_h2o_sparkling_java17.py
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Numerai](https://numer.ai/) for their cryptocurrency prediction competition
- [H2O.ai](https://www.h2o.ai/) for their machine learning framework
- [LightGBM](https://lightgbm.readthedocs.io/) and [XGBoost](https://xgboost.readthedocs.io/) for gradient boosting implementations
- [PyTorch](https://pytorch.org/) for neural network capabilities
- [Apache Spark](https://spark.apache.org/) for distributed computing framework