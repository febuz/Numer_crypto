# Numer_crypto

Numerai/Numerai Crypto competition prediction models using LGBM, H2O XGBoost, Sparkling Water, and PySpark.

## Overview

This project provides tools and models for participating in the Numerai Crypto competition. It leverages various machine learning frameworks, including H2O's XGBoost, PySpark, and Sparkling Water for distributed training and inference.

## Features

- Data retrieval from the Numerai API
- Model training using H2O XGBoost
- Distributed processing with PySpark and H2O Sparkling Water
- Feature importance analysis
- Prediction generation and submission

## Project Structure

```
numer_crypto/
├── config/              # Configuration settings
├── data/                # Data retrieval and processing
├── models/              # Model implementations
├── scripts/             # Command-line scripts
└── utils/               # Utility functions
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/numer_crypto.git
   cd numer_crypto
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the package:
   ```bash
   pip install -e .
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your Numerai API credentials
   ```

## Prerequisites

- Python 3.8+
- Java 8+ (required for H2O and Spark)

## Usage

### Download Data

```bash
python -m numer_crypto.scripts.run --download --tournament crypto
```

### Train a Model

```bash
python -m numer_crypto.scripts.run --train --tournament crypto
```

### Generate Predictions

```bash
python -m numer_crypto.scripts.run --predict --tournament crypto
```

### Submit Predictions

```bash
python -m numer_crypto.scripts.run --predict --submit --tournament crypto
```

### Complete Pipeline

```bash
python -m numer_crypto.scripts.run --download --train --predict --submit --tournament crypto
```

## Configuration

Configure the application by editing the `.env` file:

```
# Numerai API credentials
NUMERAI_PUBLIC_ID=your_public_id_here
NUMERAI_SECRET_KEY=your_secret_key_here

# External storage configuration
EXTERNAL_STORAGE_PATH=/path/to/external/storage
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
- [Apache Spark](https://spark.apache.org/) for distributed computing framework