# Notebook Directory

This directory contains Jupyter notebooks for the Numer_crypto project.

## Directory Structure

- `/notebook`: Main Jupyter notebooks directory (this directory)
  - `/notebook/colab`: Google Colab-specific notebooks (contain Google Drive integration)

## Regular Notebooks

- `cryptos_KISS.ipynb`: Keep It Simple, Stupid approach to cryptocurrency analysis
- `Data_retrieval_crypto_yiedl.ipynb`: Data retrieval from Yiedl cryptocurrency API
- `yiedl_crypto_data_for_numerai_example.ipynb`: Example of using Yiedl data with Numerai
- `yiedl_crypto_model.ipynb`: Machine learning model for Yiedl cryptocurrency data analysis

## Colab Notebooks

The `/colab` directory contains notebooks specifically designed to run in Google Colab:

- `numerai_pyspark_preprocessing_colab.ipynb`: Preprocessing for Numerai data using PySpark in Colab
- `numerai_sparkling_water_colab.ipynb`: H2O Sparkling Water integration with Apache Spark

## Running Colab Notebooks

When running Colab notebooks:

1. Upload the notebook to Google Colab
2. The notebook will automatically install the required dependencies
3. Follow the instructions to mount your Google Drive for saving models and data:

```python
from google.colab import drive
drive.mount("/content/drive")
```

4. Execute the cells in order to download data, train models, and make predictions

## GPU Acceleration

Many notebooks support GPU acceleration for faster training. In Colab, you can enable GPU by:

1. Go to Runtime > Change runtime type
2. Select "GPU" from the Hardware accelerator dropdown
3. Click "Save"

The notebook will automatically detect and use the GPU if available.