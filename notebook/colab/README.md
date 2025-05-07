# Google Colab Notebooks

This directory contains Jupyter notebooks specifically designed to run in Google Colab. These notebooks include Google Drive integration for persistent storage.

## Available Notebooks

- `numerai_pyspark_preprocessing_colab.ipynb`: Data preprocessing for Numerai using PySpark in Google Colab
- `numerai_sparkling_water_colab.ipynb`: H2O Sparkling Water integration with Apache Spark in Google Colab

## Using These Notebooks

To use these notebooks:

1. Upload the notebook to Google Colab (https://colab.research.google.com/)
2. Enable GPU acceleration if needed:
   - Go to Runtime > Change runtime type
   - Select "GPU" from the Hardware accelerator dropdown
   - Click "Save"
3. Run the first cell to mount your Google Drive:
   ```python
   from google.colab import drive
   drive.mount("/content/drive")
   ```
4. Follow the notebook instructions to install dependencies and process data

## Google Drive Integration

These notebooks use Google Drive for persistent storage of:

- Downloaded datasets
- Trained models
- Predictions and submissions

The default storage path is:
```
/content/drive/My Drive/Numerai_Crypto/
```

You can modify this path in the notebook if needed.

## GPU Acceleration

Both notebooks are configured to use GPU acceleration if available. The code will automatically detect and use the GPU. This is especially useful for:

- H2O XGBoost training
- Distributed machine learning with Sparkling Water
- Data processing with PySpark