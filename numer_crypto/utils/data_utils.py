"""
Data utility functions for the Numerai Crypto project.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numer_crypto.config.settings import DATA_DIR


def ensure_data_dir():
    """
    Ensure the data directory exists.
    
    Returns:
        str: Path to the data directory
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    return DATA_DIR


def save_dataframe(df, filename, format='parquet'):
    """
    Save a DataFrame to the data directory.
    
    Args:
        df (DataFrame): The DataFrame to save
        filename (str): The name of the file
        format (str): The format to save in ('parquet', 'csv', 'pickle')
        
    Returns:
        str: Path where the file was saved
    """
    data_dir = ensure_data_dir()
    file_path = os.path.join(data_dir, filename)
    
    if format == 'parquet':
        df.to_parquet(file_path)
    elif format == 'csv':
        df.to_csv(file_path, index=False)
    elif format == 'pickle':
        df.to_pickle(file_path)
    else:
        raise ValueError(f"Unsupported format: {format}")
        
    return file_path


def load_dataframe(filename, format='parquet'):
    """
    Load a DataFrame from the data directory.
    
    Args:
        filename (str): The name of the file
        format (str): The format to load from ('parquet', 'csv', 'pickle')
        
    Returns:
        DataFrame: The loaded DataFrame
    """
    file_path = os.path.join(DATA_DIR, filename)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    if format == 'parquet':
        return pd.read_parquet(file_path)
    elif format == 'csv':
        return pd.read_csv(file_path)
    elif format == 'pickle':
        return pd.read_pickle(file_path)
    else:
        raise ValueError(f"Unsupported format: {format}")


def convert_h2o_to_pandas(h2o_frame):
    """
    Convert an H2O frame to a pandas DataFrame.
    
    Args:
        h2o_frame: The H2O frame to convert
        
    Returns:
        DataFrame: The pandas DataFrame
    """
    return h2o_frame.as_data_frame()


def convert_pandas_to_h2o(pandas_df, h2o_instance):
    """
    Convert a pandas DataFrame to an H2O frame.
    
    Args:
        pandas_df (DataFrame): The pandas DataFrame to convert
        h2o_instance: The H2O instance
        
    Returns:
        H2OFrame: The H2O frame
    """
    return h2o_instance.H2OFrame(pandas_df)


def plot_feature_importance(model, features, n_features=20, figsize=(12, 8)):
    """
    Plot feature importance from a trained model.
    
    Args:
        model: Trained model with feature_importance_ attribute
        features (list): List of feature names
        n_features (int): Number of top features to show
        figsize (tuple): Size of the figure
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    else:
        raise AttributeError("Model doesn't have feature_importances_ attribute")
        
    # Sort feature importances
    indices = np.argsort(importance)[::-1][:n_features]
    
    # Plot
    plt.figure(figsize=figsize)
    plt.title("Feature importances")
    plt.bar(range(n_features), importance[indices], align="center")
    plt.xticks(range(n_features), [features[i] for i in indices], rotation=90)
    plt.tight_layout()
    
    return plt.gcf()