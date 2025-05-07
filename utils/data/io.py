#!/usr/bin/env python3
"""
Data I/O utilities for saving and loading dataframes in various formats.
This module handles different backend libraries (pandas, polars, cuDF)
and provides GPU acceleration when available.
"""
import os
import sys
import logging
import importlib.util

# Add parent directory to path if needed
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.log_utils import setup_logging
from config.settings import DATA_DIR, HARDWARE_CONFIG

# Set up logging
logger = setup_logging(name=__name__, level=logging.INFO)

# Check for polars availability
try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    logger.warning("Polars not available, falling back to pandas")
    import pandas as pd

# Check for RAPIDS availability for GPU acceleration
try:
    RAPIDS_AVAILABLE = importlib.util.find_spec('cudf') is not None
    if RAPIDS_AVAILABLE:
        import cudf
        logger.info("RAPIDS cuDF available for GPU acceleration")
    else:
        logger.debug("RAPIDS cuDF not available")
except ImportError:
    RAPIDS_AVAILABLE = False
    logger.debug("RAPIDS cuDF import error")

def ensure_data_dir():
    """
    Ensure the data directory exists.
    
    Returns:
        str: Path to the data directory
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    return DATA_DIR

def save_dataframe(df, filename, format='parquet', use_gpu=None):
    """
    Save a DataFrame to the data directory.
    
    Args:
        df (DataFrame): The DataFrame to save (Polars, pandas or cuDF DataFrame)
        filename (str): The name of the file
        format (str): The format to save in ('parquet', 'csv', 'pickle')
        use_gpu (bool): Whether to use GPU acceleration if available (None=auto-detect)
        
    Returns:
        str: Path where the file was saved
    """
    data_dir = ensure_data_dir()
    file_path = os.path.join(data_dir, filename)
    
    # Auto-detect GPU if not specified
    if use_gpu is None:
        use_gpu = RAPIDS_AVAILABLE and HARDWARE_CONFIG.get('gpu_count', 0) > 0
    
    # Use RAPIDS cuDF if requested and available
    if use_gpu and RAPIDS_AVAILABLE:
        try:
            # Handle cuDF DataFrame directly
            if hasattr(df, 'to_parquet') and hasattr(df, 'to_pandas') and 'cudf' in str(type(df)):
                if format == 'parquet':
                    df.to_parquet(file_path)
                elif format == 'csv':
                    df.to_csv(file_path, index=False)
                elif format == 'pickle':
                    import pickle
                    with open(file_path, 'wb') as f:
                        pickle.dump(df.to_pandas(), f)
                else:
                    raise ValueError(f"Unsupported format for cuDF: {format}")
                return file_path
            
            # Convert to cuDF if it's not already
            cudf_df = None
            if POLARS_AVAILABLE and isinstance(df, pl.DataFrame):
                # Convert Polars to pandas then to cuDF
                pandas_df = df.to_pandas()
                cudf_df = cudf.DataFrame.from_pandas(pandas_df)
            elif hasattr(df, 'to_pandas'):
                # Generic conversion via pandas
                pandas_df = df.to_pandas()
                cudf_df = cudf.DataFrame.from_pandas(pandas_df)
            else:
                # Try direct conversion (assuming it's a pandas DataFrame)
                cudf_df = cudf.DataFrame.from_pandas(df)
            
            # Save with cuDF
            if format == 'parquet':
                cudf_df.to_parquet(file_path)
            elif format == 'csv':
                cudf_df.to_csv(file_path, index=False)
            elif format == 'pickle':
                import pickle
                with open(file_path, 'wb') as f:
                    pickle.dump(cudf_df.to_pandas(), f)
            else:
                raise ValueError(f"Unsupported format for cuDF: {format}")
            
            return file_path
        except Exception as e:
            logger.warning(f"GPU acceleration failed, falling back to CPU: {e}")
            # Fall back to CPU methods
    
    # Use Polars if available
    if POLARS_AVAILABLE:
        if isinstance(df, pl.DataFrame):
            if format == 'parquet':
                df.write_parquet(file_path)
            elif format == 'csv':
                df.write_csv(file_path, include_header=True)
            elif format == 'pickle':
                # Polars doesn't have native pickle support, convert to pandas first
                import pickle
                with open(file_path, 'wb') as f:
                    pickle.dump(df.to_pandas(), f)
            else:
                raise ValueError(f"Unsupported format for Polars: {format}")
        else:
            # Convert to Polars first if possible
            try:
                if hasattr(df, 'to_pandas'):
                    # Convert to pandas then to Polars
                    pl_df = pl.from_pandas(df.to_pandas())
                else:
                    # Try direct conversion to Polars
                    pl_df = pl.from_pandas(df)
                    
                if format == 'parquet':
                    pl_df.write_parquet(file_path)
                elif format == 'csv':
                    pl_df.write_csv(file_path, include_header=True)
                elif format == 'pickle':
                    import pickle
                    with open(file_path, 'wb') as f:
                        pickle.dump(pl_df.to_pandas(), f)
                else:
                    raise ValueError(f"Unsupported format for Polars: {format}")
            except (TypeError, AttributeError) as e:
                raise ValueError(f"Unsupported DataFrame type for conversion: {type(df)}. Error: {e}")
    else:
        # Fall back to pandas
        if hasattr(df, 'to_pandas'):
            pandas_df = df.to_pandas()
        else:
            pandas_df = df
            
        if format == 'parquet':
            try:
                pandas_df.to_parquet(file_path)
            except Exception as e:
                logger.warning(f"Parquet save failed, falling back to CSV: {e}")
                pandas_df.to_csv(file_path, index=False)
        elif format == 'csv':
            pandas_df.to_csv(file_path, index=False)
        elif format == 'pickle':
            pandas_df.to_pickle(file_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
    return file_path

def load_dataframe(filename, format='parquet', use_gpu=None):
    """
    Load a DataFrame from the data directory.
    
    Args:
        filename (str): The name of the file
        format (str): The format to load from ('parquet', 'csv', 'pickle')
        use_gpu (bool): Whether to use GPU acceleration if available (None=auto-detect)
        
    Returns:
        DataFrame: The loaded DataFrame (Polars, pandas, or cuDF DataFrame)
    """
    file_path = os.path.join(DATA_DIR, filename)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Auto-detect GPU if not specified
    if use_gpu is None:
        use_gpu = RAPIDS_AVAILABLE and HARDWARE_CONFIG.get('gpu_count', 0) > 0
    
    # Use RAPIDS cuDF if requested and available
    if use_gpu and RAPIDS_AVAILABLE:
        try:
            logger.info(f"Loading {filename} with GPU acceleration (cuDF)")
            if format == 'parquet':
                return cudf.read_parquet(file_path)
            elif format == 'csv':
                return cudf.read_csv(file_path)
            elif format == 'pickle':
                import pickle
                with open(file_path, 'rb') as f:
                    pandas_df = pickle.load(f)
                return cudf.DataFrame.from_pandas(pandas_df)
            else:
                raise ValueError(f"Unsupported format for cuDF: {format}")
        except Exception as e:
            logger.warning(f"GPU acceleration failed, falling back to CPU: {e}")
            # Fall back to CPU methods
    
    # Use Polars if available
    if POLARS_AVAILABLE:
        logger.info(f"Loading {filename} with Polars")
        if format == 'parquet':
            try:
                return pl.read_parquet(file_path)
            except Exception as e:
                logger.warning(f"Polars parquet read failed, trying pandas: {e}")
                # Fall back to pandas
                import pandas as pd
                return pl.from_pandas(pd.read_parquet(file_path))
        elif format == 'csv':
            return pl.read_csv(file_path)
        elif format == 'pickle':
            # Polars doesn't have native pickle support
            import pickle
            with open(file_path, 'rb') as f:
                pandas_df = pickle.load(f)
            return pl.from_pandas(pandas_df)
        else:
            raise ValueError(f"Unsupported format for Polars: {format}")
    else:
        # Fall back to pandas
        logger.info(f"Loading {filename} with pandas")
        import pandas as pd
        if format == 'parquet':
            try:
                return pd.read_parquet(file_path)
            except Exception as e:
                logger.warning(f"Parquet read failed, trying CSV: {e}")
                return pd.read_csv(file_path)
        elif format == 'csv':
            return pd.read_csv(file_path)
        elif format == 'pickle':
            return pd.read_pickle(file_path)
        else:
            raise ValueError(f"Unsupported format for pandas: {format}")

if __name__ == "__main__":
    # Print information about available libraries
    print(f"Polars available: {POLARS_AVAILABLE}")
    print(f"RAPIDS cuDF available: {RAPIDS_AVAILABLE}")
    
    # Test with a small DataFrame
    if POLARS_AVAILABLE:
        print("Creating test DataFrame with Polars")
        df = pl.DataFrame({
            "A": [1, 2, 3, 4, 5],
            "B": ["a", "b", "c", "d", "e"]
        })
    else:
        print("Creating test DataFrame with pandas")
        import pandas as pd
        df = pd.DataFrame({
            "A": [1, 2, 3, 4, 5],
            "B": ["a", "b", "c", "d", "e"]
        })
    
    # Test save and load
    formats = ['parquet', 'csv', 'pickle']
    for fmt in formats:
        try:
            print(f"\nTesting {fmt} format:")
            file_path = save_dataframe(df, f"test.{fmt}", format=fmt)
            print(f"  Saved to {file_path}")
            
            loaded_df = load_dataframe(f"test.{fmt}", format=fmt)
            print(f"  Loaded successfully: {type(loaded_df)}")
            print(f"  Data shape: {loaded_df.shape if hasattr(loaded_df, 'shape') else 'unknown'}")
        except Exception as e:
            print(f"  Error with {fmt}: {e}")