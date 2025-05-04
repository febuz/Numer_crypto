"""
Data utility functions for the Numerai Crypto project.
"""
import os
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from numer_crypto.config.settings import DATA_DIR, SPARK_CONFIG, HARDWARE_CONFIG

# Check for RAPIDS availability for GPU acceleration
try:
    import importlib.util
    RAPIDS_AVAILABLE = importlib.util.find_spec('cudf') is not None
    if RAPIDS_AVAILABLE:
        import cudf
except ImportError:
    RAPIDS_AVAILABLE = False


def ensure_data_dir():
    """
    Ensure the data directory exists.
    
    Returns:
        str: Path to the data directory
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    return DATA_DIR


def save_dataframe(df, filename, format='parquet', use_spark=False, use_gpu=None):
    """
    Save a DataFrame to the data directory.
    
    Args:
        df (DataFrame): The DataFrame to save (Polars, Spark or cuDF DataFrame)
        filename (str): The name of the file
        format (str): The format to save in ('parquet', 'csv', 'pickle')
        use_spark (bool): Whether to use Spark for saving the DataFrame
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
    if use_gpu and RAPIDS_AVAILABLE and not use_spark:
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
            if isinstance(df, pl.DataFrame):
                # Convert Polars to pandas then to cuDF
                pandas_df = df.to_pandas()
                cudf_df = cudf.DataFrame.from_pandas(pandas_df)
            elif hasattr(df, 'toPandas'):
                # Convert Spark DataFrame to pandas then to cuDF
                pandas_df = df.toPandas()
                cudf_df = cudf.DataFrame.from_pandas(pandas_df)
            elif hasattr(df, 'as_data_frame'):
                # Convert H2O frame to pandas then to cuDF
                pandas_df = df.as_data_frame()
                cudf_df = cudf.DataFrame.from_pandas(pandas_df)
            elif hasattr(df, 'to_pandas'):
                # Generic conversion via pandas
                pandas_df = df.to_pandas()
                cudf_df = cudf.DataFrame.from_pandas(pandas_df)
            else:
                # Try direct conversion (assuming it's a pandas-like DataFrame)
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
            print(f"GPU acceleration failed, falling back to CPU: {e}")
            # Fall back to CPU methods
    
    # Use Spark if requested
    if use_spark:
        # Check if df is a Spark DataFrame
        if hasattr(df, 'write'):
            if format == 'parquet':
                df.write.parquet(file_path, mode="overwrite")
            elif format == 'csv':
                df.write.option("header", "true").csv(file_path, mode="overwrite")
            else:
                raise ValueError(f"Unsupported format for Spark: {format}")
        else:
            # Import RAPIDS for Spark utility if GPU acceleration is requested
            if use_gpu:
                try:
                    from numer_crypto.utils.rapids_utils import enable_spark_rapids
                except ImportError:
                    use_gpu = False
            
            # Create appropriate Spark session
            spark = SparkSession.builder \
                .appName(SPARK_CONFIG.get('app_name', 'NumeraiSpark')) \
                .config("spark.executor.memory", SPARK_CONFIG.get('executor_memory', '4g')) \
                .config("spark.driver.memory", SPARK_CONFIG.get('driver_memory', '4g')) \
                .getOrCreate()
            
            # Enable RAPIDS acceleration for Spark if requested
            if use_gpu:
                try:
                    spark = enable_spark_rapids(spark)
                except Exception as e:
                    print(f"Could not enable RAPIDS for Spark: {e}")
            
            # Convert to Spark DataFrame
            if hasattr(df, 'to_pandas'):
                # Convert from Polars/cuDF to pandas, then to Spark
                spark_df = spark.createDataFrame(df.to_pandas())
            else:
                # Try direct conversion
                spark_df = spark.createDataFrame(df)
                
            if format == 'parquet':
                spark_df.write.parquet(file_path, mode="overwrite")
            elif format == 'csv':
                spark_df.write.option("header", "true").csv(file_path, mode="overwrite")
            else:
                raise ValueError(f"Unsupported format for Spark: {format}")
    else:
        # Use Polars by default
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
                if hasattr(df, 'toPandas'):
                    # Convert Spark DataFrame to pandas then to Polars
                    pl_df = pl.from_pandas(df.toPandas())
                elif hasattr(df, 'as_data_frame'):
                    # Convert H2O frame to pandas then to Polars
                    pl_df = pl.from_pandas(df.as_data_frame())
                elif 'cudf' in str(type(df)):
                    # Convert cuDF to pandas then to Polars
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
        
    return file_path


def load_dataframe(filename, format='parquet', use_spark=False, use_gpu=None):
    """
    Load a DataFrame from the data directory.
    
    Args:
        filename (str): The name of the file
        format (str): The format to load from ('parquet', 'csv', 'pickle')
        use_spark (bool): Whether to use Spark for loading the DataFrame
        use_gpu (bool): Whether to use GPU acceleration if available (None=auto-detect)
        
    Returns:
        DataFrame: The loaded DataFrame (Polars, Spark, or cuDF DataFrame)
    """
    file_path = os.path.join(DATA_DIR, filename)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Auto-detect GPU if not specified
    if use_gpu is None:
        use_gpu = RAPIDS_AVAILABLE and HARDWARE_CONFIG.get('gpu_count', 0) > 0
    
    # Use RAPIDS cuDF if requested and available
    if use_gpu and RAPIDS_AVAILABLE and not use_spark:
        try:
            print(f"Loading {filename} with GPU acceleration (cuDF)")
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
            print(f"GPU acceleration failed, falling back to CPU: {e}")
            # Fall back to CPU methods
    
    if use_spark:
        # Import RAPIDS for Spark utility if GPU acceleration is requested
        if use_gpu:
            try:
                from numer_crypto.utils.rapids_utils import enable_spark_rapids
            except ImportError:
                use_gpu = False
        
        # Initialize Spark
        spark = SparkSession.builder \
            .appName(SPARK_CONFIG.get('app_name', 'NumeraiSpark')) \
            .config("spark.executor.memory", SPARK_CONFIG.get('executor_memory', '4g')) \
            .config("spark.driver.memory", SPARK_CONFIG.get('driver_memory', '4g')) \
            .getOrCreate()
        
        # Enable RAPIDS acceleration for Spark if requested
        if use_gpu:
            try:
                spark = enable_spark_rapids(spark)
                print(f"Loading {filename} with GPU-accelerated Spark")
            except Exception as e:
                print(f"Could not enable RAPIDS for Spark: {e}")
        
        if format == 'parquet':
            return spark.read.parquet(file_path)
        elif format == 'csv':
            return spark.read.option("header", "true").csv(file_path)
        else:
            raise ValueError(f"Unsupported format for Spark: {format}")
    else:
        # Use Polars by default
        if format == 'parquet':
            return pl.read_parquet(file_path)
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


def convert_h2o_to_polars(h2o_frame):
    """
    Convert an H2O frame to a Polars DataFrame.
    
    Args:
        h2o_frame: The H2O frame to convert
        
    Returns:
        DataFrame: The Polars DataFrame
    """
    # H2O frames have as_data_frame() method to convert to pandas
    # Then convert pandas to Polars
    pandas_df = h2o_frame.as_data_frame()
    return pl.from_pandas(pandas_df)


def convert_polars_to_h2o(polars_df, h2o_instance):
    """
    Convert a Polars DataFrame to an H2O frame.
    
    Args:
        polars_df (DataFrame): The Polars DataFrame to convert
        h2o_instance: The H2O instance
        
    Returns:
        H2OFrame: The H2O frame
    """
    # Convert Polars to pandas first
    pandas_df = polars_df.to_pandas()
    return h2o_instance.H2OFrame(pandas_df)


def convert_spark_to_h2o(spark_df, h2o_instance, h2o_context=None):
    """
    Convert a Spark DataFrame to an H2O frame.
    
    Args:
        spark_df: The Spark DataFrame to convert
        h2o_instance: The H2O instance
        h2o_context: Optional H2O context for optimized conversion
        
    Returns:
        H2OFrame: The H2O frame
    """
    if h2o_context is not None:
        # Use optimized H2O context conversion if available
        return h2o_context.asH2OFrame(spark_df)
    else:
        # Fallback to pandas conversion path
        pandas_df = spark_df.toPandas()
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