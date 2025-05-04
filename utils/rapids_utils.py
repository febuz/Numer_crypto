"""
RAPIDS utilities for GPU-accelerated data processing in the Numerai Crypto project.

This module provides utilities for using NVIDIA RAPIDS ecosystem to accelerate
data processing operations with GPUs, including:
- cuDF for GPU-accelerated DataFrame operations
- cuML for GPU-accelerated machine learning
- Integration with Spark via RAPIDS Accelerator for Apache Spark
"""
import os
import sys
import logging
import importlib.util
from functools import wraps

# Set up logging
logger = logging.getLogger(__name__)

def is_package_available(package_name):
    """Check if a package is available/installed."""
    return importlib.util.find_spec(package_name) is not None

# Check for RAPIDS availability
RAPIDS_AVAILABLE = is_package_available('cudf')
SPARK_RAPIDS_AVAILABLE = is_package_available('rapids.spark')
CUML_AVAILABLE = is_package_available('cuml')

def check_rapids_availability():
    """Print a report on RAPIDS components availability."""
    if RAPIDS_AVAILABLE:
        import cudf
        print(f"✓ cuDF available (version: {cudf.__version__})")
    else:
        print("✗ cuDF not available - GPU-accelerated DataFrames not supported")
    
    if CUML_AVAILABLE:
        import cuml
        print(f"✓ cuML available (version: {cuml.__version__})")
    else:
        print("✗ cuML not available - GPU-accelerated ML algorithms not supported")
    
    if SPARK_RAPIDS_AVAILABLE:
        print("✓ RAPIDS Accelerator for Apache Spark available")
    else:
        print("✗ RAPIDS Accelerator for Apache Spark not available")
    
    if not any([RAPIDS_AVAILABLE, CUML_AVAILABLE, SPARK_RAPIDS_AVAILABLE]):
        print("\nTo enable GPU acceleration, install RAPIDS using:")
        print("conda install -c rapidsai -c conda-forge -c nvidia rapids=23.12 python=3.10 cuda-version=11.8")
        print("See setup_env.sh for details")

def pandas_to_cudf(df):
    """
    Convert a pandas DataFrame to a cuDF GPU DataFrame if RAPIDS is available.
    
    Args:
        df: pandas DataFrame to convert
        
    Returns:
        cuDF DataFrame if RAPIDS is available, otherwise the original pandas DataFrame
    """
    if not RAPIDS_AVAILABLE:
        logger.warning("RAPIDS (cuDF) not available, using pandas DataFrame")
        return df
    
    try:
        import cudf
        return cudf.DataFrame.from_pandas(df)
    except Exception as e:
        logger.warning(f"Error converting pandas DataFrame to cuDF: {e}")
        return df

def cudf_to_pandas(df):
    """
    Convert a cuDF GPU DataFrame to a pandas DataFrame.
    
    Args:
        df: cuDF DataFrame to convert
        
    Returns:
        pandas DataFrame
    """
    if 'cudf' not in sys.modules or not isinstance(df, sys.modules['cudf'].DataFrame):
        # Already a pandas DataFrame or similar
        return df
    
    try:
        return df.to_pandas()
    except Exception as e:
        logger.warning(f"Error converting cuDF DataFrame to pandas: {e}")
        return df

def polars_to_cudf(df):
    """
    Convert a Polars DataFrame to a cuDF GPU DataFrame if RAPIDS is available.
    
    Args:
        df: Polars DataFrame to convert
        
    Returns:
        cuDF DataFrame if RAPIDS is available, otherwise the original Polars DataFrame
    """
    if not RAPIDS_AVAILABLE:
        logger.warning("RAPIDS (cuDF) not available, using Polars DataFrame")
        return df
    
    try:
        import cudf
        # Convert Polars to pandas first, then to cuDF
        pandas_df = df.to_pandas()
        return cudf.DataFrame.from_pandas(pandas_df)
    except Exception as e:
        logger.warning(f"Error converting Polars DataFrame to cuDF: {e}")
        return df

def cudf_to_polars(df):
    """
    Convert a cuDF GPU DataFrame to a Polars DataFrame.
    
    Args:
        df: cuDF DataFrame to convert
        
    Returns:
        Polars DataFrame
    """
    if 'cudf' not in sys.modules or not isinstance(df, sys.modules['cudf'].DataFrame):
        # Already a Polars DataFrame or similar
        return df
    
    try:
        import polars as pl
        # Convert to pandas first, then to Polars
        pandas_df = df.to_pandas()
        return pl.from_pandas(pandas_df)
    except Exception as e:
        logger.warning(f"Error converting cuDF DataFrame to Polars: {e}")
        return df

def enable_spark_rapids(spark, gpu_id=None):
    """
    Enable RAPIDS Accelerator for Apache Spark if available.
    
    Args:
        spark: SparkSession to configure
        gpu_id: GPU ID to use (defaults to all available GPUs)
        
    Returns:
        Configured SparkSession with RAPIDS enabled if available
    """
    if not SPARK_RAPIDS_AVAILABLE:
        logger.warning("RAPIDS Accelerator for Apache Spark not available")
        return spark
    
    try:
        from numer_crypto.utils.gpu_utils import get_available_gpus
        
        # Get available GPUs
        gpus = get_available_gpus()
        if not gpus:
            logger.warning("No GPUs detected, not enabling RAPIDS for Spark")
            return spark
        
        # Get RAPIDS jar location
        rapids_jars_path = os.getenv('SPARK_RAPIDS_DIR')
        if not rapids_jars_path and 'CONDA_PREFIX' in os.environ:
            # Try to find RAPIDS jars in conda environment
            import glob
            pattern = f"{os.environ['CONDA_PREFIX']}/lib/python*/site-packages/rapids/jars/*"
            jars = glob.glob(pattern)
            if jars:
                rapids_jars_path = os.path.dirname(jars[0])
        
        if not rapids_jars_path:
            logger.warning("RAPIDS jars not found, not enabling RAPIDS for Spark")
            return spark
        
        # Configure Spark for RAPIDS
        conf = spark.conf
        
        # Basic RAPIDS configuration
        conf.set("spark.plugins", "com.nvidia.spark.SQLPlugin")
        conf.set("spark.rapids.sql.enabled", "true")
        
        # Set GPU allocation
        if gpu_id is not None:
            conf.set("spark.task.resource.gpu.amount", "1")
            conf.set("spark.executor.resource.gpu.amount", "1")
            conf.set("spark.rapids.memory.gpu.allocFraction", "0.9")
            conf.set("spark.rapids.sql.concurrentGpuTasks", "1")
            # Use specific GPU
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        else:
            # Use all available GPUs
            conf.set("spark.task.resource.gpu.amount", "1")
            conf.set("spark.executor.resource.gpu.amount", str(len(gpus)))
            conf.set("spark.rapids.memory.gpu.allocFraction", "0.9")
            conf.set("spark.rapids.sql.concurrentGpuTasks", str(len(gpus)))
            
        # Advanced optimizations
        conf.set("spark.rapids.sql.incompatibleOps.enabled", "true")
        conf.set("spark.rapids.sql.explain", "ALL")
        
        logger.info("RAPIDS Accelerator for Apache Spark enabled")
        return spark
        
    except Exception as e:
        logger.warning(f"Error enabling RAPIDS for Spark: {e}")
        return spark

def with_cudf(func):
    """
    Decorator to automatically convert pandas DataFrames to cuDF on input,
    and back to pandas on output if RAPIDS is available.
    
    Usage:
        @with_cudf
        def process_data(df):
            # df will be a cuDF DataFrame if RAPIDS is available
            # Operations will be GPU-accelerated
            result = df.groupby('col').agg({'value': 'mean'})
            return result  # Will be converted back to pandas
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not RAPIDS_AVAILABLE:
            return func(*args, **kwargs)
        
        # Convert pandas DataFrames in args to cuDF
        converted_args = []
        for arg in args:
            if hasattr(arg, 'to_pandas'):  # Looks like a pandas-like DataFrame
                try:
                    import cudf
                    arg = cudf.DataFrame.from_pandas(arg)
                except Exception:
                    pass
            converted_args.append(arg)
        
        # Convert pandas DataFrames in kwargs to cuDF
        converted_kwargs = {}
        for key, value in kwargs.items():
            if hasattr(value, 'to_pandas'):  # Looks like a pandas-like DataFrame
                try:
                    import cudf
                    value = cudf.DataFrame.from_pandas(value)
                except Exception:
                    pass
            converted_kwargs[key] = value
        
        # Call the function with converted arguments
        result = func(*converted_args, **converted_kwargs)
        
        # Convert result back to pandas if it's a cuDF DataFrame
        if 'cudf' in sys.modules and isinstance(result, sys.modules['cudf'].DataFrame):
            result = result.to_pandas()
        
        return result
    
    return wrapper

def cuml_model_factory(model_type, **kwargs):
    """
    Create a GPU-accelerated ML model using cuML if available,
    falling back to scikit-learn if not.
    
    Args:
        model_type (str): Type of model to create (e.g., 'random_forest', 'xgboost')
        **kwargs: Arguments for the model constructor
        
    Returns:
        ML model instance (cuML if available, otherwise scikit-learn)
    """
    if not CUML_AVAILABLE:
        logger.warning(f"cuML not available, using scikit-learn for {model_type}")
        if model_type == 'random_forest':
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(**kwargs)
        elif model_type == 'xgboost':
            import xgboost as xgb
            return xgb.XGBRegressor(**kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    try:
        import cuml
        
        if model_type == 'random_forest':
            return cuml.ensemble.RandomForestRegressor(**kwargs)
        elif model_type == 'xgboost':
            # cuML doesn't have XGBoost, but we can configure xgboost to use GPU
            import xgboost as xgb
            kwargs['tree_method'] = 'gpu_hist'
            kwargs['predictor'] = 'gpu_predictor'
            kwargs['gpu_id'] = kwargs.get('gpu_id', 0)
            return xgb.XGBRegressor(**kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
    except Exception as e:
        logger.warning(f"Error creating cuML model: {e}")
        # Fall back to scikit-learn
        if model_type == 'random_forest':
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(**kwargs)
        elif model_type == 'xgboost':
            import xgboost as xgb
            return xgb.XGBRegressor(**kwargs)

if __name__ == "__main__":
    # Print available components when run as a script
    print("RAPIDS Acceleration Components:")
    print("-" * 30)
    check_rapids_availability()