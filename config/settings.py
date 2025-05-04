"""
Configuration settings for the Numerai Crypto project.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# External storage configuration
EXTERNAL_STORAGE = os.getenv('EXTERNAL_STORAGE_PATH')

# Numerai API credentials
NUMERAI_PUBLIC_ID = os.getenv('NUMERAI_PUBLIC_ID')
NUMERAI_SECRET_KEY = os.getenv('NUMERAI_SECRET_KEY')

# Hardware resources available
HARDWARE_CONFIG = {
    'total_memory': '640g',  # 640GB of RAM
    'gpu_count': 3,          # 3 GPUs
    'gpu_memory': '24g',     # 24GB per GPU
    'gpu_model': 'RTX5000',  # RTX5000 Ampere
}

# Spark configuration - optimized for high memory
SPARK_CONFIG = {
    'app_name': 'NumeraiSparklingWater',
    'master': 'local[*]',                      # Use all available cores
    'executor_memory': '128g',                 # Allocate substantial memory per executor
    'driver_memory': '64g',                    # Allocate memory for driver
    'executor_cores': '8',                     # Use multiple cores per executor
    'num_executors': '4',                      # Multiple executors to utilize memory
    'driver_java_options': '-XX:+UseG1GC',     # G1GC for large heap sizes
    'executor_java_options': '-XX:+UseG1GC',   # G1GC for executors too
    'spark.memory.fraction': '0.8',            # Use more memory for execution
    'spark.memory.storageFraction': '0.3',     # Dedicate more to storage
    'spark.driver.maxResultSize': '16g',       # Allow larger result sets
    'spark.local.dir': os.getenv('SPARK_TEMP_DIR', '/tmp'),  # Temp directory for shuffle data
    'spark.sql.shuffle.partitions': '1000',    # More partitions for better parallelism
    'spark.default.parallelism': '100',        # Higher parallelism for operations
}

# H2O configuration - optimized for high memory and GPUs
H2O_CONFIG = {
    'max_mem_size': '256g',                    # Allocate significant memory to H2O
    'nthreads': -1,                            # Use all available threads
    'backend': 'gpu',                          # Enable GPU backend if available in H2O version
    'gpu_id': list(range(HARDWARE_CONFIG['gpu_count'])),  # Use all GPUs
    'allow_large_jvms': True,                  # Allow large heap sizes
    'h2o_cluster_startup_timeout': 600,        # Longer timeout for big clusters
}

# Model parameters - optimized for GPUs and high memory
XGBOOST_PARAMS = {
    'ntrees': 1000,                # More trees for better accuracy
    'max_depth': 12,               # Deeper trees 
    'learn_rate': 0.05,            # Lower learning rate for better convergence
    'sample_rate': 0.8,            # Subsample rows
    'col_sample_rate': 0.8,        # Subsample columns
    'tree_method': 'gpu_hist',     # Use GPU for tree building
    'predictor': 'gpu_predictor',  # Use GPU for prediction
    'grow_policy': 'lossguide',    # Loss guided growth for better trees
    'max_bin': 256,                # More bins for better splits
    'min_child_weight': 1,         # Ensure some minimum weight in children
    'backend': 'gpu',              # Use GPU backend if available
    'gpu_id': 0,                   # Primary GPU - can be changed dynamically
}

# LightGBM parameters (alternative model)
LIGHTGBM_PARAMS = {
    'num_leaves': 255,             # More leaves for deep trees
    'learning_rate': 0.05,         # Learning rate
    'max_depth': 12,               # Maximum tree depth
    'min_data_in_leaf': 100,       # Minimum data in leaf
    'feature_fraction': 0.8,       # Column subsampling
    'bagging_fraction': 0.8,       # Row subsampling
    'bagging_freq': 5,             # Bagging frequency
    'device_type': 'gpu',          # Use GPU
    'max_bin': 255,                # Max number of bins
    'verbosity': -1,               # Less output
    'boosting_type': 'gbdt',       # Standard gradient boosting
    'objective': 'regression',     # For regression tasks
    'metric': 'rmse',              # Evaluation metric
    'n_estimators': 1000,          # Number of trees
}