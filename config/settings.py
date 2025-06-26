"""
Configuration settings for the Numerai Crypto project.
"""
import os
from pathlib import Path

# Try to load environment variables from .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Loaded environment variables from .env file")
except ImportError:
    print("dotenv package not installed, skipping .env loading")
    # Default values for API keys (will be overridden by environment variables if set)
    if 'NUMERAI_PUBLIC_ID' not in os.environ:
        os.environ['NUMERAI_PUBLIC_ID'] = 'dummy_public_id'
    if 'NUMERAI_SECRET_KEY' not in os.environ:
        os.environ['NUMERAI_SECRET_KEY'] = 'dummy_secret_key'

# Base directories - code stays in repo
BASE_DIR = Path(__file__).resolve().parent.parent

# External data directory - all data outside repo
EXTERNAL_DATA_DIR = "/media/knight2/EDB/numer_crypto_temp"
DATA_DIR = os.path.join(EXTERNAL_DATA_DIR, 'data')
MODELS_DIR = os.path.join(EXTERNAL_DATA_DIR, 'models')
SUBMISSION_DIR = os.path.join(EXTERNAL_DATA_DIR, 'submission')  # Main submission directory (singular)
LOG_DIR = os.path.join(EXTERNAL_DATA_DIR, 'log')

# Create these directories if they don't exist
os.makedirs(EXTERNAL_DATA_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)  # Main submission directory
os.makedirs(LOG_DIR, exist_ok=True)

# Subdirectories
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
FEATURES_DIR = os.path.join(DATA_DIR, 'features')
CHECKPOINTS_DIR = os.path.join(MODELS_DIR, 'checkpoints')

# Directory for data splits
TRAIN_DIR = os.path.join(PROCESSED_DATA_DIR, 'train')
VALIDATION_DIR = os.path.join(PROCESSED_DATA_DIR, 'validation')
PREDICTION_DIR = os.path.join(PROCESSED_DATA_DIR, 'prediction')

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
    'ice_root': '/media/knight2/EDB/tmp/h2o', # Use correct disk for temp files
    'jvm_custom_args': ['-Djava.io.tmpdir=/media/knight2/EDB/tmp/h2o'], # Force Java temp dir
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

# Feature generation configuration
FEATURE_GENERATION_CONFIG = {
    # Column limits - very high to avoid bottlenecks
    'max_columns': 5000000,           # 5M columns - no practical limit
    'max_columns_limited': 10000,     # For limited feature mode
    'max_columns_moderate': 100000,   # For moderate feature mode
    
    # Feature types
    'window_sizes': [7, 14, 28, 56],  # Rolling windows
    'lag_periods': [1, 3, 7, 14, 28], # Lag features
    'ewm_spans': [5, 10, 20, 40],     # Exponential weighted moving averages
    
    # Limited feature configuration (for max_features < 1000)
    'limited_config': {
        'window_sizes': [14],
        'lag_periods': [1, 7],
        'ewm_spans': [10],
        'max_columns': 10000,
    },
    
    # Moderate feature configuration (for max_features < 5000)
    'moderate_config': {
        'window_sizes': [7, 14],
        'lag_periods': [1, 3, 7],
        'ewm_spans': [10, 20],
        'max_columns': 100000,
    },
    
    # Data columns to exclude from feature generation
    'excluded_columns': {
        'date_col': 'date',
        'target_cols': ['target'],
        'id_cols': ['era', 'id', 'symbol'],
    },
    
    # GPU processing configuration
    'gpu_config': {
        'memory_cleanup_interval': 50,  # Clean memory every N symbols
        'progress_log_interval': 10,    # Log progress every N symbols
        'batch_processing': True,       # Process in batches for memory efficiency
        'timeout_minutes': 30,          # Timeout per GPU worker
    },
    
    # Multiprocessing configuration
    'multiprocessing_config': {
        'start_method': 'spawn',        # Required for CUDA compatibility
        'max_workers': 3,               # Number of GPU workers
        'timeout_seconds': 1800,        # 30 minutes timeout per worker
    }
}