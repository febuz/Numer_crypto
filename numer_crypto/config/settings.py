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

# Spark configuration
SPARK_CONFIG = {
    'app_name': 'NumeraiSparklingWater',
    'executor_memory': '4g',
    'driver_memory': '4g',
    'executor_cores': '2',
    'driver_java_options': '-XX:+UseG1GC',
    'executor_java_options': '-XX:+UseG1GC',
}

# H2O configuration
H2O_CONFIG = {
    'max_mem_size': '4g',
}

# Model parameters
XGBOOST_PARAMS = {
    'ntrees': 500,
    'max_depth': 6,
    'learn_rate': 0.1,
    'sample_rate': 0.8,
    'col_sample_rate': 0.8,
}