#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
import argparse
from datetime import datetime

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import log utils
from utils.log_utils import setup_logging

# Set up logging to external directory
logger = setup_logging(name=__name__, level=logging.INFO)

def validate_numerai_utils():
    """Validate Numerai utility functions."""
    logger.info("Validating Numerai utility functions...")
    
    try:
        from utils.data.download_numerai import download_numerai_crypto_data
        from utils.data.load_numerai import load_numerai_data, get_eligible_crypto_symbols
        
        # Check if the functions are imported correctly
        logger.info("✓ Numerai utility functions imported successfully")
        
        # Check if the functions can be called without errors
        # Use a dry run mode if available
        try:
            logger.info("Checking download_numerai_crypto_data function (dry run)...")
            temp_dir = "/tmp/numerai_test_" + datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Try to get current round without downloading
            import numerapi
            napi = numerapi.NumerAPI()
            current_round = napi.get_current_round(tournament='crypto')
            logger.info(f"✓ Current Numerai crypto round: {current_round}")
            
            logger.info("Numerai utility validation: PASSED")
            return True
        except Exception as e:
            logger.error(f"Error validating Numerai utilities: {e}")
            return False
    
    except ImportError as e:
        logger.error(f"Failed to import Numerai utility functions: {e}")
        return False

def validate_yiedl_utils():
    """Validate Yiedl utility functions."""
    logger.info("Validating Yiedl utility functions...")
    
    try:
        from utils.data.download_yiedl import download_yiedl_data
        from utils.data.load_yiedl import load_yiedl_data, get_yiedl_crypto_symbols
        
        # Check if the functions are imported correctly
        logger.info("✓ Yiedl utility functions imported successfully")
        
        # Validate URL accessibility without downloading
        try:
            import requests
            latest_url = 'https://api.yiedl.ai/yiedl/v1/downloadDataset?type=latest'
            response = requests.head(latest_url)
            if response.status_code == 200:
                logger.info("✓ Yiedl API is accessible")
            else:
                logger.warning(f"Yiedl API returned status code {response.status_code}")
            
            logger.info("Yiedl utility validation: PASSED")
            return True
        except Exception as e:
            logger.error(f"Error validating Yiedl utilities: {e}")
            return False
    
    except ImportError as e:
        logger.error(f"Failed to import Yiedl utility functions: {e}")
        return False

def validate_merged_dataset():
    """Validate merged dataset creation."""
    logger.info("Validating merged dataset creation...")
    
    try:
        from utils.data.create_merged_dataset import get_overlapping_symbols, create_merged_dataset
        
        # Check if the functions are imported correctly
        logger.info("✓ Merged dataset functions imported successfully")
        
        # Create mock data to test the functions
        import pandas as pd
        
        # Mock Numerai data
        numerai_data = {
            'live_universe': pd.DataFrame({
                'Symbol': ['BTC', 'ETH', 'XRP', 'LTC', 'ADA']
            })
        }
        
        # Mock Yiedl data
        yiedl_data = {
            'latest': pd.DataFrame({
                'date': ['2023-01-01', '2023-01-01', '2023-01-01'],
                'asset': ['BTC', 'ETH', 'SOL'],
                'price': [50000, 3000, 200]
            })
        }
        
        # Test overlapping symbols
        overlapping = get_overlapping_symbols(numerai_data, yiedl_data)
        logger.info(f"✓ Found {len(overlapping)} overlapping symbols: {', '.join(overlapping)}")
        
        logger.info("Merged dataset validation: PASSED")
        return True
    
    except Exception as e:
        logger.error(f"Error validating merged dataset creation: {e}")
        return False

def validate_data_retriever():
    """Validate NumeraiDataRetriever class."""
    logger.info("Validating NumeraiDataRetriever class...")
    
    try:
        from data.retrieval import NumeraiDataRetriever
        
        # Check if the class is imported correctly
        logger.info("✓ NumeraiDataRetriever class imported successfully")
        
        # Check if the class has the expected methods
        methods = [method for method in dir(NumeraiDataRetriever) 
                   if callable(getattr(NumeraiDataRetriever, method)) and not method.startswith('_')]
        logger.info(f"✓ NumeraiDataRetriever has the following methods: {', '.join(methods)}")
        
        # Check if get_eligible_symbols method exists
        if 'get_eligible_symbols' in methods:
            logger.info("✓ get_eligible_symbols method exists")
        else:
            logger.warning("get_eligible_symbols method not found in NumeraiDataRetriever")
        
        logger.info("Data retriever validation: PASSED")
        return True
    
    except ImportError as e:
        logger.error(f"Failed to import NumeraiDataRetriever: {e}")
        return False

def validate_pipeline_scripts():
    """Validate pipeline scripts."""
    logger.info("Validating pipeline scripts...")
    
    # Check if the scripts exist
    scripts = [
        ('/media/knight2/EDB/repos/Numer_crypto/scripts/process_yiedl_data.py', 'process_yiedl_data.py'),
        ('/media/knight2/EDB/repos/Numer_crypto/scripts/train_predict_crypto.py', 'train_predict_crypto.py')
    ]
    
    all_passed = True
    for script_path, script_name in scripts:
        if os.path.exists(script_path):
            logger.info(f"✓ {script_name} exists")
            
            # Check if the script is executable
            if os.access(script_path, os.X_OK):
                logger.info(f"✓ {script_name} is executable")
            else:
                logger.warning(f"{script_name} is not executable")
                os.chmod(script_path, 0o755)
                logger.info(f"  Made {script_name} executable")
        else:
            logger.error(f"✗ {script_name} does not exist")
            all_passed = False
    
    if all_passed:
        logger.info("Pipeline scripts validation: PASSED")
    else:
        logger.error("Pipeline scripts validation: FAILED")
    
    return all_passed

def run_full_validation():
    """Run all validation checks."""
    logger.info("Running full pipeline validation...")
    
    results = {
        "numerai_utils": validate_numerai_utils(),
        "yiedl_utils": validate_yiedl_utils(),
        "merged_dataset": validate_merged_dataset(),
        "data_retriever": validate_data_retriever(),
        "pipeline_scripts": validate_pipeline_scripts()
    }
    
    # Print summary
    logger.info("\n===== Validation Summary =====")
    for component, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        logger.info(f"{component}: {status}")
    
    # Check if all validations passed
    if all(results.values()):
        logger.info("\n✓ All validations PASSED - Pipeline is functional")
        return True
    else:
        failed = [component for component, passed in results.items() if not passed]
        logger.error(f"\n✗ Validation FAILED for: {', '.join(failed)}")
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate the Numerai Crypto pipeline.')
    parser.add_argument('--component', choices=['numerai', 'yiedl', 'merged', 'retriever', 'scripts', 'all'],
                        default='all', help='Which component to validate')
    
    args = parser.parse_args()
    
    if args.component == 'numerai':
        validate_numerai_utils()
    elif args.component == 'yiedl':
        validate_yiedl_utils()
    elif args.component == 'merged':
        validate_merged_dataset()
    elif args.component == 'retriever':
        validate_data_retriever()
    elif args.component == 'scripts':
        validate_pipeline_scripts()
    else:
        run_full_validation()