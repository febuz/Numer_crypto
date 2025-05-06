#!/usr/bin/env python3
"""
Run Functional Tests for Numerai Crypto

This script runs the functional tests to verify all components are working correctly:
- Basic ML functionality
- H2O and Sparkling Water
- PySpark and RAPIDS
- GPU support
- Feature store and checkpoint system

Usage:
    python scripts/run_functional_tests.py [--test-type TYPE] [--output-dir DIR]
"""

import os
import sys
import subprocess
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"functional_tests_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run Functional Tests for Numerai Crypto')
    parser.add_argument('--test-type', type=str, default='all',
                        choices=['all', 'basic', 'h2o', 'gpu', 'feature_store'],
                        help='Type of tests to run')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for test results')
    return parser.parse_args()

def run_test(test_path, description=None):
    """
    Run a test script and capture the output
    
    Args:
        test_path: Path to the test script
        description: Description of the test
        
    Returns:
        dict: Test results including exit code and output
    """
    if description:
        logger.info(f"Running test: {description}")
    else:
        logger.info(f"Running test: {test_path}")
    
    try:
        result = subprocess.run(
            [sys.executable, test_path],
            capture_output=True,
            text=True,
            check=False  # Don't raise exception on non-zero exit code
        )
        
        success = result.returncode == 0
        
        # Log result
        if success:
            logger.info(f"✅ Test passed: {test_path}")
        else:
            logger.error(f"❌ Test failed: {test_path}")
            logger.error(f"Error output: {result.stderr}")
        
        return {
            'path': test_path,
            'description': description,
            'success': success,
            'exit_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
    except Exception as e:
        logger.error(f"Error running test {test_path}: {e}")
        return {
            'path': test_path,
            'description': description,
            'success': False,
            'exit_code': -1,
            'error': str(e)
        }

def run_basic_tests():
    """Run basic functionality tests"""
    logger.info("=== Running Basic Functionality Tests ===")
    
    # Get project root
    project_root = Path(__file__).resolve().parent.parent
    tests_dir = project_root / "tests" / "functional"
    
    results = []
    
    # Test basic functionality
    results.append(run_test(
        tests_dir / "test_basic.py",
        "Basic ML functionality"
    ))
    
    # Test minimal functionality
    results.append(run_test(
        tests_dir / "test_minimal.py",
        "Minimal functionality"
    ))
    
    # Test hardware detection
    results.append(run_test(
        tests_dir / "test_hardware.py",
        "Hardware detection"
    ))
    
    # Test feature store
    results.append(run_test(
        project_root / "scripts" / "feature_store.py",
        "Feature store functionality"
    ))
    
    # Test model checkpoint
    results.append(run_test(
        project_root / "scripts" / "model_checkpoint.py",
        "Model checkpoint functionality"
    ))
    
    return results

def run_h2o_tests():
    """Run H2O and Sparkling Water tests"""
    logger.info("=== Running H2O and Sparkling Water Tests ===")
    
    # Get project root
    project_root = Path(__file__).resolve().parent.parent
    tests_dir = project_root / "tests" / "functional"
    
    results = []
    
    # Test H2O functionality
    results.append(run_test(
        tests_dir / "test_h2o_simple.py",
        "Basic H2O functionality"
    ))
    
    # Test H2O with Java 17
    results.append(run_test(
        tests_dir / "test_h2o_java17_simple.py",
        "H2O with Java 17"
    ))
    
    # Test minimal H2O Sparkling Water
    results.append(run_test(
        tests_dir / "test_h2o_sparkling_minimal.py",
        "Minimal H2O Sparkling Water"
    ))
    
    # Test PySparkling import
    results.append(run_test(
        tests_dir / "test_pysparkling_import.py",
        "PySparkling import"
    ))
    
    return results

def run_gpu_tests():
    """Run GPU tests"""
    logger.info("=== Running GPU Tests ===")
    
    # Get project root
    project_root = Path(__file__).resolve().parent.parent
    
    results = []
    
    # Run the GPU capability test script
    results.append(run_test(
        project_root / "scripts" / "test_gpu_capabilities.py",
        "GPU capabilities test"
    ))
    
    # Test RAPIDS
    performance_tests_dir = project_root / "tests" / "performance"
    if (performance_tests_dir / "test_rapids.py").exists():
        results.append(run_test(
            performance_tests_dir / "test_rapids.py",
            "RAPIDS performance"
        ))
    else:
        functional_tests_dir = project_root / "tests" / "functional"
        results.append(run_test(
            functional_tests_dir / "test_rapids.py",
            "RAPIDS functionality"
        ))
    
    return results

def run_feature_store_tests():
    """Run custom tests for feature store and model checkpoint"""
    logger.info("=== Running Feature Store and Model Checkpoint Tests ===")
    
    # Get project root
    project_root = Path(__file__).resolve().parent.parent
    
    results = []
    
    # Test the feature store
    from feature_store import FeatureStore
    import tempfile
    import pandas as pd
    import numpy as np
    
    try:
        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize feature store
            store = FeatureStore(temp_dir)
            
            # Create a test dataframe
            df = pd.DataFrame({
                'feature1': np.random.rand(100),
                'feature2': np.random.rand(100),
                'feature3': np.random.rand(100)
            })
            
            # Store features
            store.store_features(df, 'test_features', metadata={'source': 'test'})
            
            # Retrieve features
            retrieved_df = store.get_features('test_features')
            
            # Verify
            success = retrieved_df is not None and len(retrieved_df) == len(df)
            
            logger.info(f"Feature store test: {'Passed' if success else 'Failed'}")
            results.append({
                'description': 'Feature store functionality',
                'success': success
            })
            
    except Exception as e:
        logger.error(f"Error testing feature store: {e}")
        results.append({
            'description': 'Feature store functionality',
            'success': False,
            'error': str(e)
        })
    
    # Test the model checkpoint
    try:
        from model_checkpoint import ModelCheckpoint
        
        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize checkpoint system
            checkpoint = ModelCheckpoint(temp_dir)
            
            # Mock metrics
            metrics = {
                'validation_auc': 0.75,
                'training_time': 10.5
            }
            
            # Mock model (just use a dictionary)
            mock_model = {'model_data': 'test'}
            
            # Save the model
            success1 = checkpoint.save_model(
                mock_model,
                'test_model',
                model_type='generic',
                metrics=metrics
            )
            
            # List models
            models = checkpoint.list_models()
            
            # Retrieve model
            loaded_model, metadata = checkpoint.load_model('test_model')
            
            success2 = loaded_model is not None and metadata is not None
            
            logger.info(f"Model checkpoint test: {'Passed' if (success1 and success2) else 'Failed'}")
            results.append({
                'description': 'Model checkpoint functionality',
                'success': success1 and success2
            })
            
    except Exception as e:
        logger.error(f"Error testing model checkpoint: {e}")
        results.append({
            'description': 'Model checkpoint functionality',
            'success': False,
            'error': str(e)
        })
    
    return results

def main():
    """Main function to run functional tests"""
    args = parse_args()
    
    # Create output directory
    output_dir = args.output_dir
    if not output_dir:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        project_root = Path(__file__).resolve().parent.parent
        output_dir = project_root / "reports" / f"functional_tests_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Test results will be saved to: {output_dir}")
    
    # Dictionary to store all test results
    all_results = {}
    
    # Run selected tests
    if args.test_type in ['all', 'basic']:
        basic_results = run_basic_tests()
        all_results['basic'] = basic_results
    
    if args.test_type in ['all', 'h2o']:
        h2o_results = run_h2o_tests()
        all_results['h2o'] = h2o_results
    
    if args.test_type in ['all', 'gpu']:
        gpu_results = run_gpu_tests()
        all_results['gpu'] = gpu_results
    
    if args.test_type in ['all', 'feature_store']:
        feature_store_results = run_feature_store_tests()
        all_results['feature_store'] = feature_store_results
    
    # Calculate overall statistics
    total_tests = 0
    passed_tests = 0
    
    for category, results in all_results.items():
        for result in results:
            total_tests += 1
            if result.get('success', False):
                passed_tests += 1
    
    # Save results to file
    results_path = Path(output_dir) / "test_results.json"
    with open(results_path, 'w') as f:
        # Clean results to ensure serializable
        clean_results = {}
        for category, results in all_results.items():
            clean_results[category] = []
            for result in results:
                # Limit stdout/stderr to avoid huge files
                if 'stdout' in result and result['stdout'] and len(result['stdout']) > 2000:
                    result['stdout'] = result['stdout'][:2000] + "... (truncated)"
                if 'stderr' in result and result['stderr'] and len(result['stderr']) > 2000:
                    result['stderr'] = result['stderr'][:2000] + "... (truncated)"
                clean_results[category].append(result)
        
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'test_type': args.test_type,
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'results': clean_results
        }, f, indent=2)
    
    # Print summary
    logger.info("\n=== Test Summary ===")
    logger.info(f"Total tests: {total_tests}")
    logger.info(f"Passed tests: {passed_tests}")
    logger.info(f"Success rate: {passed_tests / total_tests * 100:.1f}%")
    logger.info(f"Results saved to: {results_path}")
    
    # Return 0 if all tests passed, 1 otherwise
    return 0 if passed_tests == total_tests else 1

if __name__ == "__main__":
    sys.exit(main())