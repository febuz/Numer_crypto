#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import unittest
import logging
import argparse

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import log utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.log_utils import setup_logging

# Set up logging to external directory
logger = setup_logging(name=__name__, level=logging.INFO)

def run_all_tests():
    """Run all test modules."""
    # Discover and run all tests
    test_loader = unittest.TestLoader()
    start_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Only include test_*.py files
    pattern = 'test_*.py'
    
    test_suite = test_loader.discover(start_dir, pattern=pattern)
    test_runner = unittest.TextTestRunner(verbosity=2)
    
    logger.info(f"Discovered test modules in {start_dir}")
    
    result = test_runner.run(test_suite)
    return result.wasSuccessful()

def run_specific_test(test_name):
    """Run a specific test module."""
    test_loader = unittest.TestLoader()
    
    # Try to load the specified test
    try:
        if test_name.endswith('.py'):
            test_name = test_name[:-3]
        
        # Form the module name
        if not test_name.startswith('test_'):
            test_name = 'test_' + test_name
        
        module_name = f'tests.{test_name}'
        test_suite = test_loader.loadTestsFromName(module_name)
        
        logger.info(f"Running test module: {module_name}")
        
        test_runner = unittest.TextTestRunner(verbosity=2)
        result = test_runner.run(test_suite)
        return result.wasSuccessful()
    
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to load test {test_name}: {e}")
        return False

def run_test_class(test_name, class_name):
    """Run a specific test class."""
    test_loader = unittest.TestLoader()
    
    # Try to load the specified test class
    try:
        if test_name.endswith('.py'):
            test_name = test_name[:-3]
        
        # Form the module and class name
        if not test_name.startswith('test_'):
            test_name = 'test_' + test_name
        
        module_name = f'tests.{test_name}.{class_name}'
        test_suite = test_loader.loadTestsFromName(module_name)
        
        logger.info(f"Running test class: {module_name}")
        
        test_runner = unittest.TextTestRunner(verbosity=2)
        result = test_runner.run(test_suite)
        return result.wasSuccessful()
    
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to load test class {module_name}: {e}")
        return False

def run_mock_data_test():
    """Generate mock data and report statistics."""
    from tests.mock_data_generator import generate_full_mock_dataset
    import tempfile
    
    logger.info("Running mock data generation test...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        mock_data = generate_full_mock_dataset(temp_dir)
        
        logger.info("\nMock Dataset Statistics:")
        
        # Numerai data stats
        numerai_symbols = mock_data['numerai_data']['symbols']
        logger.info(f"  Numerai Symbols: {len(numerai_symbols)} ({', '.join(numerai_symbols[:5])}...)")
        
        # Yiedl data stats
        yiedl_symbols = mock_data['yiedl_data']['symbols']
        logger.info(f"  Yiedl Symbols: {len(yiedl_symbols)} ({', '.join(yiedl_symbols[:5])}...)")
        
        # Overlap stats
        overlap_symbols = mock_data['overlap_symbols']
        logger.info(f"  Overlapping Symbols: {len(overlap_symbols)} ({', '.join(overlap_symbols[:5])}...)")
        
        # Compare with defined overlap percentage
        expected_overlap = len(numerai_symbols) * 0.7  # From generate_yiedl_mock_data default
        logger.info(f"  Expected Overlap: ~{expected_overlap:.1f} symbols")
        logger.info(f"  Actual Overlap: {len(overlap_symbols)} symbols")
        
        return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Run Numerai Crypto Pipeline tests.')
    parser.add_argument('--test', help='Run a specific test module (without the test_ prefix)')
    parser.add_argument('--class', dest='class_name', help='Run a specific test class')
    parser.add_argument('--mock', action='store_true', help='Run mock data generation test')
    
    args = parser.parse_args()
    
    if args.mock:
        success = run_mock_data_test()
    elif args.test and args.class_name:
        success = run_test_class(args.test, args.class_name)
    elif args.test:
        success = run_specific_test(args.test)
    else:
        success = run_all_tests()
    
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())