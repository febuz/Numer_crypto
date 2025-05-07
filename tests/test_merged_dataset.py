#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import unittest
import tempfile
import logging
import pandas as pd
from unittest.mock import patch, MagicMock

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import log utils
from utils.log_utils import setup_logging

# Set up logging to external directory - set create_file=False for unit tests
logger = setup_logging(name=__name__, level=logging.INFO, create_file=False)

# Import the mock data generator
from tests.mock_data_generator import generate_full_mock_dataset

class TestOverlappingSymbols(unittest.TestCase):
    """Tests for finding overlapping symbols between datasets."""
    
    def setUp(self):
        """Set up test case with mock data."""
        # Create a temporary directory and generate full mock dataset
        self.temp_dir = tempfile.TemporaryDirectory()
        self.mock_data = generate_full_mock_dataset(self.temp_dir.name)
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def test_get_overlapping_symbols(self):
        """Test finding overlapping symbols between Numerai and Yiedl data."""
        from utils.data.create_merged_dataset import get_overlapping_symbols
        
        # Load Numerai data
        from utils.data.load_numerai import load_numerai_data
        numerai_files = {
            'train_data': self.mock_data['numerai_data']['train_data'],
            'live_universe': self.mock_data['numerai_data']['live_universe'],
            'current_round': 123
        }
        numerai_data = load_numerai_data(numerai_files)
        
        # Load Yiedl data
        from utils.data.load_yiedl import load_yiedl_data
        yiedl_files = {
            'latest': self.mock_data['yiedl_data']['latest']
        }
        yiedl_data = load_yiedl_data(yiedl_files)
        
        # Find overlapping symbols
        overlapping = get_overlapping_symbols(numerai_data, yiedl_data)
        
        # Check that overlapping symbols were found
        self.assertGreater(len(overlapping), 0)
        
        # Check that the overlapping symbols match what we expect
        expected_overlap = self.mock_data['overlap_symbols']
        self.assertEqual(set(overlapping), set(expected_overlap))
    
    def test_get_overlapping_symbols_no_overlap(self):
        """Test finding overlapping symbols when there are none."""
        from utils.data.create_merged_dataset import get_overlapping_symbols
        
        # Create mock data with no overlap
        numerai_data = {
            'live_universe': pd.DataFrame({
                'Symbol': ['BTC', 'ETH', 'XRP']
            })
        }
        
        yiedl_data = {
            'latest': pd.DataFrame({
                'date': ['2023-01-01', '2023-01-01'],
                'asset': ['SOL', 'AVAX'],
                'price': [200, 30]
            })
        }
        
        # Find overlapping symbols
        overlapping = get_overlapping_symbols(numerai_data, yiedl_data)
        
        # Check that no overlapping symbols were found
        self.assertEqual(len(overlapping), 0)
    
    def test_get_overlapping_symbols_case_insensitivity(self):
        """Test that symbol finding is case insensitive."""
        from utils.data.create_merged_dataset import get_overlapping_symbols
        
        # Create mock data with different case
        numerai_data = {
            'live_universe': pd.DataFrame({
                'Symbol': ['BTC', 'ETH', 'XRP']
            })
        }
        
        yiedl_data = {
            'latest': pd.DataFrame({
                'date': ['2023-01-01', '2023-01-01', '2023-01-01'],
                'asset': ['btc', 'eth', 'xrp'],  # lowercase
                'price': [50000, 3000, 1]
            })
        }
        
        # Find overlapping symbols
        overlapping = get_overlapping_symbols(numerai_data, yiedl_data)
        
        # Check that all symbols were found despite case difference
        self.assertEqual(len(overlapping), 3)
        self.assertEqual(set(overlapping), {'BTC', 'ETH', 'XRP'})

class TestMergedDatasetCreation(unittest.TestCase):
    """Tests for creating merged datasets."""
    
    def setUp(self):
        """Set up test case with mock data."""
        # Create a temporary directory and generate full mock dataset
        self.temp_dir = tempfile.TemporaryDirectory()
        self.mock_data = generate_full_mock_dataset(self.temp_dir.name)
        
        # Load Numerai data
        from utils.data.load_numerai import load_numerai_data
        numerai_files = {
            'train_data': self.mock_data['numerai_data']['train_data'],
            'live_universe': self.mock_data['numerai_data']['live_universe'],
            'current_round': 123
        }
        self.numerai_data = load_numerai_data(numerai_files)
        
        # Load Yiedl data
        from utils.data.load_yiedl import load_yiedl_data
        yiedl_files = {
            'latest': self.mock_data['yiedl_data']['latest']
        }
        self.yiedl_data = load_yiedl_data(yiedl_files)
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def test_create_merged_dataset(self):
        """Test creating a merged dataset from Numerai and Yiedl data."""
        from utils.data.create_merged_dataset import create_merged_dataset
        
        # Create merged dataset
        merged_data = create_merged_dataset(self.numerai_data, self.yiedl_data)
        
        # Check that the merged dataset is not empty
        self.assertGreater(merged_data.shape[0], 0)
        
        # Check that the merged dataset has the expected columns
        numerai_cols = set(self.numerai_data['train_data'].columns)
        yiedl_cols = set(self.yiedl_data['latest'].columns)
        
        # There should be at least one column from each dataset in the merged dataset
        merged_cols = set(merged_data.columns)
        self.assertTrue(any(col in merged_cols for col in numerai_cols))
        self.assertTrue(any(col in merged_cols for col in yiedl_cols))
        
        # Check that only overlapping symbols are included
        from utils.data.create_merged_dataset import get_overlapping_symbols
        overlapping = get_overlapping_symbols(self.numerai_data, self.yiedl_data)
        
        # Convert column to lowercase for comparison
        if 'symbol' in merged_data.columns:
            symbols_in_merged = set(merged_data['symbol'].str.upper())
        elif 'Symbol' in merged_data.columns:
            symbols_in_merged = set(merged_data['Symbol'].str.upper())
        elif 'asset' in merged_data.columns:
            symbols_in_merged = set(merged_data['asset'].str.upper())
        else:
            self.fail("No symbol column found in merged dataset")
        
        overlapping_upper = set(s.upper() for s in overlapping)
        self.assertEqual(symbols_in_merged, overlapping_upper)
    
    def test_create_merged_dataset_with_empty_data(self):
        """Test creating a merged dataset with empty data."""
        from utils.data.create_merged_dataset import create_merged_dataset
        
        # Create empty dataframes
        empty_numerai_data = {
            'train_data': pd.DataFrame(),
            'live_universe': pd.DataFrame()
        }
        
        empty_yiedl_data = {
            'latest': pd.DataFrame()
        }
        
        # Create merged dataset
        merged_data = create_merged_dataset(empty_numerai_data, empty_yiedl_data)
        
        # Check that the merged dataset is empty
        self.assertEqual(merged_data.shape[0], 0)
    
    def test_create_merged_dataset_with_mismatched_dates(self):
        """Test creating a merged dataset with mismatched dates."""
        from utils.data.create_merged_dataset import create_merged_dataset
        
        # Create mock data with non-overlapping dates
        numerai_data = {
            'train_data': pd.DataFrame({
                'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
                'symbol': ['BTC', 'BTC', 'BTC'],
                'feature1': [0.1, 0.2, 0.3]
            })
        }
        
        yiedl_data = {
            'latest': pd.DataFrame({
                'date': ['2023-02-01', '2023-02-02', '2023-02-03'],
                'asset': ['BTC', 'BTC', 'BTC'],
                'price': [50000, 51000, 52000]
            })
        }
        
        # Create merged dataset
        merged_data = create_merged_dataset(numerai_data, yiedl_data)
        
        # Check that the merged dataset is empty (no date overlap)
        self.assertEqual(merged_data.shape[0], 0)

class TestDatasetReporting(unittest.TestCase):
    """Tests for dataset reporting functionality."""
    
    def setUp(self):
        """Set up test case with mock data."""
        # Create a temporary directory and generate full mock dataset
        self.temp_dir = tempfile.TemporaryDirectory()
        self.mock_data = generate_full_mock_dataset(self.temp_dir.name)
        
        # Load Numerai data
        from utils.data.load_numerai import load_numerai_data
        numerai_files = {
            'train_data': self.mock_data['numerai_data']['train_data'],
            'live_universe': self.mock_data['numerai_data']['live_universe'],
            'current_round': 123
        }
        self.numerai_data = load_numerai_data(numerai_files)
        
        # Load Yiedl data
        from utils.data.load_yiedl import load_yiedl_data
        yiedl_files = {
            'latest': self.mock_data['yiedl_data']['latest']
        }
        self.yiedl_data = load_yiedl_data(yiedl_files)
        
        # Create merged dataset
        from utils.data.create_merged_dataset import create_merged_dataset
        self.merged_data = create_merged_dataset(self.numerai_data, self.yiedl_data)
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def test_report_merge_summary(self):
        """Test generating a merge summary report."""
        from utils.data.report_merge_summary import report_merge_summary
        
        # Get overlapping symbols
        from utils.data.create_merged_dataset import get_overlapping_symbols
        overlapping = get_overlapping_symbols(self.numerai_data, self.yiedl_data)
        
        # Generate report
        report = report_merge_summary(
            self.numerai_data, 
            self.yiedl_data, 
            self.merged_data,
            overlapping
        )
        
        # Check that the report is a dict with expected keys
        self.assertIsInstance(report, dict)
        self.assertIn('num_numerai_symbols', report)
        self.assertIn('num_yiedl_symbols', report)
        self.assertIn('num_overlapping_symbols', report)
        self.assertIn('overlapping_symbols', report)
        self.assertIn('merged_dataset_shape', report)
        
        # Check that the report values match expectations
        self.assertEqual(report['num_overlapping_symbols'], len(overlapping))
        self.assertEqual(set(report['overlapping_symbols']), set(overlapping))
        self.assertEqual(report['merged_dataset_shape'], self.merged_data.shape)
    
    def test_save_merge_report(self):
        """Test saving a merge report to a file."""
        from utils.data.report_merge_summary import report_merge_summary, save_merge_report
        
        # Get overlapping symbols
        from utils.data.create_merged_dataset import get_overlapping_symbols
        overlapping = get_overlapping_symbols(self.numerai_data, self.yiedl_data)
        
        # Generate report
        report = report_merge_summary(
            self.numerai_data, 
            self.yiedl_data, 
            self.merged_data,
            overlapping
        )
        
        # Save report to a file
        report_file = os.path.join(self.temp_dir.name, 'merge_report.json')
        save_merge_report(report, report_file)
        
        # Check that the file exists
        self.assertTrue(os.path.exists(report_file))
        
        # Check that the file contains valid JSON
        import json
        with open(report_file, 'r') as f:
            loaded_report = json.load(f)
        
        # Check that the loaded report matches the original
        self.assertEqual(loaded_report['num_overlapping_symbols'], report['num_overlapping_symbols'])
        self.assertEqual(set(loaded_report['overlapping_symbols']), set(report['overlapping_symbols']))
        self.assertEqual(loaded_report['merged_dataset_shape'], report['merged_dataset_shape'])

if __name__ == '__main__':
    unittest.main()