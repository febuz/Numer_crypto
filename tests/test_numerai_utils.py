#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import unittest
import tempfile
import logging
from unittest.mock import patch, MagicMock

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import log utils
from utils.log_utils import setup_logging

# Set up logging to external directory - set create_file=False for unit tests
logger = setup_logging(name=__name__, level=logging.INFO, create_file=False)

# Import the mock data generator
from tests.mock_data_generator import generate_numerai_mock_data

class TestDownloadNumerai(unittest.TestCase):
    """Tests for Numerai data download functions."""
    
    @patch('utils.data.download_numerai.numerapi.NumerAPI')
    def test_download_with_api_keys(self, mock_numerapi):
        """Test downloading Numerai data with API keys."""
        from utils.data.download_numerai import download_numerai_crypto_data
        
        # Mock the NumerAPI client
        mock_client = MagicMock()
        mock_numerapi.return_value = mock_client
        mock_client.get_current_round.return_value = 123
        
        # Mock the download functions
        mock_client.download_dataset.return_value = "/tmp/train.parquet"
        mock_client.download_live_universe.return_value = "/tmp/live.parquet"
        
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            result = download_numerai_crypto_data(
                temp_dir, 
                api_key="test_key", 
                api_secret="test_secret"
            )
            
            # Check that NumerAPI was initialized with the provided API keys
            mock_numerapi.assert_called_once_with("test_key", "test_secret")
            
            # Check the result contains the expected keys
            self.assertIn('train_targets', result)
            self.assertIn('live_universe', result)
            self.assertIn('train_data', result)
            self.assertIn('current_round', result)
    
    @patch('utils.data.download_numerai.numerapi.NumerAPI')
    def test_download_without_api_keys(self, mock_numerapi):
        """Test downloading Numerai data without API keys."""
        from utils.data.download_numerai import download_numerai_crypto_data
        
        # Mock the NumerAPI client
        mock_client = MagicMock()
        mock_numerapi.return_value = mock_client
        mock_client.get_current_round.return_value = 123
        
        # Mock the download functions
        mock_client.download_dataset.return_value = "/tmp/train.parquet"
        mock_client.download_live_universe.return_value = "/tmp/live.parquet"
        
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            result = download_numerai_crypto_data(temp_dir)
            
            # Check that NumerAPI was initialized without API keys
            mock_numerapi.assert_called_once_with()
            
            # Check the result contains the expected keys
            self.assertIn('train_targets', result)
            self.assertIn('live_universe', result)
            self.assertIn('train_data', result)
            self.assertIn('current_round', result)
    
    @patch('utils.data.download_numerai.numerapi.NumerAPI')
    def test_download_error_handling(self, mock_numerapi):
        """Test error handling during Numerai data download."""
        from utils.data.download_numerai import download_numerai_crypto_data
        
        # Mock the NumerAPI client
        mock_client = MagicMock()
        mock_numerapi.return_value = mock_client
        
        # Make get_current_round raise an exception
        mock_client.get_current_round.side_effect = Exception("API error")
        
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(Exception):
                download_numerai_crypto_data(temp_dir)

class TestLoadNumerai(unittest.TestCase):
    """Tests for Numerai data loading functions."""
    
    def setUp(self):
        """Set up test case with mock data."""
        # Create a temporary directory for mock data
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Generate mock Numerai data
        self.mock_data = generate_numerai_mock_data(self.temp_dir.name)
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def test_load_numerai_data(self):
        """Test loading Numerai data from files."""
        from utils.data.load_numerai import load_numerai_data
        
        # Create a mock file dict
        numerai_files = {
            'train_data': self.mock_data['train_data'],
            'live_universe': self.mock_data['live_universe'],
            'current_round': 123
        }
        
        # Load the data
        result = load_numerai_data(numerai_files)
        
        # Check the result contains the expected keys
        self.assertIn('train_data', result)
        self.assertIn('live_universe', result)
        self.assertIn('current_round', result)
        
        # Check the data was loaded correctly
        self.assertEqual(result['current_round'], 123)
        
        # Check DataFrame shapes match the original mock data
        if 'train_df' in self.mock_data:
            self.assertEqual(result['train_data'].shape, self.mock_data['train_df'].shape)
        
        if 'universe_df' in self.mock_data:
            self.assertEqual(result['live_universe'].shape, self.mock_data['universe_df'].shape)
    
    def test_get_eligible_crypto_symbols_from_pre_existing(self):
        """Test getting eligible crypto symbols when already present in data."""
        from utils.data.load_numerai import get_eligible_crypto_symbols
        
        # Mock data with pre-existing symbols
        numerai_data = {'symbols': ['BTC', 'ETH', 'XRP']}
        
        symbols = get_eligible_crypto_symbols(numerai_data)
        self.assertEqual(symbols, ['BTC', 'ETH', 'XRP'])
    
    def test_get_eligible_crypto_symbols_from_live_universe(self):
        """Test getting eligible crypto symbols from live universe data."""
        from utils.data.load_numerai import get_eligible_crypto_symbols
        
        # Load mock data
        from utils.data.load_numerai import load_numerai_data
        
        numerai_files = {
            'train_data': self.mock_data['train_data'],
            'live_universe': self.mock_data['live_universe'],
            'current_round': 123
        }
        
        # Load the data and extract symbols
        numerai_data = load_numerai_data(numerai_files)
        symbols = get_eligible_crypto_symbols(numerai_data)
        
        # Check that symbols were found
        self.assertGreater(len(symbols), 0)
        
        # Check that the symbols are the same as in the mock data
        if 'symbols' in self.mock_data:
            self.assertEqual(set(symbols), set(self.mock_data['symbols']))
    
    def test_get_eligible_crypto_symbols_from_train_data(self):
        """Test getting eligible crypto symbols from train data."""
        from utils.data.load_numerai import get_eligible_crypto_symbols
        
        # Create mock data with only train_data (no live_universe)
        import pandas as pd
        numerai_data = {
            'train_data': self.mock_data['train_df'] if 'train_df' in self.mock_data else pd.DataFrame({
                'symbol': ['BTC', 'ETH', 'XRP', 'ADA']
            })
        }
        
        # Extract symbols
        symbols = get_eligible_crypto_symbols(numerai_data)
        
        # Check that symbols were found
        self.assertGreater(len(symbols), 0)
        
        # Check that the symbols are as expected
        if 'symbols' in self.mock_data:
            self.assertEqual(set(symbols), set(self.mock_data['symbols']))
    
    def test_get_eligible_crypto_symbols_no_data(self):
        """Test getting eligible crypto symbols with no data."""
        from utils.data.load_numerai import get_eligible_crypto_symbols
        
        # Mock empty data
        numerai_data = {}
        
        # Should return an empty list
        symbols = get_eligible_crypto_symbols(numerai_data)
        self.assertEqual(symbols, [])

class TestNumeraiDataVersioning(unittest.TestCase):
    """Tests for Numerai data versioning functionality."""
    
    @patch('utils.data.download_numerai.datetime')
    @patch('utils.data.download_numerai.numerapi.NumerAPI')
    def test_file_naming_with_timestamp(self, mock_numerapi, mock_datetime):
        """Test that downloaded files have timestamps in their names."""
        from utils.data.download_numerai import download_numerai_crypto_data
        
        # Mock datetime
        mock_datetime.now.return_value.strftime.return_value = '20250101_120000'
        
        # Mock the NumerAPI client
        mock_client = MagicMock()
        mock_numerapi.return_value = mock_client
        mock_client.get_current_round.return_value = 123
        
        # Mock download functions to return regular file paths
        mock_client.download_dataset.return_value = "/tmp/train.parquet"
        mock_client.download_live_universe.return_value = "/tmp/live.parquet"
        
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            result = download_numerai_crypto_data(temp_dir)
            
            # Check that the file names contain the timestamp and round number
            self.assertIn('20250101_120000', result['train_targets'])
            self.assertIn('20250101_120000', result['live_universe'])
            self.assertIn('20250101_120000', result['train_data'])
            self.assertIn('123', result['train_targets'])
            self.assertIn('123', result['live_universe'])
            self.assertIn('123', result['train_data'])

if __name__ == '__main__':
    unittest.main()