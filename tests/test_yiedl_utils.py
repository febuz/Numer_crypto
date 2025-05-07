#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import unittest
import tempfile
import logging
import io
import zipfile
from unittest.mock import patch, MagicMock

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import log utils
from utils.log_utils import setup_logging

# Set up logging to external directory - set create_file=False for unit tests
logger = setup_logging(name=__name__, level=logging.INFO, create_file=False)

# Import the mock data generator
from tests.mock_data_generator import generate_yiedl_mock_data

class TestDownloadYiedl(unittest.TestCase):
    """Tests for Yiedl data download functions."""
    
    @patch('utils.data.download_yiedl.requests.get')
    def test_download_latest_data(self, mock_get):
        """Test downloading latest Yiedl data."""
        from utils.data.download_yiedl import download_yiedl_data
        
        # Mock response for latest data
        mock_latest_response = MagicMock()
        mock_latest_response.status_code = 200
        
        # Create a mock zip file with a CSV file inside for latest data
        latest_zip_buffer = io.BytesIO()
        with zipfile.ZipFile(latest_zip_buffer, 'w') as mock_zip:
            mock_zip.writestr('latest_data.csv', 'date,asset,price\n2023-01-01,BTC,50000\n')
        
        mock_latest_response.content = latest_zip_buffer.getvalue()
        
        # Mock response for historical data
        mock_historical_response = MagicMock()
        mock_historical_response.status_code = 200
        
        # Create a mock zip file with a CSV file inside for historical data
        historical_zip_buffer = io.BytesIO()
        with zipfile.ZipFile(historical_zip_buffer, 'w') as mock_zip:
            mock_zip.writestr('historical_data.csv', 'date,asset,price\n2022-01-01,BTC,40000\n')
        
        mock_historical_response.content = historical_zip_buffer.getvalue()
        
        # Set up the mock to return different responses for different URLs
        def mock_get_response(url, *args, **kwargs):
            if 'type=latest' in url:
                return mock_latest_response
            elif 'type=historical' in url:
                return mock_historical_response
            return MagicMock(status_code=404)
        
        mock_get.side_effect = mock_get_response
        
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            result = download_yiedl_data(temp_dir)
            
            # Check that the function called requests.get with the right URLs
            self.assertEqual(mock_get.call_count, 2)
            
            # Check the calls were made with the expected URLs
            calls = mock_get.call_args_list
            self.assertIn('type=latest', calls[0][0][0])
            self.assertIn('type=historical', calls[1][0][0])
            
            # Check the result contains the expected keys
            self.assertIn('latest', result)
            self.assertIn('historical', result)
            
            # Check the files exist
            self.assertTrue(os.path.exists(result['latest']))
            self.assertTrue(os.path.exists(result['historical']))
    
    @patch('utils.data.download_yiedl.requests.get')
    def test_download_error_handling(self, mock_get):
        """Test error handling during Yiedl data download."""
        from utils.data.download_yiedl import download_yiedl_data
        
        # Mock response for latest data
        mock_response = MagicMock()
        mock_response.status_code = 404  # Simulate a 404 error
        mock_get.return_value = mock_response
        
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(Exception):
                download_yiedl_data(temp_dir)
    
    @patch('utils.data.download_yiedl.requests.get')
    def test_download_invalid_zip(self, mock_get):
        """Test handling of invalid zip file."""
        from utils.data.download_yiedl import download_yiedl_data
        
        # Mock response with invalid zip content
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'Not a valid zip file'  # Invalid zip content
        mock_get.return_value = mock_response
        
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(Exception):
                download_yiedl_data(temp_dir)

class TestLoadYiedl(unittest.TestCase):
    """Tests for Yiedl data loading functions."""
    
    def setUp(self):
        """Set up test case with mock data."""
        # Create a temporary directory for mock data
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Generate mock Yiedl data
        self.mock_data = generate_yiedl_mock_data(self.temp_dir.name)
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def test_load_yiedl_data(self):
        """Test loading Yiedl data from files."""
        from utils.data.load_yiedl import load_yiedl_data
        
        # Create a mock file dict
        yiedl_files = {
            'latest': self.mock_data['latest']
        }
        
        # Load the data
        result = load_yiedl_data(yiedl_files)
        
        # Check the result contains the expected keys
        self.assertIn('latest', result)
        
        # Check DataFrame was loaded correctly
        self.assertTrue(result['latest'].shape[0] > 0)
        
        # Check the columns exist
        required_columns = ['date', 'asset', 'price']
        for col in required_columns:
            self.assertIn(col, result['latest'].columns)
    
    def test_load_yiedl_data_with_missing_file(self):
        """Test loading Yiedl data with missing file."""
        from utils.data.load_yiedl import load_yiedl_data
        
        # Create a mock file dict with non-existent file
        yiedl_files = {
            'latest': '/non/existent/file.csv'
        }
        
        # Should handle missing files gracefully
        result = load_yiedl_data(yiedl_files)
        
        # Result should still be a dict, but with empty dataframes
        self.assertIsInstance(result, dict)
    
    def test_get_yiedl_crypto_symbols(self):
        """Test extracting crypto symbols from Yiedl data."""
        from utils.data.load_yiedl import get_yiedl_crypto_symbols
        
        # Load the data first
        from utils.data.load_yiedl import load_yiedl_data
        
        yiedl_files = {
            'latest': self.mock_data['latest']
        }
        
        # Load the data and extract symbols
        yiedl_data = load_yiedl_data(yiedl_files)
        symbols = get_yiedl_crypto_symbols(yiedl_data)
        
        # Check that symbols were found
        self.assertGreater(len(symbols), 0)
        
        # Check that the symbols are the same as in the mock data
        if 'symbols' in self.mock_data:
            self.assertEqual(set(symbols), set(self.mock_data['symbols']))
    
    def test_get_yiedl_crypto_symbols_with_empty_data(self):
        """Test extracting crypto symbols from empty Yiedl data."""
        from utils.data.load_yiedl import get_yiedl_crypto_symbols
        
        # Mock empty data
        yiedl_data = {}
        
        # Should return an empty list
        symbols = get_yiedl_crypto_symbols(yiedl_data)
        self.assertEqual(symbols, [])
    
    def test_get_yiedl_crypto_symbols_with_alternative_column(self):
        """Test extracting crypto symbols when 'asset' column has alternative name."""
        from utils.data.load_yiedl import get_yiedl_crypto_symbols
        
        # Mock data with alternative column name
        import pandas as pd
        yiedl_data = {
            'latest': pd.DataFrame({
                'date': ['2023-01-01', '2023-01-01', '2023-01-01'],
                'symbol': ['BTC', 'ETH', 'XRP'],  # 'symbol' instead of 'asset'
                'price': [50000, 3000, 1]
            })
        }
        
        # Extract symbols
        symbols = get_yiedl_crypto_symbols(yiedl_data)
        
        # Check that symbols were found using alternative column
        self.assertEqual(set(symbols), {'BTC', 'ETH', 'XRP'})

class TestYiedlDataPolarsSupport(unittest.TestCase):
    """Tests for Yiedl data loading with polars support."""
    
    def setUp(self):
        """Set up test case with mock data."""
        # Create a temporary directory for mock data
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Generate mock Yiedl data
        self.mock_data = generate_yiedl_mock_data(self.temp_dir.name)
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    @patch('utils.data.load_yiedl.import_module')
    def test_load_with_polars(self, mock_import_module):
        """Test loading Yiedl data with polars."""
        # Mock successful polars import
        mock_polars = MagicMock()
        mock_import_module.return_value = mock_polars
        
        # Mock read_csv method
        mock_polars.read_csv.return_value = MagicMock()
        
        # Import module to test
        from utils.data.load_yiedl import load_yiedl_data_with_polars
        
        # Create a mock file dict
        yiedl_files = {
            'latest': self.mock_data['latest']
        }
        
        # Load the data
        result = load_yiedl_data_with_polars(yiedl_files)
        
        # Check that polars.read_csv was called
        mock_polars.read_csv.assert_called_once()
        
        # Check the result should contain the expected keys
        self.assertIn('latest', result)
    
    @patch('utils.data.load_yiedl.import_module')
    def test_load_with_polars_import_error(self, mock_import_module):
        """Test handling of ImportError for polars."""
        # Mock failed polars import
        mock_import_module.side_effect = ImportError("No module named 'polars'")
        
        # Import module to test
        from utils.data.load_yiedl import load_yiedl_data_with_polars
        
        # Create a mock file dict
        yiedl_files = {
            'latest': self.mock_data['latest']
        }
        
        # Should fallback to pandas
        with patch('utils.data.load_yiedl.load_yiedl_data') as mock_load_with_pandas:
            mock_load_with_pandas.return_value = {'latest': MagicMock()}
            result = load_yiedl_data_with_polars(yiedl_files)
            
            # Check that pandas fallback was called
            mock_load_with_pandas.assert_called_once()
            
            # Check the result should contain the expected keys
            self.assertIn('latest', result)

if __name__ == '__main__':
    unittest.main()