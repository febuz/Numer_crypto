#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import unittest
import logging
from unittest.mock import patch, MagicMock

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import log utils
from utils.log_utils import setup_logging

# Set up logging to external directory - set create_file=False for unit tests
logger = setup_logging(name=__name__, level=logging.INFO, create_file=False)

class TestNumeraiUtils(unittest.TestCase):
    """Tests for Numerai utility functions."""
    
    @patch('utils.data.download_numerai.numerapi.NumerAPI')
    def test_download_numerai_crypto_data(self, mock_numerapi):
        """Test downloading Numerai crypto data."""
        from utils.data.download_numerai import download_numerai_crypto_data
        
        # Mock the NumerAPI client
        mock_client = MagicMock()
        mock_numerapi.return_value = mock_client
        mock_client.get_current_round.return_value = 123
        
        # Mock download functions
        mock_client.download_dataset.return_value = "/tmp/test_train.parquet"
        mock_client.download_live_universe.return_value = "/tmp/test_universe.parquet"
        
        # Create a temporary directory
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            result = download_numerai_crypto_data(temp_dir)
            
            # Check that the function called the right methods
            mock_client.get_current_round.assert_called_once_with(tournament='crypto')
            self.assertEqual(mock_client.download_dataset.call_count, 2)
            self.assertEqual(mock_client.download_live_universe.call_count, 1)
            
            # Check the result contains expected keys
            self.assertIn('train_targets', result)
            self.assertIn('live_universe', result)
            self.assertIn('train_data', result)
            self.assertIn('current_round', result)
            self.assertEqual(result['current_round'], 123)

    def test_get_eligible_crypto_symbols(self):
        """Test extracting eligible crypto symbols from Numerai data."""
        from utils.data.load_numerai import get_eligible_crypto_symbols
        
        # Mock numerai data with symbols already available
        numerai_data = {'symbols': ['BTC', 'ETH', 'XRP']}
        symbols = get_eligible_crypto_symbols(numerai_data)
        self.assertEqual(symbols, ['BTC', 'ETH', 'XRP'])
        
        # Mock numerai data with live universe
        import pandas as pd
        numerai_data = {
            'live_universe': pd.DataFrame({
                'Symbol': ['BTC', 'ETH', 'XRP', 'LTC']
            })
        }
        symbols = get_eligible_crypto_symbols(numerai_data)
        self.assertEqual(set(symbols), {'BTC', 'ETH', 'XRP', 'LTC'})
        
        # Mock numerai data with train data
        numerai_data = {
            'train_data': pd.DataFrame({
                'symbol': ['BTC', 'ETH', 'XRP', 'LTC', 'ADA']
            })
        }
        symbols = get_eligible_crypto_symbols(numerai_data)
        self.assertEqual(set(symbols), {'BTC', 'ETH', 'XRP', 'LTC', 'ADA'})

class TestYiedlUtils(unittest.TestCase):
    """Tests for Yiedl utility functions."""
    
    @patch('utils.data.download_yiedl.requests.get')
    def test_download_yiedl_data(self, mock_get):
        """Test downloading Yiedl data."""
        from utils.data.download_yiedl import download_yiedl_data
        
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        
        # Mock binary content (simple zip file with a single file)
        import io
        import zipfile
        
        # Create a mock zip file with a CSV file inside
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as mock_zip:
            mock_zip.writestr('test_data.csv', 'date,asset,price\n2023-01-01,BTC,50000\n')
        
        mock_response.content = zip_buffer.getvalue()
        mock_get.return_value = mock_response
        
        # Create a temporary directory
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            result = download_yiedl_data(temp_dir)
            
            # Check that the function called requests.get with the right URLs
            self.assertEqual(mock_get.call_count, 2)
            
            # Check the result contains expected keys
            self.assertIn('latest', result)
            self.assertIn('historical', result)

    def test_load_yiedl_data(self):
        """Test loading Yiedl data."""
        from utils.data.load_yiedl import load_yiedl_data, get_yiedl_crypto_symbols
        
        # Create mock Yiedl data
        import pandas as pd
        mock_data = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02'],
            'asset': ['BTC', 'ETH', 'BTC', 'ETH'],
            'price': [50000, 3000, 51000, 3100]
        })
        
        # Create a temporary directory with mock data
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_file = os.path.join(temp_dir, 'latest.csv')
            mock_data.to_csv(mock_file, index=False)
            
            # Mock yiedl files
            yiedl_files = {'latest': mock_file}
            
            # Test loading the data
            result = load_yiedl_data(yiedl_files)
            self.assertIn('latest', result)
            
            # Test getting symbols
            symbols = get_yiedl_crypto_symbols(result)
            self.assertEqual(set(symbols), {'BTC', 'ETH'})

class TestMergedDataset(unittest.TestCase):
    """Tests for merged dataset creation."""
    
    def test_get_overlapping_symbols(self):
        """Test finding overlapping symbols between Numerai and Yiedl data."""
        from utils.data.create_merged_dataset import get_overlapping_symbols
        
        # Create mock data
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
        self.assertEqual(set(overlapping), {'BTC', 'ETH'})
        
        # Test with no overlap
        yiedl_data = {
            'latest': pd.DataFrame({
                'date': ['2023-01-01'],
                'asset': ['SOL'],
                'price': [200]
            })
        }
        overlapping = get_overlapping_symbols(numerai_data, yiedl_data)
        self.assertEqual(overlapping, [])

    def test_create_merged_dataset(self):
        """Test creating merged dataset from Numerai and Yiedl data."""
        from utils.data.create_merged_dataset import create_merged_dataset
        
        # Create mock data
        import pandas as pd
        
        # Mock Numerai data with training targets
        numerai_data = {
            'train_data': pd.DataFrame({
                'date': ['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02'],
                'symbol': ['BTC', 'ETH', 'BTC', 'ETH'],
                'feature1': [0.1, 0.2, 0.3, 0.4],
                'feature2': [0.5, 0.6, 0.7, 0.8],
                'target': [1, 0, 1, 0]
            })
        }
        
        # Mock Yiedl data
        yiedl_data = {
            'latest': pd.DataFrame({
                'date': ['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02'],
                'asset': ['BTC', 'ETH', 'BTC', 'ETH'],
                'price': [50000, 3000, 51000, 3100],
                'volume': [100, 200, 150, 250]
            })
        }
        
        # Test creating the merged dataset
        merged_data = create_merged_dataset(numerai_data, yiedl_data)
        
        # Check the merged dataset contains the right columns
        self.assertIn('date', merged_data.columns)
        self.assertIn('symbol', merged_data.columns)
        self.assertIn('feature1', merged_data.columns)
        self.assertIn('feature2', merged_data.columns)
        self.assertIn('target', merged_data.columns)
        self.assertIn('price', merged_data.columns)
        self.assertIn('volume', merged_data.columns)
        
        # Check the shape is correct (4 rows, merged correctly)
        self.assertEqual(merged_data.shape[0], 4)

class TestDataRetriever(unittest.TestCase):
    """Tests for the NumeraiDataRetriever class."""
    
    def test_data_retriever_initialization(self):
        """Test initializing the NumeraiDataRetriever class."""
        from data.retrieval import NumeraiDataRetriever
        
        # Mock the required utility functions
        with patch('data.retrieval.download_numerai_crypto_data') as mock_download_numerai, \
             patch('data.retrieval.download_yiedl_data') as mock_download_yiedl, \
             patch('data.retrieval.load_numerai_data') as mock_load_numerai, \
             patch('data.retrieval.load_yiedl_data') as mock_load_yiedl:
            
            # Mock return values
            mock_download_numerai.return_value = {'train_data': '/tmp/train.parquet'}
            mock_download_yiedl.return_value = {'latest': '/tmp/latest.csv'}
            mock_load_numerai.return_value = {'train_data': MagicMock()}
            mock_load_yiedl.return_value = {'latest': MagicMock()}
            
            # Initialize the data retriever
            retriever = NumeraiDataRetriever()
            
            # Check the data retriever was initialized correctly
            self.assertIsNotNone(retriever.numerai_data)
            self.assertIsNotNone(retriever.yiedl_data)
            
            # Check the download functions were called
            mock_download_numerai.assert_called_once()
            mock_download_yiedl.assert_called_once()
            mock_load_numerai.assert_called_once()
            mock_load_yiedl.assert_called_once()
    
    def test_get_eligible_symbols(self):
        """Test getting eligible symbols from the data retriever."""
        from data.retrieval import NumeraiDataRetriever
        
        # Mock the data retriever and utility functions
        with patch('data.retrieval.get_eligible_crypto_symbols') as mock_get_numerai_symbols, \
             patch('data.retrieval.get_yiedl_crypto_symbols') as mock_get_yiedl_symbols, \
             patch('data.retrieval.get_overlapping_symbols') as mock_get_overlapping_symbols, \
             patch('data.retrieval.download_numerai_crypto_data'), \
             patch('data.retrieval.download_yiedl_data'), \
             patch('data.retrieval.load_numerai_data'), \
             patch('data.retrieval.load_yiedl_data'):
            
            # Mock return values
            mock_get_numerai_symbols.return_value = ['BTC', 'ETH', 'XRP']
            mock_get_yiedl_symbols.return_value = ['BTC', 'ETH', 'SOL']
            mock_get_overlapping_symbols.return_value = ['BTC', 'ETH']
            
            # Initialize the data retriever
            retriever = NumeraiDataRetriever()
            
            # Get eligible symbols
            symbols = retriever.get_eligible_symbols()
            
            # Check the symbols
            self.assertEqual(set(symbols), {'BTC', 'ETH'})
            
            # Check the utility functions were called with the right arguments
            mock_get_numerai_symbols.assert_called_once_with(retriever.numerai_data)
            mock_get_yiedl_symbols.assert_called_once_with(retriever.yiedl_data)
            mock_get_overlapping_symbols.assert_called_once()

if __name__ == '__main__':
    unittest.main()