#!/usr/bin/env python3
"""
Tests for the Feature Registry functionality.

This module tests the feature tracking system that keeps track of processed features
from Yiedl and Numerai datasets.
"""

import os
import sys
import unittest
import tempfile
import shutil
import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.feature.feature_registry import FeatureRegistry

class TestFeatureRegistry(unittest.TestCase):
    """Test cases for the FeatureRegistry class"""
    
    def setUp(self):
        """Set up test environment"""
        # Create a temporary directory for the registry
        self.temp_dir = tempfile.mkdtemp()
        self.registry = FeatureRegistry(self.temp_dir)
        
        # Create test data files
        self.create_test_files()
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def create_test_files(self):
        """Create test data files"""
        # Create directories
        self.data_dir = Path(self.temp_dir) / "data"
        self.yiedl_dir = self.data_dir / "yiedl"
        self.numerai_dir = self.data_dir / "numerai"
        
        self.yiedl_dir.mkdir(parents=True, exist_ok=True)
        self.numerai_dir.mkdir(parents=True, exist_ok=True)
        
        # Create test Yiedl file
        df_yiedl = pl.DataFrame({
            "symbol": ["BTC", "ETH", "SOL", "ADA"],
            "date": ["2025-01-01", "2025-01-01", "2025-01-01", "2025-01-01"],
            "price": [50000.0, 3000.0, 150.0, 1.2],
            "volume": [1000000, 500000, 200000, 100000],
            "market_cap": [1000000000, 500000000, 100000000, 50000000],
            "feature1": [1.1, 2.2, 3.3, 4.4],
            "feature2": [5.5, 6.6, 7.7, 8.8]
        })
        
        self.yiedl_file1 = str(self.yiedl_dir / "yiedl_latest_20250101.parquet")
        df_yiedl.write_parquet(self.yiedl_file1)
        
        # Create test Yiedl file with some new features
        df_yiedl2 = pl.DataFrame({
            "symbol": ["BTC", "ETH", "SOL", "ADA"],
            "date": ["2025-01-02", "2025-01-02", "2025-01-02", "2025-01-02"],
            "price": [51000.0, 3100.0, 155.0, 1.25],
            "volume": [1100000, 550000, 220000, 110000],
            "market_cap": [1100000000, 550000000, 110000000, 55000000],
            "feature1": [1.2, 2.3, 3.4, 4.5],
            "feature2": [5.6, 6.7, 7.8, 8.9],
            "new_feature1": [9.9, 10.10, 11.11, 12.12],
            "new_feature2": [13.13, 14.14, 15.15, 16.16]
        })
        
        self.yiedl_file2 = str(self.yiedl_dir / "yiedl_latest_20250102.parquet")
        df_yiedl2.write_parquet(self.yiedl_file2)
        
        # Create test Numerai file
        df_numerai = pl.DataFrame({
            "id": ["BTC_2025-01-01", "ETH_2025-01-01", "SOL_2025-01-01", "ADA_2025-01-01"],
            "target": [0.1, 0.2, 0.3, 0.4],
            "data_type": ["validation", "validation", "validation", "validation"],
            "num_feature1": [0.5, 0.6, 0.7, 0.8],
            "num_feature2": [0.9, 1.0, 1.1, 1.2]
        })
        
        self.numerai_file1 = str(self.numerai_dir / "train_targets_r100.parquet")
        df_numerai.write_parquet(self.numerai_file1)
        
        # Create test Numerai file with some new features
        df_numerai2 = pl.DataFrame({
            "id": ["BTC_2025-01-02", "ETH_2025-01-02", "SOL_2025-01-02", "ADA_2025-01-02"],
            "target": [0.15, 0.25, 0.35, 0.45],
            "data_type": ["validation", "validation", "validation", "validation"],
            "num_feature1": [0.55, 0.65, 0.75, 0.85],
            "num_feature2": [0.95, 1.05, 1.15, 1.25],
            "new_num_feature1": [1.3, 1.4, 1.5, 1.6],
            "new_num_feature2": [1.7, 1.8, 1.9, 2.0]
        })
        
        self.numerai_file2 = str(self.numerai_dir / "train_targets_r101.parquet")
        df_numerai2.write_parquet(self.numerai_file2)
    
    def test_register_file(self):
        """Test registering a file"""
        file_id = self.registry.register_file("yiedl", self.yiedl_file1)
        self.assertGreater(file_id, 0, "File registration should return a valid ID")
        
        # Register the same file again - should return the same ID
        file_id2 = self.registry.register_file("yiedl", self.yiedl_file1)
        self.assertEqual(file_id, file_id2, "Registering the same file should return the same ID")
    
    def test_register_features(self):
        """Test registering features from a file"""
        file_id = self.registry.register_file("yiedl", self.yiedl_file1)
        df = pl.read_parquet(self.yiedl_file1)
        
        feature_ids = self.registry.register_features("yiedl", file_id, df)
        self.assertEqual(len(feature_ids), len(df.columns), 
                         "All columns should be registered as features")
        
        # Register the same features again - should return the same IDs
        feature_ids2 = self.registry.register_features("yiedl", file_id, df)
        self.assertEqual(len(feature_ids), len(feature_ids2), 
                         "Registering the same features should return the same count")
    
    def test_get_new_features(self):
        """Test getting new features from a file"""
        # First register original file and its features
        df1 = pl.read_parquet(self.yiedl_file1)
        new_df1, common_cols1 = self.registry.get_new_features("yiedl", self.yiedl_file1, df1)
        
        # All features should be new the first time
        self.assertEqual(new_df1.width, df1.width, "All features should be new in the first file")
        
        # Now try with a file that has some new features
        df2 = pl.read_parquet(self.yiedl_file2)
        new_df2, common_cols2 = self.registry.get_new_features("yiedl", self.yiedl_file2, df2)
        
        # Only new features should be returned (plus common columns)
        expected_new_features = 2  # new_feature1, new_feature2
        expected_common_cols = ["symbol", "date"]  # Essential columns that are always kept
        
        self.assertEqual(new_df2.width, len(expected_common_cols) + expected_new_features,
                         "Only new features plus common columns should be returned")
        
        # Check that common columns are included
        for col in expected_common_cols:
            self.assertIn(col, new_df2.columns, f"Common column {col} should be included")
        
        # Check that new features are included
        self.assertIn("new_feature1", new_df2.columns, "New feature should be included")
        self.assertIn("new_feature2", new_df2.columns, "New feature should be included")
    
    def test_numerai_features(self):
        """Test tracking Numerai features"""
        # First register original file and its features
        df1 = pl.read_parquet(self.numerai_file1)
        new_df1, common_cols1 = self.registry.get_new_features("numerai", self.numerai_file1, df1)
        
        # All features should be new the first time
        self.assertEqual(new_df1.width, df1.width, "All features should be new in the first file")
        
        # Now try with a file that has some new features
        df2 = pl.read_parquet(self.numerai_file2)
        new_df2, common_cols2 = self.registry.get_new_features("numerai", self.numerai_file2, df2)
        
        # Only new features should be returned (plus common columns)
        expected_new_features = 2  # new_num_feature1, new_num_feature2
        expected_common_cols = ["id", "target", "data_type"]  # Essential columns that are always kept
        
        self.assertEqual(new_df2.width, len(expected_common_cols) + expected_new_features,
                         "Only new features plus common columns should be returned")
        
        # Check that common columns are included
        for col in expected_common_cols:
            self.assertIn(col, new_df2.columns, f"Common column {col} should be included")
        
        # Check that new features are included
        self.assertIn("new_num_feature1", new_df2.columns, "New feature should be included")
        self.assertIn("new_num_feature2", new_df2.columns, "New feature should be included")
    
    def test_get_feature_count(self):
        """Test getting feature counts"""
        # Register some features
        df_yiedl = pl.read_parquet(self.yiedl_file1)
        self.registry.get_new_features("yiedl", self.yiedl_file1, df_yiedl)
        
        df_numerai = pl.read_parquet(self.numerai_file1)
        self.registry.get_new_features("numerai", self.numerai_file1, df_numerai)
        
        # Get feature counts
        counts = self.registry.get_feature_count()
        
        # Should have counts for both sources
        self.assertIn("yiedl", counts, "Should have count for Yiedl features")
        self.assertIn("numerai", counts, "Should have count for Numerai features")
        
        # Counts should match the number of columns
        self.assertEqual(counts["yiedl"], len(df_yiedl.columns), 
                         "Yiedl feature count should match number of columns")
        self.assertEqual(counts["numerai"], len(df_numerai.columns), 
                         "Numerai feature count should match number of columns")
    
    def test_feature_history(self):
        """Test getting feature history"""
        # Register some features
        df_yiedl = pl.read_parquet(self.yiedl_file1)
        self.registry.get_new_features("yiedl", self.yiedl_file1, df_yiedl)
        
        # Get feature history
        history = self.registry.get_feature_history("yiedl")
        
        # Should have entries for all features
        self.assertEqual(len(history), len(df_yiedl.columns), 
                         "History should have entries for all features")
        
        # Check that first_seen and last_seen are set
        self.assertIsNotNone(history['first_seen'].iloc[0], "first_seen should be set")
        self.assertIsNotNone(history['last_seen'].iloc[0], "last_seen should be set")


if __name__ == "__main__":
    unittest.main()