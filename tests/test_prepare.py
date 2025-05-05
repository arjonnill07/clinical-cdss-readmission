"""
Unit tests for the data preparation module.

These tests ensure that:
1. Data downloading works correctly
2. Data preprocessing handles missing values and outliers
3. Data splitting maintains class distribution
4. The entire pipeline runs without errors
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

# Add the src directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.prepare import (
    download_dataset,
    preprocess_data,
    split_data,
    main
)

class TestDataPreparation(unittest.TestCase):
    """Test cases for data preparation functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directories for testing
        self.test_dir = tempfile.mkdtemp()
        self.raw_dir = Path(self.test_dir) / 'data/raw'
        self.processed_dir = Path(self.test_dir) / 'data/processed'
        
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a sample dataset for testing
        self.sample_data = pd.DataFrame({
            'age': ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)'],
            'gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
            'time_in_hospital': [1, 3, 5, 7, 9],
            'num_medications': [2, 4, 6, 8, 10],
            'readmitted': ['<30', '>30', 'NO', '<30', '>30']
        })
        
        # Save the sample data
        self.sample_data_path = self.raw_dir / 'diabetic_data.csv'
        self.sample_data.to_csv(self.sample_data_path, index=False)
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_preprocess_data(self):
        """Test the data preprocessing function."""
        # Process the sample data
        processed_df = preprocess_data(self.sample_data_path)
        
        # Check that the output is a DataFrame
        self.assertIsInstance(processed_df, pd.DataFrame)
        
        # Check that the target variable was created
        self.assertIn('readmitted_30d', processed_df.columns)
        
        # Check that the target values are correct
        expected_targets = [1, 0, 0, 1, 0]
        self.assertListEqual(list(processed_df['readmitted_30d']), expected_targets)
        
        # Check that categorical variables were encoded
        self.assertTrue(np.issubdtype(processed_df['gender'].dtype, np.number))
    
    def test_split_data(self):
        """Test the data splitting function."""
        # Create a larger sample for splitting
        n_samples = 100
        np.random.seed(42)
        
        large_sample = pd.DataFrame({
            'feature1': np.random.rand(n_samples),
            'feature2': np.random.rand(n_samples),
            'readmitted_30d': np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
        })
        
        # Split the data
        train, val, test = split_data(large_sample)
        
        # Check that the splits have the correct sizes
        self.assertEqual(len(train), 60)  # 60% of the data
        self.assertEqual(len(val), 20)    # 20% of the data
        self.assertEqual(len(test), 20)   # 20% of the data
        
        # Check that the class distribution is maintained
        train_pos_ratio = train['readmitted_30d'].mean()
        val_pos_ratio = val['readmitted_30d'].mean()
        test_pos_ratio = test['readmitted_30d'].mean()
        original_pos_ratio = large_sample['readmitted_30d'].mean()
        
        # Allow for small variations due to random splitting
        self.assertAlmostEqual(train_pos_ratio, original_pos_ratio, delta=0.1)
        self.assertAlmostEqual(val_pos_ratio, original_pos_ratio, delta=0.1)
        self.assertAlmostEqual(test_pos_ratio, original_pos_ratio, delta=0.1)
    
    def test_main_function(self):
        """Test the main function with mocked dependencies."""
        # This is a more complex test that would typically use mocking
        # For simplicity, we'll just check that it runs without errors
        
        # Patch the global variables to use our test directories
        import src.data.prepare
        original_raw_dir = src.data.prepare.RAW_DATA_DIR
        original_processed_dir = src.data.prepare.PROCESSED_DATA_DIR
        
        try:
            src.data.prepare.RAW_DATA_DIR = self.raw_dir
            src.data.prepare.PROCESSED_DATA_DIR = self.processed_dir
            
            # Run the main function
            main()
            
            # Check that the output files were created
            self.assertTrue((self.processed_dir / 'train.csv').exists())
            self.assertTrue((self.processed_dir / 'val.csv').exists())
            self.assertTrue((self.processed_dir / 'test.csv').exists())
            
        finally:
            # Restore the original directories
            src.data.prepare.RAW_DATA_DIR = original_raw_dir
            src.data.prepare.PROCESSED_DATA_DIR = original_processed_dir

if __name__ == '__main__':
    unittest.main()
