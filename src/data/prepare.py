"""
Data preparation script for the Clinical CDSS Readmission project.

This script:
1. Downloads the dataset from a public source (if not already downloaded)
2. Performs initial cleaning and preprocessing
3. Splits the data into train/validation/test sets
4. Saves the processed data

Theory:
- Data preprocessing is crucial for ML model performance
- Proper train/test splitting prevents data leakage
- Stratified splitting preserves the class distribution in imbalanced datasets
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
from pathlib import Path
import requests
import zipfile
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
RAW_DATA_DIR = Path('data/raw')
PROCESSED_DATA_DIR = Path('data/processed')

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

def download_dataset():
    """
    Download the diabetic readmission dataset if not already present.
    
    We're using the UCI Diabetes 130-US hospitals dataset as an example.
    In a real-world scenario, you would connect to your secure data source.
    """
    dataset_path = RAW_DATA_DIR / 'diabetic_data.csv'
    
    if dataset_path.exists():
        logger.info(f"Dataset already exists at {dataset_path}")
        return dataset_path
    
    # URL for the UCI Diabetes dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip"
    
    try:
        logger.info(f"Downloading dataset from {url}")
        response = requests.get(url)
        response.raise_for_status()
        
        # Extract the zip file
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(RAW_DATA_DIR)
        
        logger.info(f"Dataset downloaded and extracted to {RAW_DATA_DIR}")
        return dataset_path
    
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        raise

def preprocess_data(data_path):
    """
    Preprocess the raw data.
    
    Steps:
    1. Load the data
    2. Clean missing values
    3. Encode categorical variables
    4. Handle outliers
    5. Create derived features
    
    Args:
        data_path: Path to the raw data file
        
    Returns:
        Preprocessed DataFrame
    """
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    logger.info(f"Raw data shape: {df.shape}")
    
    # Basic cleaning
    logger.info("Performing basic cleaning")
    
    # Replace missing values with appropriate placeholders
    df.replace('?', np.nan, inplace=True)
    
    # Drop columns with too many missing values (threshold can be adjusted)
    missing_threshold = 0.5
    cols_to_drop = [col for col in df.columns if df[col].isnull().mean() > missing_threshold]
    df.drop(columns=cols_to_drop, inplace=True)
    
    # Handle remaining missing values
    # For categorical columns, fill with the most frequent value
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    # For numerical columns, fill with the median
    num_cols = df.select_dtypes(include=['number']).columns
    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)
    
    # Encode categorical variables
    logger.info("Encoding categorical variables")
    for col in cat_cols:
        df[col] = pd.Categorical(df[col]).codes
    
    # Create target variable (readmitted within 30 days)
    logger.info("Creating target variable")
    if 'readmitted' in df.columns:
        # Assuming 'readmitted' has values like '<30', '>30', 'NO'
        df['readmitted_30d'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)
    
    logger.info(f"Preprocessed data shape: {df.shape}")
    return df

def split_data(df, target_col='readmitted_30d'):
    """
    Split the data into train, validation, and test sets.
    
    Args:
        df: Preprocessed DataFrame
        target_col: Name of the target column
        
    Returns:
        train, validation, and test DataFrames
    """
    logger.info("Splitting data into train, validation, and test sets")
    
    # First split: 80% train+val, 20% test
    train_val, test = train_test_split(
        df, 
        test_size=0.2, 
        stratify=df[target_col],
        random_state=42
    )
    
    # Second split: 75% train, 25% validation (from the 80% train+val)
    train, val = train_test_split(
        train_val, 
        test_size=0.25, 
        stratify=train_val[target_col],
        random_state=42
    )
    
    logger.info(f"Train set shape: {train.shape}")
    logger.info(f"Validation set shape: {val.shape}")
    logger.info(f"Test set shape: {test.shape}")
    
    return train, val, test

def main():
    """Main function to execute the data preparation pipeline."""
    logger.info("Starting data preparation")
    
    # Download dataset
    data_path = download_dataset()
    
    # Preprocess data
    df = preprocess_data(data_path)
    
    # Split data
    train, val, test = split_data(df)
    
    # Save processed datasets
    logger.info("Saving processed datasets")
    train.to_csv(PROCESSED_DATA_DIR / 'train.csv', index=False)
    val.to_csv(PROCESSED_DATA_DIR / 'val.csv', index=False)
    test.to_csv(PROCESSED_DATA_DIR / 'test.csv', index=False)
    
    logger.info("Data preparation completed successfully")

if __name__ == "__main__":
    main()
