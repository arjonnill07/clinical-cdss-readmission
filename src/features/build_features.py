"""
Feature engineering script for the Clinical CDSS Readmission project.

This script:
1. Loads the processed data
2. Extracts features from structured EHR data
3. Extracts features from unstructured clinical notes using NLP
4. Combines all features and saves the feature matrices

Theory:
- Feature engineering transforms raw data into meaningful inputs for ML models
- Domain-specific features often improve model performance significantly
- NLP techniques can extract valuable information from unstructured clinical notes
- Feature selection helps reduce dimensionality and prevent overfitting
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import spacy
import re
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
PROCESSED_DATA_DIR = Path('data/processed')
FEATURES_DATA_DIR = Path('data/features')

# Ensure directories exist
FEATURES_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Load spaCy model for NLP
try:
    nlp = spacy.load('en_core_web_sm')
    logger.info("Loaded spaCy model: en_core_web_sm")
except OSError:
    logger.warning("spaCy model not found. Downloading en_core_web_sm...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load('en_core_web_sm')
    logger.info("Downloaded and loaded spaCy model: en_core_web_sm")

def load_data():
    """
    Load the processed data.
    
    Returns:
        train, val, test DataFrames
    """
    logger.info("Loading processed data")
    
    train = pd.read_csv(PROCESSED_DATA_DIR / 'train.csv')
    val = pd.read_csv(PROCESSED_DATA_DIR / 'val.csv')
    test = pd.read_csv(PROCESSED_DATA_DIR / 'test.csv')
    
    logger.info(f"Loaded train set with shape: {train.shape}")
    logger.info(f"Loaded validation set with shape: {val.shape}")
    logger.info(f"Loaded test set with shape: {test.shape}")
    
    return train, val, test

def extract_structured_features(df):
    """
    Extract features from structured EHR data.
    
    This includes:
    1. Demographic features (age, gender, etc.)
    2. Medical history features
    3. Lab test results
    4. Medication information
    5. Derived features (e.g., comorbidity indices)
    
    Args:
        df: DataFrame with processed data
        
    Returns:
        DataFrame with extracted features
    """
    logger.info("Extracting structured features")
    
    # Create a copy to avoid modifying the original DataFrame
    features_df = df.copy()
    
    # 1. Demographic features
    # Age groups (assuming 'age' column exists with categorical values)
    if 'age' in features_df.columns:
        # Convert age categories to numeric values
        # Example: '[0-10)' -> 5, '[10-20)' -> 15, etc.
        features_df['age_numeric'] = features_df['age'].apply(
            lambda x: int(re.findall(r'\d+', str(x))[0]) + 5 if isinstance(x, str) and re.findall(r'\d+', str(x)) else 50
        )
    
    # 2. Medical history features
    # Create comorbidity count (assuming diagnosis columns exist)
    diagnosis_cols = [col for col in features_df.columns if col.startswith('diag')]
    if diagnosis_cols:
        features_df['comorbidity_count'] = features_df[diagnosis_cols].apply(
            lambda x: x.astype(bool).sum(), axis=1
        )
    
    # 3. Create interaction features
    # Example: Interaction between age and number of medications
    if 'age_numeric' in features_df.columns and 'num_medications' in features_df.columns:
        features_df['age_med_interaction'] = features_df['age_numeric'] * features_df['num_medications']
    
    # 4. Create time-based features
    # Example: Length of stay
    if 'time_in_hospital' in features_df.columns:
        # Bin length of stay
        features_df['los_group'] = pd.cut(
            features_df['time_in_hospital'],
            bins=[0, 3, 7, 14, float('inf')],
            labels=['short', 'medium', 'long', 'very_long']
        ).astype(str)
        
        # One-hot encode the binned feature
        los_dummies = pd.get_dummies(features_df['los_group'], prefix='los')
        features_df = pd.concat([features_df, los_dummies], axis=1)
    
    # 5. Normalize numerical features
    num_cols = features_df.select_dtypes(include=['number']).columns
    scaler = StandardScaler()
    features_df[num_cols] = scaler.fit_transform(features_df[num_cols])
    
    # Save the scaler for later use
    with open(FEATURES_DATA_DIR / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    logger.info(f"Extracted structured features with shape: {features_df.shape}")
    return features_df

def extract_text_features(df, text_col='clinical_notes', max_features=100):
    """
    Extract features from unstructured clinical notes using NLP.
    
    This includes:
    1. TF-IDF vectorization
    2. Topic modeling (LSA)
    3. Named entity recognition
    4. Medical concept extraction
    
    Args:
        df: DataFrame with processed data
        text_col: Column name containing the clinical notes
        max_features: Maximum number of features to extract
        
    Returns:
        DataFrame with extracted text features
    """
    logger.info("Extracting text features")
    
    # Check if text column exists
    if text_col not in df.columns:
        logger.warning(f"Text column '{text_col}' not found. Skipping text feature extraction.")
        return pd.DataFrame(index=df.index)
    
    # Fill missing values
    df[text_col] = df[text_col].fillna('')
    
    # 1. TF-IDF Vectorization
    logger.info("Performing TF-IDF vectorization")
    tfidf = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        ngram_range=(1, 2)
    )
    tfidf_matrix = tfidf.fit_transform(df[text_col])
    
    # Save the vectorizer for later use
    with open(FEATURES_DATA_DIR / 'tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf, f)
    
    # 2. Dimensionality reduction with LSA (Topic Modeling)
    logger.info("Performing LSA for topic modeling")
    n_components = min(50, tfidf_matrix.shape[1] - 1)  # Ensure we don't exceed the number of features
    lsa = TruncatedSVD(n_components=n_components)
    lsa_features = lsa.fit_transform(tfidf_matrix)
    
    # Save the LSA model for later use
    with open(FEATURES_DATA_DIR / 'lsa_model.pkl', 'wb') as f:
        pickle.dump(lsa, f)
    
    # Create DataFrame with LSA features
    lsa_df = pd.DataFrame(
        lsa_features,
        index=df.index,
        columns=[f'topic_{i}' for i in range(n_components)]
    )
    
    # 3. Named Entity Recognition (simplified version)
    # In a real-world scenario, you would use a medical NER model
    logger.info("Extracting medical entities from text")
    
    # Function to extract medical entities
    def extract_medical_entities(text):
        if not text:
            return {}
        
        doc = nlp(text[:1000])  # Limit text length for processing speed
        entities = {}
        
        # Count entity types
        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_] += 1
            else:
                entities[ent.label_] = 1
        
        return entities
    
    # Apply NER to a sample of the data (for demonstration)
    sample_size = min(1000, len(df))
    entity_counts = df[text_col].head(sample_size).apply(extract_medical_entities)
    
    # Create features for the most common entity types
    entity_types = set()
    for entities in entity_counts:
        entity_types.update(entities.keys())
    
    # Initialize entity features with zeros
    entity_features = pd.DataFrame(0, index=df.index, columns=[f'ent_{ent}' for ent in entity_types])
    
    # Fill in entity counts for the sampled rows
    for i, (idx, entities) in enumerate(zip(entity_counts.index, entity_counts)):
        for ent_type, count in entities.items():
            entity_features.loc[idx, f'ent_{ent_type}'] = count
    
    # Combine all text features
    text_features = pd.concat([lsa_df, entity_features], axis=1)
    
    logger.info(f"Extracted text features with shape: {text_features.shape}")
    return text_features

def combine_features(structured_features, text_features, target_col='readmitted_30d'):
    """
    Combine structured and text features into a single feature matrix.
    
    Args:
        structured_features: DataFrame with structured features
        text_features: DataFrame with text features
        target_col: Name of the target column
        
    Returns:
        X: Combined feature matrix
        y: Target variable
    """
    logger.info("Combining features")
    
    # Extract target variable
    if target_col in structured_features.columns:
        y = structured_features[target_col].copy()
        structured_features = structured_features.drop(columns=[target_col])
    else:
        logger.warning(f"Target column '{target_col}' not found. Using dummy target.")
        y = pd.Series(0, index=structured_features.index)
    
    # Combine features
    X = pd.concat([structured_features, text_features], axis=1)
    
    # Remove any remaining non-numeric columns
    non_numeric_cols = X.select_dtypes(exclude=['number']).columns
    X = X.drop(columns=non_numeric_cols)
    
    logger.info(f"Combined feature matrix shape: {X.shape}")
    return X, y

def main():
    """Main function to execute the feature engineering pipeline."""
    logger.info("Starting feature engineering")
    
    # Load data
    train, val, test = load_data()
    
    # Extract structured features
    train_structured = extract_structured_features(train)
    val_structured = extract_structured_features(val)
    test_structured = extract_structured_features(test)
    
    # Extract text features (assuming 'clinical_notes' column exists)
    # In a real scenario, you would have this column or create it from multiple text fields
    text_col = 'clinical_notes'
    
    # If the column doesn't exist, create a dummy one for demonstration
    if text_col not in train.columns:
        logger.warning(f"Text column '{text_col}' not found. Creating dummy column for demonstration.")
        # In a real scenario, you would have actual clinical notes
        train[text_col] = "Patient has diabetes. Blood glucose levels are high."
        val[text_col] = "Patient has diabetes. Blood glucose levels are high."
        test[text_col] = "Patient has diabetes. Blood glucose levels are high."
    
    train_text = extract_text_features(train, text_col)
    val_text = extract_text_features(val, text_col)
    test_text = extract_text_features(test, text_col)
    
    # Combine features
    X_train, y_train = combine_features(train_structured, train_text)
    X_val, y_val = combine_features(val_structured, val_text)
    X_test, y_test = combine_features(test_structured, test_text)
    
    # Save feature matrices and target variables
    logger.info("Saving feature matrices")
    X_train.to_csv(FEATURES_DATA_DIR / 'X_train.csv', index=False)
    y_train.to_csv(FEATURES_DATA_DIR / 'y_train.csv', index=False)
    X_val.to_csv(FEATURES_DATA_DIR / 'X_val.csv', index=False)
    y_val.to_csv(FEATURES_DATA_DIR / 'y_val.csv', index=False)
    X_test.to_csv(FEATURES_DATA_DIR / 'X_test.csv', index=False)
    y_test.to_csv(FEATURES_DATA_DIR / 'y_test.csv', index=False)
    
    logger.info("Feature engineering completed successfully")

if __name__ == "__main__":
    main()
