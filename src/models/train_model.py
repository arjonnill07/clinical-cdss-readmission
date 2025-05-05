"""
Model training script for the Clinical CDSS Readmission project.

This script:
1. Loads the feature matrices
2. Trains multiple models (traditional ML and deep learning)
3. Performs hyperparameter tuning
4. Tracks experiments with MLflow
5. Saves the best model

Theory:
- Model selection is crucial for healthcare applications
- Ensemble methods often perform well for tabular clinical data
- Hyperparameter tuning improves model performance
- Explainability is essential for clinical decision support
- MLflow helps track experiments and reproduce results
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import pickle
import json
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import shap

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
FEATURES_DATA_DIR = Path('data/features')
MODELS_DIR = Path('models')

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """
    Load the feature matrices.

    Returns:
        X_train, y_train, X_val, y_val
    """
    logger.info("Loading feature matrices")

    X_train = pd.read_csv(FEATURES_DATA_DIR / 'X_train.csv')
    y_train = pd.read_csv(FEATURES_DATA_DIR / 'y_train.csv').iloc[:, 0]
    X_val = pd.read_csv(FEATURES_DATA_DIR / 'X_val.csv')
    y_val = pd.read_csv(FEATURES_DATA_DIR / 'y_val.csv').iloc[:, 0]

    logger.info(f"Loaded training set with shape: {X_train.shape}")
    logger.info(f"Loaded validation set with shape: {X_val.shape}")

    return X_train, y_train, X_val, y_val

def train_models(X_train, y_train, X_val, y_val):
    """
    Train multiple models and select the best one.

    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target

    Returns:
        best_model: The best performing model
        best_model_name: Name of the best model
    """
    logger.info("Training models")

    # Set up MLflow tracking
    mlflow.set_experiment("clinical-cdss-readmission")

    # Define models to train
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "random_forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "xgboost": XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    # Define smaller hyperparameter grids for faster tuning
    param_grids = {
        "logistic_regression": {
            'C': [0.1, 1.0],
            'solver': ['liblinear']
        },
        "random_forest": {
            'n_estimators': [100],
            'max_depth': [10],
            'min_samples_split': [5]
        },
        "gradient_boosting": {
            'n_estimators': [100],
            'learning_rate': [0.1],
            'max_depth': [5]
        },
        "xgboost": {
            'n_estimators': [100],
            'learning_rate': [0.1],
            'max_depth': [5],
            'subsample': [0.8]
        }
    }

    # Train and evaluate each model
    best_auc = 0
    best_model = None
    best_model_name = None
    results = {}

    for model_name, model in models.items():
        logger.info(f"Training {model_name}")

        with mlflow.start_run(run_name=model_name):
            # Log model parameters
            mlflow.log_params({
                "model_type": model_name,
                "train_samples": X_train.shape[0],
                "features": X_train.shape[1]
            })

            # Perform hyperparameter tuning
            logger.info(f"Performing hyperparameter tuning for {model_name}")
            grid_search = GridSearchCV(
                model,
                param_grids[model_name],
                cv=5,
                scoring='roc_auc',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)

            # Get the best model
            best_params = grid_search.best_params_
            model = grid_search.best_estimator_

            # Log best parameters
            mlflow.log_params(best_params)

            # Make predictions on validation set
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]

            # Calculate metrics
            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred)
            recall = recall_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred)
            auc = roc_auc_score(y_val, y_pred_proba)

            # Log metrics
            mlflow.log_metrics({
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "auc": auc
            })

            # Generate and log confusion matrix
            cm = confusion_matrix(y_val, y_pred)
            cm_dict = {
                "true_negatives": cm[0, 0],
                "false_positives": cm[0, 1],
                "false_negatives": cm[1, 0],
                "true_positives": cm[1, 1]
            }
            mlflow.log_dict(cm_dict, "confusion_matrix.json")

            # Log the model
            mlflow.sklearn.log_model(model, model_name)

            # Store results
            results[model_name] = {
                "model": model,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "auc": auc,
                "best_params": best_params
            }

            logger.info(f"{model_name} - AUC: {auc:.4f}, F1: {f1:.4f}")

            # Update best model if this one is better
            if auc > best_auc:
                best_auc = auc
                best_model = model
                best_model_name = model_name

    logger.info(f"Best model: {best_model_name} with AUC: {best_auc:.4f}")

    # Save results
    results_dict = {
        name: {
            k: v for k, v in info.items() if k != "model"
        } for name, info in results.items()
    }
    with open(MODELS_DIR / 'training_results.json', 'w') as f:
        json.dump(results_dict, f, indent=4)

    return best_model, best_model_name

def generate_explanations(model, X_train, feature_names):
    """
    Generate model explanations using SHAP.

    Args:
        model: Trained model
        X_train: Training data
        feature_names: Names of the features

    Returns:
        shap_values: SHAP values for explaining the model
    """
    logger.info("Generating model explanations with SHAP")

    # Create a sample of the training data for SHAP analysis
    # (using full dataset can be computationally expensive)
    sample_size = min(1000, X_train.shape[0])
    X_sample = X_train.sample(sample_size, random_state=42)

    # Create SHAP explainer based on model type
    if hasattr(model, 'predict_proba'):
        # For tree-based models
        if hasattr(model, 'estimators_') or isinstance(model, XGBClassifier):
            explainer = shap.TreeExplainer(model)
        else:
            # For other models
            explainer = shap.KernelExplainer(
                model.predict_proba,
                shap.sample(X_sample, 100),
                link="logit"
            )
    else:
        logger.warning("Model doesn't support probability predictions. Using KernelExplainer.")
        explainer = shap.KernelExplainer(model.predict, shap.sample(X_sample, 100))

    # Calculate SHAP values
    shap_values = explainer.shap_values(X_sample)

    # If shap_values is a list (for multi-class), take the values for class 1
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Save SHAP values and feature names for later visualization
    with open(MODELS_DIR / 'shap_values.pkl', 'wb') as f:
        pickle.dump({
            'shap_values': shap_values,
            'X_sample': X_sample,
            'feature_names': feature_names
        }, f)

    logger.info("SHAP explanations generated and saved")
    return shap_values

def main():
    """Main function to execute the model training pipeline."""
    logger.info("Starting model training")

    # Load data
    X_train, y_train, X_val, y_val = load_data()

    # Train models and select the best one
    best_model, best_model_name = train_models(X_train, y_train, X_val, y_val)

    # Generate explanations
    generate_explanations(best_model, X_train, X_train.columns)

    # Save the best model
    logger.info(f"Saving the best model: {best_model_name}")
    with open(MODELS_DIR / 'model.pkl', 'wb') as f:
        pickle.dump(best_model, f)

    # Save model metadata
    model_info = {
        "model_name": best_model_name,
        "feature_count": X_train.shape[1],
        "training_samples": X_train.shape[0],
        "validation_samples": X_val.shape[0],
        "feature_names": list(X_train.columns)
    }
    with open(MODELS_DIR / 'model_info.json', 'w') as f:
        json.dump(model_info, f, indent=4)

    logger.info("Model training completed successfully")

if __name__ == "__main__":
    main()
