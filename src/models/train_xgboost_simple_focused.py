"""
Simplified focused XGBoost optimization for clinical readmission prediction.

This script implements targeted optimizations for the readmission prediction task:
1. Class imbalance handling with scale_pos_weight
2. Feature engineering focused on clinically relevant predictors
3. Direct optimization for F1 score
4. Focused parameter tuning
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import pickle
import json
import mlflow
import mlflow.sklearn
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, make_scorer
)
import matplotlib.pyplot as plt
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

def select_clinical_features(X_train, X_val):
    """
    Select clinically relevant features based on domain knowledge.
    
    Args:
        X_train: Training features
        X_val: Validation features
        
    Returns:
        X_train_selected, X_val_selected, selected_features
    """
    logger.info("Selecting clinically relevant features")
    
    # Define groups of features that are clinically relevant for readmission
    demographic_features = ['race', 'gender', 'age']
    
    admission_features = [
        'admission_type_id', 'discharge_disposition_id', 
        'admission_source_id', 'time_in_hospital'
    ]
    
    medical_features = [
        'num_lab_procedures', 'num_procedures', 'num_medications',
        'number_outpatient', 'number_emergency', 'number_inpatient',
        'number_diagnoses', 'medical_specialty'
    ]
    
    # Identify medication features
    medication_features = [col for col in X_train.columns if col.startswith('medical_')]
    
    # Identify diagnosis features
    diagnosis_features = [col for col in X_train.columns if col.startswith('diag_')]
    
    # Combine all selected features
    selected_features = (
        demographic_features + 
        admission_features + 
        medical_features + 
        medication_features + 
        diagnosis_features
    )
    
    # Filter to only include features that exist in the dataset
    selected_features = [f for f in selected_features if f in X_train.columns]
    
    logger.info(f"Selected {len(selected_features)} clinically relevant features")
    
    # Return selected features
    return X_train[selected_features], X_val[selected_features], selected_features

def train_focused_xgboost(X_train, y_train, X_val, y_val):
    """
    Train a focused XGBoost model with targeted optimizations.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        
    Returns:
        best_model: The optimized XGBoost model
    """
    logger.info("Training focused XGBoost model")
    
    # Set up MLflow tracking
    mlflow.set_experiment("clinical-cdss-readmission-focused")
    
    # Calculate class imbalance ratio
    neg_pos_ratio = np.sum(y_train == 0) / np.sum(y_train == 1)
    logger.info(f"Class imbalance ratio (negative/positive): {neg_pos_ratio:.2f}")
    
    # Select clinically relevant features
    X_train, X_val, selected_features = select_clinical_features(X_train, X_val)
    
    # Define F1 scorer for optimization
    f1_scorer = make_scorer(f1_score)
    
    # Define focused parameter grid for key parameters
    param_grid = {
        'n_estimators': [500, 1000],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5, 6],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'scale_pos_weight': [1, neg_pos_ratio/2, neg_pos_ratio, neg_pos_ratio*1.5]
    }
    
    # Set up cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Define the XGBoost model with initial parameters
    xgb_model = XGBClassifier(
        objective='binary:logistic',
        random_state=42,
        tree_method='hist',
        booster='gbtree'
    )
    
    # Perform grid search with F1 optimization
    logger.info("Performing grid search with F1 optimization")
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring=f1_scorer,
        cv=cv,
        verbose=2,
        n_jobs=-1
    )
    
    # Fit the grid search
    grid_search.fit(X_train, y_train)
    
    # Get best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # Make predictions on validation set
    y_pred = best_model.predict(X_val)
    y_pred_proba = best_model.predict_proba(X_val)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred_proba)
    
    # Log results with MLflow
    with mlflow.start_run(run_name="xgboost_simple_focused"):
        # Log parameters
        mlflow.log_params({
            "model_type": "xgboost_simple_focused",
            "train_samples": X_train.shape[0],
            "features": X_train.shape[1],
            "selected_features": len(selected_features)
        })
        
        # Log best parameters
        for param, value in best_params.items():
            mlflow.log_param(param, value)
        
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
        mlflow.sklearn.log_model(best_model, "xgboost_simple_focused")
        
        # Generate feature importance plot
        plt.figure(figsize=(12, 8))
        plot_importance(best_model, max_num_features=20)
        plt.tight_layout()
        importance_path = MODELS_DIR / 'feature_importance_simple_focused.png'
        plt.savefig(importance_path)
        plt.close()
        
        # Log feature importance plot
        mlflow.log_artifact(str(importance_path))
        
        logger.info(f"XGBoost Simple Focused - AUC: {auc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    
    # Save results
    results_dict = {
        "xgboost_simple_focused": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auc": float(auc),
            "best_params": best_params
        }
    }
    
    with open(MODELS_DIR / 'simple_focused_results.json', 'w') as f:
        json.dump(results_dict, f, indent=4)
    
    # Save the model
    with open(MODELS_DIR / 'simple_focused_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    return best_model, selected_features

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
    sample_size = min(1000, X_train.shape[0])
    X_sample = X_train.sample(sample_size, random_state=42)

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X_sample)

    # If shap_values is a list (for multi-class), take the values for class 1
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Save SHAP values and feature names for later visualization
    with open(MODELS_DIR / 'simple_focused_shap_values.pkl', 'wb') as f:
        pickle.dump({
            'shap_values': shap_values,
            'X_sample': X_sample,
            'feature_names': feature_names
        }, f)

    # Create directory for SHAP visualizations
    shap_dir = MODELS_DIR / 'simple_focused_shap_explanations'
    shap_dir.mkdir(exist_ok=True)
    
    # Generate SHAP summary plot
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(shap_dir / 'shap_summary.png')
    plt.close()
    
    # Generate SHAP dependence plots for top features
    # Get mean absolute SHAP values for each feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-5:]  # Top 5 features
    
    for idx in top_indices:
        feature_name = feature_names[idx]
        plt.figure(figsize=(12, 8))
        shap.dependence_plot(idx, shap_values, X_sample, feature_names=feature_names, show=False)
        plt.title(f'SHAP Dependence Plot for {feature_name}')
        plt.tight_layout()
        plt.savefig(shap_dir / f'shap_dependence_{feature_name}.png')
        plt.close()
    
    logger.info(f"SHAP explanations generated and saved to {shap_dir}")
    return shap_values

def main():
    """Main function to execute the focused model training pipeline."""
    logger.info("Starting simplified focused XGBoost model training")

    # Load data
    X_train, y_train, X_val, y_val = load_data()

    # Train focused XGBoost model
    best_model, selected_features = train_focused_xgboost(X_train, y_train, X_val, y_val)

    # Generate explanations
    X_train_selected = X_train[selected_features]
    generate_explanations(best_model, X_train_selected, selected_features)

    # Save model metadata
    model_info = {
        "model_name": "xgboost_simple_focused",
        "feature_count": len(selected_features),
        "training_samples": X_train.shape[0],
        "validation_samples": X_val.shape[0],
        "feature_names": list(selected_features),
        "model_version": "3.0",
        "optimization_techniques": [
            "Clinically relevant feature selection",
            "Class imbalance handling with scale_pos_weight",
            "Direct F1 score optimization",
            "Focused parameter tuning"
        ]
    }
    
    with open(MODELS_DIR / 'simple_focused_model_info.json', 'w') as f:
        json.dump(model_info, f, indent=4)

    logger.info("Simplified focused model training completed successfully")

if __name__ == "__main__":
    main()
