"""
Robust XGBoost optimization for clinical readmission prediction.

This script implements a robust approach to optimize XGBoost for readmission prediction:
1. Proper handling of categorical features
2. Class imbalance handling with scale_pos_weight
3. Direct optimization for F1 score
4. Simplified parameter tuning with error reporting
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
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
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

def select_and_prepare_features(X_train, X_val):
    """
    Select important features and prepare them for modeling.
    
    Args:
        X_train: Training features
        X_val: Validation features
        
    Returns:
        X_train_prepared, X_val_prepared, feature_names
    """
    logger.info("Selecting and preparing features")
    
    # Select a smaller set of important features based on domain knowledge
    # Focus on features that are known to be predictive of readmission
    important_features = [
        'time_in_hospital', 'num_lab_procedures', 'num_procedures',
        'num_medications', 'number_outpatient', 'number_emergency',
        'number_inpatient', 'number_diagnoses', 'age',
        'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
        'A1Cresult', 'metformin', 'glimepiride', 'glipizide', 'glyburide',
        'pioglitazone', 'rosiglitazone', 'insulin', 'change', 'diabetesMed'
    ]
    
    # Filter to only include features that exist in the dataset
    selected_features = [f for f in important_features if f in X_train.columns]
    
    # Add diagnosis codes if they exist
    diag_features = [col for col in X_train.columns if col.startswith('diag_')][:5]  # Limit to first 5
    selected_features.extend(diag_features)
    
    logger.info(f"Selected {len(selected_features)} important features")
    
    # Identify categorical features
    categorical_features = []
    for feature in selected_features:
        if X_train[feature].dtype == 'object' or X_train[feature].nunique() < 10:
            categorical_features.append(feature)
    
    # Identify numerical features
    numerical_features = [f for f in selected_features if f not in categorical_features]
    
    logger.info(f"Identified {len(categorical_features)} categorical and {len(numerical_features)} numerical features")
    
    # Create a simple preprocessing pipeline
    # For XGBoost, we don't need to scale numerical features
    # We'll just one-hot encode categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )
    
    # Fit and transform the training data
    X_train_prepared = preprocessor.fit_transform(X_train[selected_features])
    X_val_prepared = preprocessor.transform(X_val[selected_features])
    
    # Get feature names after preprocessing
    cat_feature_names = []
    if categorical_features:
        ohe = preprocessor.named_transformers_['cat']
        cat_feature_names = ohe.get_feature_names_out(categorical_features).tolist()
    
    feature_names = cat_feature_names + numerical_features
    
    logger.info(f"Prepared data with {X_train_prepared.shape[1]} features after preprocessing")
    
    return X_train_prepared, X_val_prepared, feature_names

def train_robust_xgboost(X_train, y_train, X_val, y_val, feature_names):
    """
    Train a robust XGBoost model with proper error handling.
    
    Args:
        X_train: Prepared training features
        y_train: Training target
        X_val: Prepared validation features
        y_val: Validation target
        feature_names: Names of features after preprocessing
        
    Returns:
        best_model: The optimized XGBoost model
    """
    logger.info("Training robust XGBoost model")
    
    # Set up MLflow tracking
    mlflow.set_experiment("clinical-cdss-readmission-robust")
    
    # Calculate class imbalance ratio
    neg_pos_ratio = np.sum(y_train == 0) / np.sum(y_train == 1)
    logger.info(f"Class imbalance ratio (negative/positive): {neg_pos_ratio:.2f}")
    
    # Define F1 scorer for optimization
    f1_scorer = make_scorer(f1_score)
    
    # Define a focused parameter grid with fewer combinations
    param_grid = {
        'n_estimators': [500, 1000],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5],
        'scale_pos_weight': [1, neg_pos_ratio]
    }
    
    # Set up cross-validation
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    # Define the XGBoost model with initial parameters
    xgb_model = XGBClassifier(
        objective='binary:logistic',
        random_state=42,
        tree_method='hist',
        booster='gbtree'
    )
    
    # Perform grid search with F1 optimization and error reporting
    logger.info("Performing grid search with F1 optimization")
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring=f1_scorer,
        cv=cv,
        verbose=2,
        n_jobs=-1,
        error_score='raise'  # Raise errors instead of setting to NaN
    )
    
    try:
        # Fit the grid search
        grid_search.fit(X_train, y_train)
        
        # Get best model and parameters
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        logger.info(f"Best parameters: {best_params}")
        
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
        with mlflow.start_run(run_name="xgboost_robust"):
            # Log parameters
            mlflow.log_params({
                "model_type": "xgboost_robust",
                "train_samples": X_train.shape[0],
                "features": X_train.shape[1]
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
            mlflow.sklearn.log_model(best_model, "xgboost_robust")
            
            # Try to generate feature importance plot
            try:
                plt.figure(figsize=(12, 8))
                plot_importance(best_model, max_num_features=20)
                plt.tight_layout()
                importance_path = MODELS_DIR / 'feature_importance_robust.png'
                plt.savefig(importance_path)
                plt.close()
                
                # Log feature importance plot
                mlflow.log_artifact(str(importance_path))
            except Exception as e:
                logger.warning(f"Could not generate feature importance plot: {e}")
            
            logger.info(f"XGBoost Robust - AUC: {auc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        
        # Save results
        results_dict = {
            "xgboost_robust": {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "auc": float(auc),
                "best_params": best_params
            }
        }
        
        with open(MODELS_DIR / 'robust_results.json', 'w') as f:
            json.dump(results_dict, f, indent=4)
        
        # Save the model
        with open(MODELS_DIR / 'robust_model.pkl', 'wb') as f:
            pickle.dump(best_model, f)
        
        return best_model, feature_names
    
    except Exception as e:
        logger.error(f"Error during grid search: {e}")
        
        # Try a simpler approach with a single model
        logger.info("Falling back to a single model with default parameters")
        
        # Train a single model with default parameters
        model = XGBClassifier(
            objective='binary:logistic',
            n_estimators=500,
            learning_rate=0.1,
            max_depth=3,
            scale_pos_weight=neg_pos_ratio,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_pred_proba)
        
        logger.info(f"Single model - AUC: {auc:.4f}, F1: {f1:.4f}")
        
        # Save the model
        with open(MODELS_DIR / 'robust_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        return model, feature_names

def generate_explanations(model, X_val_prepared, feature_names):
    """
    Generate model explanations using SHAP.

    Args:
        model: Trained model
        X_val_prepared: Prepared validation data
        feature_names: Names of features after preprocessing

    Returns:
        shap_values: SHAP values for explaining the model
    """
    logger.info("Generating model explanations with SHAP")

    try:
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_val_prepared)

        # If shap_values is a list (for multi-class), take the values for class 1
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # Save SHAP values and feature names for later visualization
        with open(MODELS_DIR / 'robust_shap_values.pkl', 'wb') as f:
            pickle.dump({
                'shap_values': shap_values,
                'feature_names': feature_names
            }, f)

        # Create directory for SHAP visualizations
        shap_dir = MODELS_DIR / 'robust_shap_explanations'
        shap_dir.mkdir(exist_ok=True)
        
        # Generate SHAP summary plot
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X_val_prepared, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(shap_dir / 'shap_summary.png')
        plt.close()
        
        logger.info(f"SHAP explanations generated and saved to {shap_dir}")
        return shap_values
    
    except Exception as e:
        logger.error(f"Error generating SHAP explanations: {e}")
        return None

def main():
    """Main function to execute the robust model training pipeline."""
    logger.info("Starting robust XGBoost model training")

    # Load data
    X_train, y_train, X_val, y_val = load_data()

    # Select and prepare features
    X_train_prepared, X_val_prepared, feature_names = select_and_prepare_features(X_train, X_val)

    # Train robust XGBoost model
    best_model, feature_names = train_robust_xgboost(X_train_prepared, y_train, X_val_prepared, y_val, feature_names)

    # Generate explanations
    generate_explanations(best_model, X_val_prepared, feature_names)

    # Save model metadata
    model_info = {
        "model_name": "xgboost_robust",
        "feature_count": len(feature_names),
        "training_samples": X_train.shape[0],
        "validation_samples": X_val.shape[0],
        "model_version": "4.0",
        "optimization_techniques": [
            "Proper feature preprocessing",
            "Class imbalance handling",
            "Direct F1 score optimization",
            "Robust error handling"
        ]
    }
    
    with open(MODELS_DIR / 'robust_model_info.json', 'w') as f:
        json.dump(model_info, f, indent=4)

    logger.info("Robust model training completed successfully")

if __name__ == "__main__":
    main()
