"""
Model training script for the Clinical CDSS Readmission project.

This script:
1. Loads the feature matrices
2. Trains an optimized XGBoost model for predicting 30-day readmission risk
3. Performs advanced hyperparameter tuning with RandomizedSearchCV
4. Uses early stopping to prevent overfitting
5. Implements StratifiedKFold for better cross-validation
6. Utilizes DMatrix for faster training
7. Tracks experiments with MLflow
8. Generates comprehensive model explanations with SHAP
9. Saves the best model

Theory:
- XGBoost performs well for tabular clinical data
- Advanced hyperparameter tuning improves model performance
- Early stopping prevents overfitting
- StratifiedKFold maintains class distribution in imbalanced datasets
- DMatrix provides faster training for XGBoost
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
from xgboost import XGBClassifier, DMatrix, plot_importance
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
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

def train_xgboost_model(X_train, y_train, X_val, y_val):
    """
    Train an optimized XGBoost model for predicting 30-day readmission risk.

    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target

    Returns:
        best_model: The optimized XGBoost model
    """
    logger.info("Training XGBoost model")

    # Set up MLflow tracking
    mlflow.set_experiment("clinical-cdss-readmission")

    # Convert data to DMatrix format for faster training
    dtrain = DMatrix(X_train, label=y_train)
    dval = DMatrix(X_val, label=y_val)

    # Create evaluation list for early stopping
    eval_list = [(dtrain, 'train'), (dval, 'validation')]

    # Define expanded hyperparameter search space for RandomizedSearchCV
    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6, 8, 10],
        'min_child_weight': [1, 3, 5, 7],
        'gamma': [0, 0.1, 0.2, 0.3, 0.4],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5, 1, 2],
        'reg_lambda': [0.1, 0.5, 1, 2, 5],
        'scale_pos_weight': [1, 3, 5]  # Helps with class imbalance
    }

    # Initialize base XGBoost model
    xgb_model = XGBClassifier(
        objective='binary:logistic',
        eval_metric=['logloss', 'auc', 'error'],
        use_label_encoder=False,
        random_state=42
    )

    # Set up StratifiedKFold for cross-validation to maintain class distribution
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Set up RandomizedSearchCV for more efficient hyperparameter tuning
    logger.info("Performing hyperparameter tuning with RandomizedSearchCV")
    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_dist,
        n_iter=20,  # Number of parameter settings sampled
        scoring='roc_auc',
        cv=cv,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    # Define a callable to pass to fit_params for early stopping
    fit_params = {
        "eval_set": [(X_val, y_val)],
        "eval_metric": "auc",
        "early_stopping_rounds": 20,
        "verbose": True
    }

    # Fit the model with fit_params
    random_search.fit(X_train, y_train, **fit_params)

    # Get the best model and parameters
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_

    # Log the training process with MLflow
    with mlflow.start_run(run_name="xgboost_optimized"):
        # Log model parameters
        mlflow.log_params({
            "model_type": "xgboost_optimized",
            "train_samples": X_train.shape[0],
            "features": X_train.shape[1]
        })

        # Log best parameters
        mlflow.log_params(best_params)

        # Make predictions on validation set
        y_pred = best_model.predict(X_val)
        y_pred_proba = best_model.predict_proba(X_val)[:, 1]

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
        mlflow.sklearn.log_model(best_model, "xgboost_optimized")

        # Generate feature importance plot
        plt.figure(figsize=(12, 8))
        plot_importance(best_model, max_num_features=20)
        plt.tight_layout()
        importance_path = MODELS_DIR / 'feature_importance.png'
        plt.savefig(importance_path)
        plt.close()

        # Log feature importance plot
        mlflow.log_artifact(str(importance_path))

        logger.info(f"XGBoost - AUC: {auc:.4f}, F1: {f1:.4f}")

    # Save results
    results_dict = {
        "xgboost_optimized": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auc": float(auc),
            "best_params": best_params
        }
    }

    with open(MODELS_DIR / 'training_results.json', 'w') as f:
        json.dump(results_dict, f, indent=4)

    return best_model

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

    # Create directory for SHAP visualizations
    shap_dir = MODELS_DIR / 'shap_explanations'
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

    # Generate SHAP force plot for a sample of patients
    # This shows how each feature contributes to the prediction for individual patients
    sample_indices = np.random.choice(X_sample.shape[0], min(10, X_sample.shape[0]), replace=False)
    force_plot = shap.force_plot(
        explainer.expected_value if not isinstance(explainer.expected_value, list) else explainer.expected_value[1],
        shap_values[sample_indices, :],
        X_sample.iloc[sample_indices, :],
        feature_names=feature_names,
        show=False
    )
    shap.save_html(str(shap_dir / 'shap_force_plot.html'), force_plot)

    logger.info(f"SHAP explanations generated and saved to {shap_dir}")
    return shap_values

def main():
    """Main function to execute the model training pipeline."""
    logger.info("Starting model training")

    # Load data
    X_train, y_train, X_val, y_val = load_data()

    # Train XGBoost model
    best_model = train_xgboost_model(X_train, y_train, X_val, y_val)

    # Generate explanations
    generate_explanations(best_model, X_train, X_train.columns)

    # Save the best model
    logger.info("Saving the optimized XGBoost model")
    with open(MODELS_DIR / 'model.pkl', 'wb') as f:
        pickle.dump(best_model, f)

    # Save model metadata
    model_info = {
        "model_name": "xgboost_optimized",
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
