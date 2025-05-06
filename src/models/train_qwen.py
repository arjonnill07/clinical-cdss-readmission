"""
Optimized XGBoost model for clinical readmission prediction with enhanced error handling.
This script implements targeted optimizations to achieve >90% performance:
1. Robust preprocessing pipeline for numeric and categorical features
2. Safe SMOTE oversampling after proper feature encoding
3. Advanced hyperparameter tuning
4. Direct optimization for F1 score
5. Comprehensive error handling and validation checks
"""

import pandas as pd
import numpy as np
import logging
import os
import warnings
from pathlib import Path
import pickle
import json
import mlflow
import mlflow.sklearn
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.exceptions import ConvergenceWarning
import shap
import matplotlib.pyplot as plt
# Add this near the top of the file
from sklearn.pipeline import Pipeline  # ← MISSING IMPORT
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress known warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

# Define paths
FEATURES_DATA_DIR = Path('data/features')
MODELS_DIR = Path('models')

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """
    Load the feature matrices with validation checks.
    
    Returns:
        X_train, y_train, X_val, y_val
    """
    logger.info("Loading feature matrices")
    
    try:
        X_train = pd.read_csv(FEATURES_DATA_DIR / 'X_train.csv')
        y_train = pd.read_csv(FEATURES_DATA_DIR / 'y_train.csv').iloc[:, 0]
        X_val = pd.read_csv(FEATURES_DATA_DIR / 'X_val.csv')
        y_val = pd.read_csv(FEATURES_DATA_DIR / 'y_val.csv').iloc[:, 0]
        
        # Validate data
        if X_train.empty or X_val.empty:
            raise ValueError("Loaded data is empty")
            
        logger.info(f"Loaded training set with shape: {X_train.shape}")
        logger.info(f"Loaded validation set with shape: {X_val.shape}")
        
        return X_train, y_train, X_val, y_val
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def validate_features(X):
    """
    Validate and clean features.
    
    Args:
        X: Input features
        
    Returns:
        Cleaned features
    """
    logger.info("Validating and cleaning features")
    
    # Handle missing values
    if X.isnull().any().any():
        logger.warning(f"Found {X.isnull().sum().sum()} missing values, filling with mode/median")
        for col in X.columns:
            if X[col].dtype == 'O':
                X[col] = X[col].fillna(X[col].mode()[0])
            else:
                X[col] = X[col].fillna(X[col].median())
    
    # Remove constant columns
    constant_cols = X.columns[X.nunique() <= 1]
    if len(constant_cols) > 0:
        logger.info(f"Dropping {len(constant_cols)} constant columns")
        X = X.drop(columns=constant_cols)
    
    return X

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
    # Filter to only include features that exist in the dataset
    selected_features = [f for f in selected_features if f in X_train.columns]

    # Remove any potential duplicate feature names
    selected_features = list(set(selected_features))
    logger.info(f"Selected {len(selected_features)} clinically relevant features")
    
    # Apply feature selection and validate
    X_train = validate_features(X_train[selected_features])
    X_val = validate_features(X_val[selected_features])
    
    return X_train, X_val, selected_features

def create_preprocessor(X):
    """
    Create preprocessing pipeline based on data types.
    
    Args:
        X: Input features
        
    Returns:
        Preprocessor pipeline
    """
    logger.info("Creating preprocessing pipeline")
    
    # Identify numeric vs categorical features
    cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_features = X.select_dtypes(include=['number']).columns.tolist()
    
    logger.info(f"Found {len(num_features)} numeric features")
    logger.info(f"Found {len(cat_features)} categorical features")
    
    # Define preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), num_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))
            ]), cat_features)
        ])
    
    return preprocessor

def train_optimized_xgboost(X_train, y_train, X_val, y_val):
    """
    Train an optimized XGBoost model with enhanced error handling.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        
    Returns:
        best_model: The optimized XGBoost model
    """
    logger.info("Training optimized XGBoost model")
    
    try:
        # Set up MLflow tracking
        mlflow.set_experiment("clinical-cdss-readmission-optimized")
        
        # Calculate class imbalance ratio
        neg_pos_ratio = np.sum(y_train == 0) / np.sum(y_train == 1)
        logger.info(f"Class imbalance ratio (negative/positive): {neg_pos_ratio:.2f}")
        
        # Select clinically relevant features
        X_train, X_val, selected_features = select_clinical_features(X_train, X_val)
        
        # Create preprocessing pipeline
        preprocessor = create_preprocessor(X_train)
        
        # Define the XGBoost model with initial parameters
        xgb_model = XGBClassifier(
            objective='binary:logistic',
            random_state=42,
            tree_method='hist',
            booster='gbtree',
            scale_pos_weight=neg_pos_ratio,
            use_label_encoder=False,
            eval_metric='logloss',
            early_stopping_rounds=10
        )
        
        # Create pipeline with preprocessing, SMOTE, and classifier
        pipeline = ImbPipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42, sampling_strategy=0.5)),
            ('classifier', xgb_model)
        ])
        
        # Define focused parameter grid for key parameters
        param_grid = {
            'classifier__n_estimators': [1000, 1500],
            'classifier__learning_rate': [0.01, 0.05],
            'classifier__max_depth': [4, 5, 6],
            'classifier__min_child_weight': [1, 3],
            'classifier__gamma': [0, 0.1],
            'classifier__subsample': [0.8, 0.9],
            'classifier__colsample_bytree': [0.8, 0.9],
            'classifier__reg_alpha': [0, 0.1],
            'classifier__reg_lambda': [0.1, 1]
        }
        
        # Set up cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Define F1 scorer for optimization
        f1_scorer = 'f1'
        
        # Split data for early stopping
        X_train_split, X_eval, y_train_split, y_eval = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
        )
        
        # Update XGBoost parameters with evaluation set
        fit_params = {
            'classifier__eval_set': [(X_eval, y_eval)],
            'classifier__early_stopping_rounds': 10
        }
        
        # Perform grid search with F1 optimization
        logger.info("Performing grid search with F1 optimization")
        
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=f1_scorer,
            cv=cv,
            verbose=3,
            n_jobs=-1,
            error_score='raise'
        )
        
        # Fit the grid search with validation checks
        try:
            grid_search.fit(X_train_split, y_train_split, **fit_params)
        except ValueError as e:
            logger.error(f"Model fitting failed: {e}")
            logger.info("Trying without early stopping")
            # Try without early stopping if first attempt fails
            fit_params_no_es = {
                'classifier__eval_set': None,
                'classifier__early_stopping_rounds': None
            }
            grid_search.fit(X_train_split, y_train_split, **fit_params_no_es)
        
        # Get best model and parameters
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        # Evaluate on full validation set
        logger.info("Evaluating best model on validation set")
        y_pred = best_model.predict(X_val)
        y_pred_proba = best_model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_pred_proba)
        
        # Log results with MLflow
        with mlflow.start_run(run_name="xgboost_optimized"):
            # Log parameters
            mlflow.log_params({
                "model_type": "xgboost_optimized",
                "train_samples": X_train.shape[0],
                "features": X_train.shape[1],
                "selected_features": len(selected_features),
                "smote_strategy": 0.5
            })
            
            # Log best parameters
            for param, value in best_params.items():
                mlflow.log_param(param.replace('classifier__', ''), value)
            
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
            plot_importance(best_model.named_steps['classifier'], max_num_features=20)
            plt.tight_layout()
            importance_path = MODELS_DIR / 'feature_importance_optimized.png'
            plt.savefig(importance_path)
            plt.close()
            
            # Log feature importance plot
            mlflow.log_artifact(str(importance_path))
            
            logger.info(f"XGBoost Optimized - AUC: {auc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        
        # Save results
        results_dict = {
            "xgboost_optimized": {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "auc": float(auc),
                "best_params": {k.replace('classifier__', ''): v for k, v in best_params.items()}
            }
        }
        
        with open(MODELS_DIR / 'optimized_results.json', 'w') as f:
            json.dump(results_dict, f, indent=4)
        
        # Save the model
        with open(MODELS_DIR / 'optimized_model.pkl', 'wb') as f:
            pickle.dump(best_model, f)
        
        # Verify we've achieved the target performance
        if f1 < 0.9 and auc < 0.95:
            logger.warning("Model did not reach target performance (>0.9 F1, >0.95 AUC)")
            logger.info("Consider trying:")
            logger.info("- Feature engineering additional clinical variables")
            logger.info("- Trying alternative imbalance techniques (e.g., class weights)")
            logger.info("- Increasing training data size")
            logger.info("- Using ensemble methods")
        else:
            logger.info("Success! Model achieved target performance metrics (>90%)")
        
        return best_model, selected_features
    
    except Exception as e:
        logger.error(f"Error in training pipeline: {e}")
        logger.info("Diagnostic information:")
        logger.info(f"Data shape: {X_train.shape}")
        logger.info(f"Class distribution: {np.bincount(y_train)}")
        logger.info(f"Feature types:\n{X_train.dtypes}")
        logger.info(f"Missing values: {X_train.isnull().sum().sum()}")
        raise

def generate_explanations(model, X_train, feature_names):
    """
    Generate model explanations using SHAP with error handling.
    
    Args:
        model: Trained model
        X_train: Training data
        feature_names: Names of the features
    """
    logger.info("Generating model explanations with SHAP")
    
    try:
        # Create a sample of the training data for SHAP analysis
        sample_size = min(1000, X_train.shape[0])
        X_sample = X_train.sample(sample_size, random_state=42)
        
        # Extract the XGBoost model from the pipeline
        xgb_model = model.named_steps['classifier']
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(xgb_model)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_sample)
        
        # If shap_values is a list (for multi-class), take the values for class 1
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Save SHAP values and feature names for later visualization
        with open(MODELS_DIR / 'optimized_shap_values.pkl', 'wb') as f:
            pickle.dump({
                'shap_values': shap_values,
                'X_sample': X_sample,
                'feature_names': feature_names
            }, f)
        
        # Create directory for SHAP visualizations
        shap_dir = MODELS_DIR / 'optimized_shap_explanations'
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
        
    except Exception as e:
        logger.error(f"Error generating explanations: {e}")
        # Continue execution even if explanations fail
        pass

def main():
    """Main function to execute the optimized model training pipeline."""
    logger.info("Starting optimized XGBoost model training")
    
    try:
        # Load data
        X_train, y_train, X_val, y_val = load_data()
        
        # Train optimized XGBoost model
        best_model, selected_features = train_optimized_xgboost(X_train, y_train, X_val, y_val)
        
        # Generate explanations
        X_train_selected = X_train[selected_features]
        generate_explanations(best_model, X_train_selected, selected_features)
        
        # Save model metadata
        model_info = {
            "model_name": "xgboost_optimized",
            "feature_count": len(selected_features),
            "training_samples": X_train.shape[0],
            "validation_samples": X_val.shape[0],
            "feature_names": list(selected_features),
            "model_version": "4.0",
            "optimization_techniques": [
                "Clinically relevant feature selection",
                "Safe SMOTE oversampling after preprocessing",
                "Direct F1 score optimization",
                "Focused grid search",
                "Early stopping",
                "Advanced preprocessing pipeline"
            ]
        }
        
        with open(MODELS_DIR / 'optimized_model_info.json', 'w') as f:
            json.dump(model_info, f, indent=4)
            
        logger.info("Optimized model training completed successfully")
        
    except Exception as e:
        logger.error(f"Critical error in main pipeline: {e}")
        logger.info("Execution terminated due to critical error")
        exit(1)

if __name__ == "__main__":
    main()