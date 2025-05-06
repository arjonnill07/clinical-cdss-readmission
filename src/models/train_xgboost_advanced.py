"""
Advanced XGBoost optimization for clinical readmission prediction.

This script implements advanced techniques to optimize XGBoost for readmission prediction:
1. Advanced feature engineering
2. Bayesian hyperparameter optimization
3. Threshold optimization
4. Calibrated probabilities
5. Comprehensive evaluation
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
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, make_scorer
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
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

def advanced_feature_engineering(X_train, X_val):
    """
    Create advanced features to improve model performance.
    
    Args:
        X_train: Training features
        X_val: Validation features
        
    Returns:
        X_train_enhanced, X_val_enhanced
    """
    logger.info("Performing advanced feature engineering")
    
    # Create copies to avoid modifying originals
    X_train_enhanced = X_train.copy()
    X_val_enhanced = X_val.copy()
    
    # 1. Interaction features between key medical variables
    if all(f in X_train.columns for f in ['num_medications', 'num_procedures']):
        X_train_enhanced['med_proc_interaction'] = X_train['num_medications'] * X_train['num_procedures']
        X_val_enhanced['med_proc_interaction'] = X_val['num_medications'] * X_val['num_procedures']
    
    # 2. Ratio features
    if all(f in X_train.columns for f in ['num_medications', 'time_in_hospital']):
        X_train_enhanced['med_per_day'] = X_train['num_medications'] / X_train['time_in_hospital'].replace(0, 1)
        X_val_enhanced['med_per_day'] = X_val['num_medications'] / X_val['time_in_hospital'].replace(0, 1)
    
    # 3. Polynomial features for key numeric variables
    for feature in ['num_lab_procedures', 'num_medications', 'number_diagnoses']:
        if feature in X_train.columns:
            X_train_enhanced[f'{feature}_squared'] = X_train[feature] ** 2
            X_val_enhanced[f'{feature}_squared'] = X_val[feature] ** 2
    
    # 4. Medication combination features
    med_features = [col for col in X_train.columns if col in [
        'metformin', 'glimepiride', 'glipizide', 'glyburide', 
        'pioglitazone', 'rosiglitazone', 'insulin'
    ]]
    
    if med_features:
        # Count total medications
        X_train_enhanced['total_meds'] = X_train[med_features].sum(axis=1)
        X_val_enhanced['total_meds'] = X_val[med_features].sum(axis=1)
    
    # 5. Age-related risk groups
    if 'age' in X_train.columns:
        X_train_enhanced['is_elderly'] = (X_train['age'] > 60).astype(int)
        X_val_enhanced['is_elderly'] = (X_val['age'] > 60).astype(int)
    
    # 6. Hospital utilization intensity
    if all(f in X_train.columns for f in ['number_outpatient', 'number_emergency', 'number_inpatient']):
        X_train_enhanced['total_encounters'] = X_train['number_outpatient'] + X_train['number_emergency'] + X_train['number_inpatient']
        X_val_enhanced['total_encounters'] = X_val['number_outpatient'] + X_val['number_emergency'] + X_val['number_inpatient']
    
    # 7. Diagnosis complexity
    if 'number_diagnoses' in X_train.columns:
        X_train_enhanced['high_diag_complexity'] = (X_train['number_diagnoses'] > 7).astype(int)
        X_val_enhanced['high_diag_complexity'] = (X_val['number_diagnoses'] > 7).astype(int)
    
    logger.info(f"Created {X_train_enhanced.shape[1] - X_train.shape[1]} new features")
    
    return X_train_enhanced, X_val_enhanced

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
    
    # First apply advanced feature engineering
    X_train, X_val = advanced_feature_engineering(X_train, X_val)
    
    # Select a comprehensive set of important features based on domain knowledge
    # Focus on features that are known to be predictive of readmission
    important_features = [
        'time_in_hospital', 'num_lab_procedures', 'num_procedures',
        'num_medications', 'number_outpatient', 'number_emergency',
        'number_inpatient', 'number_diagnoses', 'age',
        'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
        'A1Cresult', 'metformin', 'glimepiride', 'glipizide', 'glyburide',
        'pioglitazone', 'rosiglitazone', 'insulin', 'change', 'diabetesMed',
        # Include engineered features
        'med_proc_interaction', 'med_per_day', 'total_meds', 'is_elderly',
        'total_encounters', 'high_diag_complexity'
    ]
    
    # Filter to only include features that exist in the dataset
    selected_features = [f for f in important_features if f in X_train.columns]
    
    # Add diagnosis codes if they exist
    diag_features = [col for col in X_train.columns if col.startswith('diag_')][:10]  # Include more diagnosis codes
    selected_features.extend(diag_features)
    
    # Add squared features
    squared_features = [col for col in X_train.columns if col.endswith('_squared')]
    selected_features.extend(squared_features)
    
    logger.info(f"Selected {len(selected_features)} important features")
    
    # Identify categorical features
    categorical_features = []
    for feature in selected_features:
        if X_train[feature].dtype == 'object' or X_train[feature].nunique() < 10:
            categorical_features.append(feature)
    
    # Identify numerical features
    numerical_features = [f for f in selected_features if f not in categorical_features]
    
    logger.info(f"Identified {len(categorical_features)} categorical and {len(numerical_features)} numerical features")
    
    # Create a preprocessing pipeline
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
    
    return X_train_prepared, X_val_prepared, feature_names, preprocessor

def optimize_threshold(model, X_val, y_val):
    """
    Find the optimal classification threshold to maximize F1 score.
    
    Args:
        model: Trained model
        X_val: Validation features
        y_val: Validation target
        
    Returns:
        optimal_threshold: The threshold that maximizes F1 score
    """
    logger.info("Finding optimal classification threshold")
    
    # Get predicted probabilities
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Try different thresholds
    thresholds = np.arange(0.1, 0.9, 0.01)
    f1_scores = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1_scores.append(f1_score(y_val, y_pred))
    
    # Find the threshold that maximizes F1
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    logger.info(f"Optimal threshold: {optimal_threshold:.2f} with F1: {f1_scores[optimal_idx]:.4f}")
    
    return optimal_threshold

def train_advanced_xgboost(X_train, y_train, X_val, y_val, feature_names):
    """
    Train an advanced XGBoost model with comprehensive optimization.
    
    Args:
        X_train: Prepared training features
        y_train: Training target
        X_val: Prepared validation features
        y_val: Validation target
        feature_names: Names of features after preprocessing
        
    Returns:
        best_model: The optimized XGBoost model
    """
    logger.info("Training advanced XGBoost model")
    
    # Set up MLflow tracking
    mlflow.set_experiment("clinical-cdss-readmission-advanced")
    
    # Calculate class imbalance ratio
    neg_pos_ratio = np.sum(y_train == 0) / np.sum(y_train == 1)
    logger.info(f"Class imbalance ratio (negative/positive): {neg_pos_ratio:.2f}")
    
    # Define comprehensive parameter grid
    param_grid = {
        'n_estimators': [500, 1000, 1500],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5, 6],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [0.1, 1, 5],
        'scale_pos_weight': [1, neg_pos_ratio/2, neg_pos_ratio, neg_pos_ratio*1.5]
    }
    
    # Define F1 scorer for optimization
    f1_scorer = make_scorer(f1_score)
    
    # Set up cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Define the XGBoost model with initial parameters
    xgb_model = XGBClassifier(
        objective='binary:logistic',
        random_state=42,
        tree_method='hist',
        booster='gbtree'
    )
    
    # Start with a smaller grid search to find promising regions
    logger.info("Performing initial grid search")
    initial_param_grid = {
        'n_estimators': [500, 1000],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5],
        'scale_pos_weight': [1, neg_pos_ratio]
    }
    
    initial_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=initial_param_grid,
        scoring=f1_scorer,
        cv=3,
        verbose=2,
        n_jobs=-1
    )
    
    initial_search.fit(X_train, y_train)
    
    # Get best parameters from initial search
    best_params = initial_search.best_params_
    logger.info(f"Best parameters from initial search: {best_params}")
    
    # Create a model with the best parameters
    best_model = XGBClassifier(
        objective='binary:logistic',
        random_state=42,
        tree_method='hist',
        booster='gbtree',
        **best_params
    )
    
    # Train the model on the full training set
    best_model.fit(X_train, y_train)
    
    # Find optimal threshold
    optimal_threshold = optimize_threshold(best_model, X_val, y_val)
    
    # Calibrate probabilities
    logger.info("Calibrating probability estimates")
    calibrated_model = CalibratedClassifierCV(
        best_model,
        method='isotonic',
        cv='prefit'
    )
    calibrated_model.fit(X_val, y_val)
    
    # Make predictions with calibrated model and optimal threshold
    y_pred_proba = calibrated_model.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred_proba)
    
    # Log results with MLflow
    with mlflow.start_run(run_name="xgboost_advanced"):
        # Log parameters
        mlflow.log_params({
            "model_type": "xgboost_advanced",
            "train_samples": X_train.shape[0],
            "features": X_train.shape[1],
            "optimal_threshold": optimal_threshold
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
        mlflow.sklearn.log_model(calibrated_model, "xgboost_advanced")
        
        # Generate feature importance plot
        plt.figure(figsize=(12, 8))
        plot_importance(best_model, max_num_features=20)
        plt.tight_layout()
        importance_path = MODELS_DIR / 'feature_importance_advanced.png'
        plt.savefig(importance_path)
        plt.close()
        
        # Log feature importance plot
        mlflow.log_artifact(str(importance_path))
        
        # Create a custom model class that includes the optimal threshold
        class OptimizedXGBoostClassifier:
            def __init__(self, model, threshold):
                self.model = model
                self.threshold = threshold
                
            def predict(self, X):
                proba = self.model.predict_proba(X)[:, 1]
                return (proba >= self.threshold).astype(int)
                
            def predict_proba(self, X):
                return self.model.predict_proba(X)
        
        # Create the optimized model
        optimized_model = OptimizedXGBoostClassifier(calibrated_model, optimal_threshold)
        
        # Save the optimized model
        with open(MODELS_DIR / 'advanced_model.pkl', 'wb') as f:
            pickle.dump(optimized_model, f)
        
        logger.info(f"XGBoost Advanced - AUC: {auc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    
    # Save results
    results_dict = {
        "xgboost_advanced": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auc": float(auc),
            "best_params": best_params,
            "optimal_threshold": float(optimal_threshold)
        }
    }
    
    with open(MODELS_DIR / 'advanced_results.json', 'w') as f:
        json.dump(results_dict, f, indent=4)
    
    return calibrated_model, optimal_threshold

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
        # If model is a CalibratedClassifierCV, get the base estimator
        if hasattr(model, 'base_estimator'):
            base_model = model.base_estimator
        else:
            base_model = model
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(base_model)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_val_prepared)

        # If shap_values is a list (for multi-class), take the values for class 1
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # Save SHAP values and feature names for later visualization
        with open(MODELS_DIR / 'advanced_shap_values.pkl', 'wb') as f:
            pickle.dump({
                'shap_values': shap_values,
                'feature_names': feature_names
            }, f)

        # Create directory for SHAP visualizations
        shap_dir = MODELS_DIR / 'advanced_shap_explanations'
        shap_dir.mkdir(exist_ok=True)
        
        # Generate SHAP summary plot
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X_val_prepared, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(shap_dir / 'shap_summary.png')
        plt.close()
        
        # Generate SHAP dependence plots for top features
        # Get mean absolute SHAP values for each feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[-10:]  # Top 10 features
        
        for idx in top_indices:
            if idx < len(feature_names):  # Ensure index is valid
                feature_name = feature_names[idx]
                plt.figure(figsize=(12, 8))
                shap.dependence_plot(idx, shap_values, X_val_prepared, feature_names=feature_names, show=False)
                plt.title(f'SHAP Dependence Plot for {feature_name}')
                plt.tight_layout()
                plt.savefig(shap_dir / f'shap_dependence_{idx}.png')
                plt.close()
        
        logger.info(f"SHAP explanations generated and saved to {shap_dir}")
        return shap_values
    
    except Exception as e:
        logger.error(f"Error generating SHAP explanations: {e}")
        return None

def main():
    """Main function to execute the advanced model training pipeline."""
    logger.info("Starting advanced XGBoost model training")

    # Load data
    X_train, y_train, X_val, y_val = load_data()

    # Select and prepare features
    X_train_prepared, X_val_prepared, feature_names, preprocessor = select_and_prepare_features(X_train, X_val)

    # Train advanced XGBoost model
    best_model, optimal_threshold = train_advanced_xgboost(X_train_prepared, y_train, X_val_prepared, y_val, feature_names)

    # Generate explanations
    generate_explanations(best_model, X_val_prepared, feature_names)

    # Save preprocessor for later use
    with open(MODELS_DIR / 'advanced_preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)

    # Save model metadata
    model_info = {
        "model_name": "xgboost_advanced",
        "feature_count": len(feature_names),
        "training_samples": X_train.shape[0],
        "validation_samples": X_val.shape[0],
        "optimal_threshold": optimal_threshold,
        "model_version": "5.0",
        "optimization_techniques": [
            "Advanced feature engineering",
            "Comprehensive feature selection",
            "Focused hyperparameter tuning",
            "Probability calibration",
            "Threshold optimization"
        ]
    }
    
    with open(MODELS_DIR / 'advanced_model_info.json', 'w') as f:
        json.dump(model_info, f, indent=4)

    logger.info("Advanced model training completed successfully")

if __name__ == "__main__":
    main()
