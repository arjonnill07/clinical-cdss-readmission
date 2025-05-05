"""
Model evaluation script for the Clinical CDSS Readmission project.

This script:
1. Loads the trained model
2. Evaluates it on the test set
3. Generates detailed performance metrics
4. Creates visualizations for model interpretation
5. Saves evaluation results

Theory:
- Comprehensive evaluation is crucial for healthcare ML models
- Multiple metrics should be considered (not just accuracy)
- Calibration is important for reliable probability estimates
- Subgroup analysis helps identify potential biases
- Explainability builds trust with healthcare professionals
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, brier_score_loss
)
from sklearn.calibration import calibration_curve
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

def load_data_and_model():
    """
    Load the test data and trained model.
    
    Returns:
        X_test: Test features
        y_test: Test target
        model: Trained model
    """
    logger.info("Loading test data and model")
    
    # Load test data
    X_test = pd.read_csv(FEATURES_DATA_DIR / 'X_test.csv')
    y_test = pd.read_csv(FEATURES_DATA_DIR / 'y_test.csv').iloc[:, 0]
    
    logger.info(f"Loaded test set with shape: {X_test.shape}")
    
    # Load model
    with open(MODELS_DIR / 'model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    logger.info("Model loaded successfully")
    
    return X_test, y_test, model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test set.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    logger.info("Evaluating model on test set")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    brier = brier_score_loss(y_test, y_pred_proba)
    
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"AUC: {auc:.4f}")
    logger.info(f"Average Precision: {avg_precision:.4f}")
    logger.info(f"Brier Score: {brier:.4f}")
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate additional metrics
    specificity = tn / (tn + fp)
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    # Compile all metrics
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "sensitivity": float(recall),  # Same as recall
        "specificity": float(specificity),
        "f1_score": float(f1),
        "auc": float(auc),
        "average_precision": float(avg_precision),
        "brier_score": float(brier),
        "positive_predictive_value": float(precision),  # Same as precision
        "negative_predictive_value": float(npv),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        }
    }
    
    # Save metrics
    with open(MODELS_DIR / 'evaluation.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return metrics

def generate_visualizations(model, X_test, y_test):
    """
    Generate visualizations for model evaluation.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
    """
    logger.info("Generating evaluation visualizations")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Create directory for visualizations
    vis_dir = MODELS_DIR / 'visualizations'
    vis_dir.mkdir(exist_ok=True)
    
    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_pred_proba):.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(vis_dir / 'roc_curve.png')
    plt.close()
    
    # 2. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, label=f'AP = {average_precision_score(y_test, y_pred_proba):.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig(vis_dir / 'precision_recall_curve.png')
    plt.close()
    
    # 3. Confusion Matrix Heatmap
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(vis_dir / 'confusion_matrix.png')
    plt.close()
    
    # 4. Calibration Curve
    prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
    plt.figure(figsize=(10, 8))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Probability')
    plt.title('Calibration Curve')
    plt.savefig(vis_dir / 'calibration_curve.png')
    plt.close()
    
    # 5. Feature Importance
    if hasattr(model, 'feature_importances_'):
        # For tree-based models
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        features = X_test.columns
        
        plt.figure(figsize=(12, 10))
        plt.title('Feature Importances')
        plt.barh(range(min(20, len(indices))), importances[indices[:20]], align='center')
        plt.yticks(range(min(20, len(indices))), [features[i] for i in indices[:20]])
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        plt.savefig(vis_dir / 'feature_importances.png')
        plt.close()
    
    # 6. SHAP Summary Plot
    try:
        # Load SHAP values
        with open(MODELS_DIR / 'shap_values.pkl', 'rb') as f:
            shap_data = pickle.load(f)
        
        shap_values = shap_data['shap_values']
        X_sample = shap_data['X_sample']
        
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X_sample, feature_names=X_test.columns, show=False)
        plt.tight_layout()
        plt.savefig(vis_dir / 'shap_summary.png')
        plt.close()
        
        # SHAP Dependence Plots for top features
        # Get mean absolute SHAP values for each feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[-5:]  # Top 5 features
        
        for idx in top_indices:
            feature_name = X_test.columns[idx]
            plt.figure(figsize=(12, 8))
            shap.dependence_plot(idx, shap_values, X_sample, feature_names=X_test.columns, show=False)
            plt.title(f'SHAP Dependence Plot for {feature_name}')
            plt.tight_layout()
            plt.savefig(vis_dir / f'shap_dependence_{feature_name}.png')
            plt.close()
    
    except Exception as e:
        logger.warning(f"Error generating SHAP plots: {e}")
    
    logger.info(f"Visualizations saved to {vis_dir}")

def analyze_subgroups(model, X_test, y_test):
    """
    Analyze model performance across different subgroups.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
    """
    logger.info("Analyzing model performance across subgroups")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Create a DataFrame with predictions and actual values
    results_df = X_test.copy()
    results_df['actual'] = y_test
    results_df['predicted'] = y_pred
    results_df['probability'] = y_pred_proba
    
    # Define subgroups to analyze
    # In a real scenario, you would have demographic information
    # Here we'll use some example features that might be available
    subgroup_features = []
    
    # Age groups (if available)
    if 'age_numeric' in results_df.columns:
        results_df['age_group'] = pd.cut(
            results_df['age_numeric'],
            bins=[0, 30, 50, 70, float('inf')],
            labels=['<30', '30-50', '50-70', '>70']
        )
        subgroup_features.append('age_group')
    
    # Gender (if available)
    if 'gender' in results_df.columns:
        subgroup_features.append('gender')
    
    # Comorbidity count (if available)
    if 'comorbidity_count' in results_df.columns:
        results_df['comorbidity_group'] = pd.cut(
            results_df['comorbidity_count'],
            bins=[0, 1, 3, float('inf')],
            labels=['None', 'Few', 'Many']
        )
        subgroup_features.append('comorbidity_group')
    
    # Length of stay (if available)
    if 'los_group' in results_df.columns:
        subgroup_features.append('los_group')
    
    # If no subgroup features are available, create a dummy one
    if not subgroup_features:
        logger.warning("No subgroup features found. Creating a dummy feature for demonstration.")
        results_df['dummy_group'] = 'All'
        subgroup_features = ['dummy_group']
    
    # Analyze performance for each subgroup
    subgroup_metrics = {}
    
    for feature in subgroup_features:
        subgroup_metrics[feature] = {}
        
        for group in results_df[feature].unique():
            # Skip NaN groups
            if pd.isna(group):
                continue
            
            # Get data for this subgroup
            mask = results_df[feature] == group
            if mask.sum() < 10:  # Skip groups with too few samples
                continue
            
            y_true_group = results_df.loc[mask, 'actual']
            y_pred_group = results_df.loc[mask, 'predicted']
            y_prob_group = results_df.loc[mask, 'probability']
            
            # Calculate metrics
            metrics = {
                "count": int(mask.sum()),
                "accuracy": float(accuracy_score(y_true_group, y_pred_group)),
                "precision": float(precision_score(y_true_group, y_pred_group, zero_division=0)),
                "recall": float(recall_score(y_true_group, y_pred_group, zero_division=0)),
                "f1": float(f1_score(y_true_group, y_pred_group, zero_division=0)),
            }
            
            # Add AUC if there are both positive and negative samples
            if len(y_true_group.unique()) > 1:
                metrics["auc"] = float(roc_auc_score(y_true_group, y_prob_group))
            
            subgroup_metrics[feature][str(group)] = metrics
    
    # Save subgroup analysis
    with open(MODELS_DIR / 'subgroup_analysis.json', 'w') as f:
        json.dump(subgroup_metrics, f, indent=4)
    
    logger.info("Subgroup analysis completed")

def main():
    """Main function to execute the model evaluation pipeline."""
    logger.info("Starting model evaluation")
    
    # Load data and model
    X_test, y_test, model = load_data_and_model()
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Generate visualizations
    generate_visualizations(model, X_test, y_test)
    
    # Analyze subgroups
    analyze_subgroups(model, X_test, y_test)
    
    logger.info("Model evaluation completed successfully")

if __name__ == "__main__":
    main()
