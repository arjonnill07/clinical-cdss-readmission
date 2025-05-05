"""
Data drift monitoring script for the Clinical CDSS Readmission project.

This script:
1. Loads the reference data (training data)
2. Compares it with current production data
3. Detects data drift using statistical tests
4. Generates drift reports and visualizations
5. Triggers alerts if significant drift is detected

Theory:
- Data drift occurs when the statistical properties of the data change over time
- Concept drift occurs when the relationship between features and target changes
- Monitoring data drift helps maintain model performance
- Statistical tests can quantify the degree of drift
- Alerting enables proactive model maintenance
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import json
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import classification_report
import evidently
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab, CatTargetDriftTab
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
FEATURES_DATA_DIR = Path('data/features')
MODELS_DIR = Path('models')
MONITORING_DIR = Path('monitoring')

# Ensure directories exist
MONITORING_DIR.mkdir(parents=True, exist_ok=True)

def load_reference_data():
    """
    Load the reference data (training data).
    
    Returns:
        X_ref: Reference features
        y_ref: Reference target
    """
    logger.info("Loading reference data")
    
    X_ref = pd.read_csv(FEATURES_DATA_DIR / 'X_train.csv')
    y_ref = pd.read_csv(FEATURES_DATA_DIR / 'y_train.csv').iloc[:, 0]
    
    logger.info(f"Loaded reference data with shape: {X_ref.shape}")
    return X_ref, y_ref

def load_current_data():
    """
    Load the current production data.
    
    In a real-world scenario, this would come from your production database.
    For demonstration, we'll use the test data as a proxy.
    
    Returns:
        X_curr: Current features
        y_curr: Current target
    """
    logger.info("Loading current data")
    
    X_curr = pd.read_csv(FEATURES_DATA_DIR / 'X_test.csv')
    y_curr = pd.read_csv(FEATURES_DATA_DIR / 'y_test.csv').iloc[:, 0]
    
    logger.info(f"Loaded current data with shape: {X_curr.shape}")
    return X_curr, y_curr

def detect_data_drift(X_ref, X_curr):
    """
    Detect data drift between reference and current data.
    
    Args:
        X_ref: Reference features
        X_curr: Current features
        
    Returns:
        drift_results: Dictionary with drift detection results
    """
    logger.info("Detecting data drift")
    
    # Initialize results dictionary
    drift_results = {
        "timestamp": datetime.now().isoformat(),
        "feature_drift": {},
        "overall_drift": False,
        "drift_score": 0.0,
        "drifted_features": []
    }
    
    # Ensure both datasets have the same columns
    common_cols = list(set(X_ref.columns) & set(X_curr.columns))
    X_ref = X_ref[common_cols]
    X_curr = X_curr[common_cols]
    
    # Calculate drift for each feature
    drift_count = 0
    total_drift_score = 0.0
    
    for col in common_cols:
        # Skip columns with all identical values
        if X_ref[col].nunique() <= 1 or X_curr[col].nunique() <= 1:
            continue
        
        # Perform Kolmogorov-Smirnov test
        ks_stat, p_value = stats.ks_2samp(X_ref[col], X_curr[col])
        
        # Calculate drift score (1 - p_value)
        drift_score = 1.0 - p_value
        total_drift_score += drift_score
        
        # Determine if drift is significant (p < 0.05)
        is_drift = p_value < 0.05
        
        if is_drift:
            drift_count += 1
            drift_results["drifted_features"].append(col)
        
        # Store results
        drift_results["feature_drift"][col] = {
            "ks_statistic": float(ks_stat),
            "p_value": float(p_value),
            "drift_score": float(drift_score),
            "is_drift": bool(is_drift)
        }
    
    # Calculate overall drift metrics
    drift_results["overall_drift"] = drift_count > len(common_cols) * 0.1  # If >10% of features have drift
    drift_results["drift_score"] = float(total_drift_score / len(common_cols))
    drift_results["drift_percentage"] = float(drift_count / len(common_cols) * 100)
    
    logger.info(f"Detected drift in {drift_count} out of {len(common_cols)} features")
    logger.info(f"Overall drift score: {drift_results['drift_score']:.4f}")
    
    return drift_results

def generate_drift_report(X_ref, y_ref, X_curr, y_curr, drift_results):
    """
    Generate a comprehensive drift report.
    
    Args:
        X_ref: Reference features
        y_ref: Reference target
        X_curr: Current features
        y_curr: Current target
        drift_results: Results from drift detection
        
    Returns:
        report_path: Path to the generated report
    """
    logger.info("Generating drift report")
    
    # Create report directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = MONITORING_DIR / f"drift_report_{timestamp}"
    report_dir.mkdir(exist_ok=True)
    
    # Save drift results
    with open(report_dir / 'drift_results.json', 'w') as f:
        json.dump(drift_results, f, indent=4)
    
    # Generate visualizations for drifted features
    if drift_results["drifted_features"]:
        plt.figure(figsize=(12, 8))
        
        # Plot drift scores for all features
        features = list(drift_results["feature_drift"].keys())
        drift_scores = [info["drift_score"] for info in drift_results["feature_drift"].values()]
        
        # Sort by drift score
        sorted_indices = np.argsort(drift_scores)[::-1]
        sorted_features = [features[i] for i in sorted_indices]
        sorted_scores = [drift_scores[i] for i in sorted_indices]
        
        # Plot top 20 features
        plt.barh(sorted_features[:20], sorted_scores[:20])
        plt.xlabel('Drift Score')
        plt.ylabel('Feature')
        plt.title('Feature Drift Scores')
        plt.tight_layout()
        plt.savefig(report_dir / 'drift_scores.png')
        plt.close()
        
        # Generate distribution plots for top drifted features
        top_drifted = sorted_features[:5]
        
        for feature in top_drifted:
            plt.figure(figsize=(10, 6))
            
            # Plot histograms
            plt.hist(X_ref[feature], alpha=0.5, label='Reference', bins=30)
            plt.hist(X_curr[feature], alpha=0.5, label='Current', bins=30)
            
            plt.xlabel(feature)
            plt.ylabel('Frequency')
            plt.title(f'Distribution Comparison for {feature}')
            plt.legend()
            plt.tight_layout()
            plt.savefig(report_dir / f'dist_{feature}.png')
            plt.close()
    
    # Generate Evidently AI report
    try:
        # Prepare data for Evidently
        ref_data = X_ref.copy()
        ref_data['target'] = y_ref
        
        curr_data = X_curr.copy()
        curr_data['target'] = y_curr
        
        # Create data drift dashboard
        data_drift_dashboard = Dashboard(tabs=[DataDriftTab()])
        data_drift_dashboard.calculate(ref_data, curr_data, column_mapping=None)
        data_drift_dashboard.save(report_dir / 'data_drift_dashboard.html')
        
        # Create target drift dashboard
        target_drift_dashboard = Dashboard(tabs=[CatTargetDriftTab()])
        target_drift_dashboard.calculate(ref_data, curr_data, column_mapping=None)
        target_drift_dashboard.save(report_dir / 'target_drift_dashboard.html')
        
        logger.info("Evidently AI reports generated successfully")
    except Exception as e:
        logger.warning(f"Error generating Evidently AI reports: {e}")
    
    logger.info(f"Drift report generated at {report_dir}")
    return report_dir

def check_model_performance(X_curr, y_curr):
    """
    Check the model's performance on current data.
    
    Args:
        X_curr: Current features
        y_curr: Current target
        
    Returns:
        performance_metrics: Dictionary with performance metrics
    """
    logger.info("Checking model performance on current data")
    
    try:
        # Load the model
        with open(MODELS_DIR / 'model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Make predictions
        y_pred = model.predict(X_curr)
        y_pred_proba = model.predict_proba(X_curr)[:, 1]
        
        # Calculate performance metrics
        report = classification_report(y_curr, y_pred, output_dict=True)
        
        # Extract key metrics
        performance_metrics = {
            "accuracy": float(report["accuracy"]),
            "precision": float(report["1"]["precision"]),
            "recall": float(report["1"]["recall"]),
            "f1": float(report["1"]["f1-score"]),
            "support": int(report["1"]["support"])
        }
        
        # Calculate AUC if possible
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_curr, y_pred_proba)
            performance_metrics["auc"] = float(auc)
        except Exception as e:
            logger.warning(f"Error calculating AUC: {e}")
        
        logger.info(f"Model performance - Accuracy: {performance_metrics['accuracy']:.4f}, "
                   f"AUC: {performance_metrics.get('auc', 'N/A')}")
        
        return performance_metrics
    
    except Exception as e:
        logger.error(f"Error checking model performance: {e}")
        return {"error": str(e)}

def trigger_alerts(drift_results, performance_metrics):
    """
    Trigger alerts if significant drift is detected or performance degrades.
    
    In a real-world scenario, this would send emails, Slack messages, etc.
    
    Args:
        drift_results: Results from drift detection
        performance_metrics: Model performance metrics
    """
    logger.info("Checking if alerts should be triggered")
    
    alerts = []
    
    # Check for significant data drift
    if drift_results["overall_drift"]:
        drift_msg = (f"DATA DRIFT ALERT: Significant data drift detected with score "
                    f"{drift_results['drift_score']:.4f}. "
                    f"{len(drift_results['drifted_features'])} features have drifted.")
        alerts.append(drift_msg)
        logger.warning(drift_msg)
    
    # Check for performance degradation
    if "error" not in performance_metrics:
        # Assuming we have a threshold for acceptable performance
        if performance_metrics.get("auc", 1.0) < 0.7:
            perf_msg = (f"PERFORMANCE ALERT: Model performance has degraded. "
                       f"AUC: {performance_metrics.get('auc', 'N/A')}")
            alerts.append(perf_msg)
            logger.warning(perf_msg)
    
    # Send alerts (in a real system, this would use a notification service)
    if alerts:
        alert_file = MONITORING_DIR / f"alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(alert_file, 'w') as f:
            f.write("\n".join(alerts))
        
        logger.info(f"Alerts saved to {alert_file}")
    else:
        logger.info("No alerts triggered")

def main():
    """Main function to execute the data drift monitoring pipeline."""
    logger.info("Starting data drift monitoring")
    
    # Load reference and current data
    X_ref, y_ref = load_reference_data()
    X_curr, y_curr = load_current_data()
    
    # Detect data drift
    drift_results = detect_data_drift(X_ref, X_curr)
    
    # Generate drift report
    report_dir = generate_drift_report(X_ref, y_ref, X_curr, y_curr, drift_results)
    
    # Check model performance
    performance_metrics = check_model_performance(X_curr, y_curr)
    
    # Trigger alerts if necessary
    trigger_alerts(drift_results, performance_metrics)
    
    logger.info("Data drift monitoring completed successfully")

if __name__ == "__main__":
    main()
