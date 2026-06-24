import os
import time
import joblib
import json
import numpy as np
import pandas as pd
from datetime import datetime
from src.config import BEST_MODEL_PATH, PREPROCESSOR_PATH, REPORTS_DIR, MODEL_DIR

def calculate_psi(expected, actual, num_buckets=10):
    """
    Computes the Population Stability Index (PSI) between two 1D numerical arrays.
    PSI = sum( (Actual% - Expected%) * ln(Actual% / Expected%) )
    Interpretations:
    - PSI < 0.1: No significant change / Stable.
    - 0.1 <= PSI < 0.25: Moderate change / Monitor.
    - PSI >= 0.25: Significant change / Action required (re-train).
    """
    # Remove nulls
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
        
    # Get quantile boundaries from expected
    percentiles = np.linspace(0, 100, num_buckets + 1)
    buckets = np.percentile(expected, percentiles)
    
    # Adjust boundaries to prevent duplicates
    buckets = np.unique(buckets)
    if len(buckets) < 2:
        return 0.0
        
    buckets[0] = -np.inf
    buckets[-1] = np.inf
    
    # Calculate counts in each bucket
    expected_counts = np.histogram(expected, bins=buckets)[0]
    actual_counts = np.histogram(actual, bins=buckets)[0]
    
    # Convert to fractions
    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)
    
    # Handle zeros to avoid log division by zero errors
    expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
    actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)
    
    # Calculate PSI
    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi_value)

def detect_data_drift(train_df, new_df, numeric_cols):
    """
    Checks for data drift on numerical features between training data and incoming data.
    """
    drift_results = {}
    significant_drift = False
    
    for col in numeric_cols:
        if col in train_df.columns and col in new_df.columns:
            psi = calculate_psi(train_df[col].values, new_df[col].values)
            
            status = "Stable"
            if psi >= 0.25:
                status = "Drift Detected"
                significant_drift = True
            elif psi >= 0.10:
                status = "Moderate Shift"
                
            drift_results[col] = {
                "PSI": round(psi, 4),
                "Status": status
            }
            
    return drift_results, significant_drift

def simulate_production_monitoring():
    """
    Simulates production monitoring, checking drift on a simulated incoming batch of data,
    saves a drift report, and determines if retraining should be triggered.
    """
    print("Running MLOps production monitoring simulation...")
    
    # Load raw data and split into 'historical train' and 'incoming production'
    from src.data_loader import load_raw_data, preprocess_raw_data
    raw_df = load_raw_data()
    clean_df = preprocess_raw_data(raw_df)
    
    # Split
    from src.data_loader import get_train_test_split
    train_df, test_df = get_train_test_split(clean_df, split_method="temporal")
    
    # Let's create a simulated "drifted" production batch by modifying a feature
    # e.g., increasing Sales and Order Item Discount Rate significantly in the test set
    drifted_df = test_df.copy()
    # Shift sales up by 30% and discounts up by 25% (simulating a holiday promo)
    drifted_df["Sales"] = drifted_df["Sales"] * 1.3
    drifted_df["Order Item Discount Rate"] = drifted_df["Order Item Discount Rate"] * 1.25
    
    from src.config import NUMERIC_COLS
    
    # Run drift detection
    drift_results, trigger_retrain = detect_data_drift(train_df, drifted_df, NUMERIC_COLS)
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "drift_detected": trigger_retrain,
        "metrics": drift_results
    }
    
    report_path = os.path.join(REPORTS_DIR, "drift_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"Drift monitoring report saved to {report_path}")
    print(f"Drift status: {'DRIFT DETECTED - RETRAINING REQUIRED' if trigger_retrain else 'SYSTEM STABLE'}")
    
    return report

def promote_new_model(new_model_path, production_model_path, new_metrics, prod_metrics):
    """
    MLOps Model Registry promoter.
    Compares the newly trained model's F1-Score against the production model.
    Promotes the new model if it outperforms the old model.
    """
    new_f1 = new_metrics.get("F1-Score", 0.0)
    prod_f1 = prod_metrics.get("F1-Score", 0.0)
    
    print(f"Comparing model metrics:")
    print(f"  Production Model F1: {prod_f1:.4f}")
    print(f"  New Candidate Model F1: {new_f1:.4f}")
    
    if new_f1 > prod_f1:
        print("Promotion condition met! Copying new model to production path...")
        import shutil
        shutil.copy(new_model_path, production_model_path)
        print("Model promoted successfully.")
        return True
    else:
        print("Production model out-performed the candidate. Promotion cancelled.")
        return False
