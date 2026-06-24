import os
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score
)

from src.config import (
    TARGET_COL, COLS_TO_DROP, ONE_HOT_COLS, TARGET_ENCODE_COLS, 
    NUMERIC_COLS, RANDOM_STATE, BEST_MODEL_PATH, PREPROCESSOR_PATH, 
    MODEL_COMPARISON_PATH, MODEL_DIR
)
from src.data_loader import load_raw_data, preprocess_raw_data, get_train_test_split
from src.features import LogisticsFeatureEngineer
from src.preprocess import LogisticsPreprocessor

def evaluate_model(model, X_test, y_test):
    """
    Computes performance metrics for a trained model on a test set.
    """
    start_time = time.time()
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    latency = (time.time() - start_time) / len(X_test) * 1000  # Latency per sample in ms
    
    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    roc_auc = roc_auc_score(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)
    
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "Inference Latency (ms/sample)": latency
    }

def run_training_pipeline():
    print("=================== STARTING TRAINING PIPELINE ===================")
    
    # 1. Load and Clean raw data
    raw_df = load_raw_data()
    clean_df = preprocess_raw_data(raw_df)
    
    # 2. Train-Test Split (Temporal/Out-of-time)
    train_df, test_df = get_train_test_split(clean_df, split_method="temporal")
    
    # 3. Fit Feature Engineer and engineer features
    fe = LogisticsFeatureEngineer()
    fe.fit(train_df)
    
    train_fe = fe.transform(train_df)
    test_fe = fe.transform(test_df)
    
    # Drop target-leakage and identifier columns
    cols_to_drop = [c for c in COLS_TO_DROP if c in train_fe.columns]
    # Keep track of target and Order Date for splitting, but separate features
    X_train_raw = train_fe.drop(columns=cols_to_drop + [TARGET_COL, "Order Date"], errors="ignore")
    y_train = train_fe[TARGET_COL]
    
    X_test_raw = test_fe.drop(columns=cols_to_drop + [TARGET_COL, "Order Date"], errors="ignore")
    y_test = test_fe[TARGET_COL]
    
    # Update lists based on engineered features
    engineered_numeric_cols = [
        "Shipping Pressure Index", "Order Complexity Score", "Discount Risk Score", 
        "Geographical Distance", "Regional Congestion Index", "Route Congestion Index", 
        "Customer Reliability Score", "Historical Shipping Efficiency Gap"
    ]
    engineered_flag_cols = [
        "High Value Order Flag", "Express Shipment Flag", "International Shipment Flag"
    ]
    
    all_numeric_cols = NUMERIC_COLS + engineered_numeric_cols + engineered_flag_cols
    
    # 4. Preprocessing (Scaling and Categorical encoding)
    preprocessor = LogisticsPreprocessor(
        one_hot_cols=ONE_HOT_COLS,
        target_encode_cols=TARGET_ENCODE_COLS,
        numeric_cols=all_numeric_cols
    )
    preprocessor.fit(X_train_raw, y_train)
    
    X_train = preprocessor.transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    
    print(f"Preprocessed Train Shape: {X_train.shape}")
    print(f"Preprocessed Test Shape: {X_test.shape}")
    
    # Define models dictionary
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced'),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=RANDOM_STATE, scale_pos_weight=1.2, n_jobs=-1, eval_metric="logloss"),
        "LightGBM": LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1, verbose=-1),
        "CatBoost": CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, random_state=RANDOM_STATE, auto_class_weights="Balanced", verbose=0)
    }
    
    results = {}
    trained_models = {}
    
    # 5. Model Training & Comparison
    for name, model in models.items():
        print(f"\nTraining {name}...")
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        print(f"{name} trained in {train_time:.2f} seconds.")
        
        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)
        metrics["Training Time (s)"] = train_time
        results[name] = metrics
        trained_models[name] = model
        print(f"  Test F1: {metrics['F1-Score']:.4f} | Recall: {metrics['Recall']:.4f} | PR-AUC: {metrics['PR-AUC']:.4f}")
        
    # Compile comparison table
    comparison_df = pd.DataFrame(results).T
    comparison_df.index.name = "Model"
    print("\nModel Comparison Table:")
    print(comparison_df.to_string())
    
    # Save comparison report
    comparison_df.to_csv(MODEL_COMPARISON_PATH)
    print(f"Model comparison saved to {MODEL_COMPARISON_PATH}")
    
    # 6. Hyperparameter Tuning for the Top Performer (XGBoost / LightGBM)
    # Let's tune LightGBM as it is fast and highly performing
    print("\nPerforming Hyperparameter Tuning on LightGBM...")
    lgbm_param_grid = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.03, 0.1, 0.2],
        'num_leaves': [31, 63, 127],
        'max_depth': [4, 6, 8]
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    
    # Use a subset of training data for faster hyperparameter tuning (20% of data)
    tune_idx = np.random.choice(len(X_train), size=int(len(X_train) * 0.2), replace=False)
    X_train_sub = X_train.iloc[tune_idx]
    y_train_sub = y_train.iloc[tune_idx]
    
    tuner = RandomizedSearchCV(
        estimator=LGBMClassifier(random_state=RANDOM_STATE, class_weight='balanced', verbose=-1),
        param_distributions=lgbm_param_grid,
        n_iter=5,
        cv=cv,
        scoring='average_precision',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    print("Running RandomizedSearchCV...")
    tuner.fit(X_train_sub, y_train_sub)
    print("Best params found:", tuner.best_params_)
    
    # Train final best LightGBM model on full training set
    best_lgbm = LGBMClassifier(**tuner.best_params_, random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1, verbose=-1)
    print("\nTraining final optimized LightGBM model on full training set...")
    best_lgbm.fit(X_train, y_train)
    
    best_metrics = evaluate_model(best_lgbm, X_test, y_test)
    print(f"Optimized LightGBM Test F1: {best_metrics['F1-Score']:.4f} | Recall: {best_metrics['Recall']:.4f} | PR-AUC: {best_metrics['PR-AUC']:.4f}")
    
    # Determine overall best model based on F1-score/Recall trade-off (we will use optimized LGBM as our platform model)
    best_model_name = "Optimized LightGBM"
    best_model = best_lgbm
    
    # 7. Serialize Artifacts
    print(f"\nSerializing best model and preprocessors...")
    joblib.dump(best_model, BEST_MODEL_PATH)
    
    # We will save the feature engineer, preprocessor and column info inside preprocessor.joblib for deployment
    pipeline_components = {
        "feature_engineer": fe,
        "preprocessor": preprocessor,
        "feature_cols": X_train.columns.tolist(),
        "all_numeric_cols": all_numeric_cols
    }
    joblib.dump(pipeline_components, PREPROCESSOR_PATH)
    print("Model pipeline artifacts successfully saved.")
    print("=================== TRAINING PIPELINE COMPLETE ===================")

if __name__ == "__main__":
    run_training_pipeline()
