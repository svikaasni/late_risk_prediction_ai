import os
import joblib
import pandas as pd
import numpy as np
import shap
from src.config import BEST_MODEL_PATH, PREPROCESSOR_PATH, MODEL_DIR

def compute_and_save_shap():
    """
    Loads the serialized model and preprocessor, runs a sample of training data,
    computes SHAP values, and serializes the SHAP Explainer for the Streamlit dashboard.
    """
    print("Computing SHAP values for dashboard visualizations...")
    if not os.path.exists(BEST_MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError("Model or preprocessor artifacts are missing. Run training first.")
        
    model = joblib.load(BEST_MODEL_PATH)
    pipeline = joblib.load(PREPROCESSOR_PATH)
    
    # Load raw data and preprocess to get training features
    from src.data_loader import load_raw_data, preprocess_raw_data, get_train_test_split
    raw_df = load_raw_data()
    clean_df = preprocess_raw_data(raw_df)
    train_df, _ = get_train_test_split(clean_df, split_method="temporal")
    
    fe = pipeline["feature_engineer"]
    preprocessor = pipeline["preprocessor"]
    
    train_fe = fe.transform(train_df)
    
    from src.config import COLS_TO_DROP, TARGET_COL
    cols_to_drop = [c for c in COLS_TO_DROP if c in train_fe.columns]
    X_train_raw = train_fe.drop(columns=cols_to_drop + [TARGET_COL, "Order Date"], errors="ignore")
    y_train = train_fe[TARGET_COL]
    
    # Preprocess
    X_train = preprocessor.transform(X_train_raw)
    
    # Use a representative sample of 200 observations to construct the TreeExplainer
    # TreeExplainer is extremely fast for LightGBM
    print("Initializing SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model, data=shap.sample(X_train, 200, random_state=42))
    
    explainer_path = os.path.join(MODEL_DIR, "shap_explainer.joblib")
    joblib.dump(explainer, explainer_path)
    print(f"SHAP Explainer saved to {explainer_path}")
    
    # Also pre-compute SHAP values on a test sample for global importance plots
    print("Pre-computing SHAP values on test sample...")
    test_sample = X_train.sample(n=500, random_state=42)
    shap_values = explainer(test_sample)
    
    shap_data = {
        "shap_values": shap_values,
        "test_sample": test_sample
    }
    shap_values_path = os.path.join(MODEL_DIR, "shap_values_sample.joblib")
    joblib.dump(shap_data, shap_values_path)
    print(f"SHAP values sample saved to {shap_values_path}")

def get_natural_language_explanation(prob, shap_values_row, feature_names):
    """
    Translates SHAP values for a single order into a natural language description
    designed for logistics operations managers.
    """
    risk_level = "LOW"
    if prob >= 0.7:
        risk_level = "HIGH"
    elif prob >= 0.3:
        risk_level = "MEDIUM"
        
    base_value = shap_values_row.base_values
    if isinstance(base_value, np.ndarray):
        base_value = base_value[0]
        
    shap_vals = shap_values_row.values
    
    # Pair feature names with SHAP values
    feature_impacts = []
    for i, name in enumerate(feature_names):
        impact = shap_vals[i]
        feature_impacts.append((name, impact))
        
    # Sort impacts: positive drivers (increase delay risk) and negative drivers (decrease risk)
    positive_drivers = sorted([x for x in feature_impacts if x[1] > 0], key=lambda x: x[1], reverse=True)
    negative_drivers = sorted([x for x in feature_impacts if x[1] < 0], key=lambda x: x[1])
    
    # Format top drivers
    top_pos = []
    for name, imp in positive_drivers[:3]:
        # User-friendly name renaming
        clean_name = name.replace("_", " ").title()
        top_pos.append(f"• **{clean_name}** (+{imp*100:.1f}% risk increase)")
        
    top_neg = []
    for name, imp in negative_drivers[:2]:
        clean_name = name.replace("_", " ").title()
        top_neg.append(f"• **{clean_name}** ({imp*100:.1f}% risk reduction)")
        
    pos_text = "\n".join(top_pos) if top_pos else "None"
    neg_text = "\n".join(top_neg) if top_neg else "None"
    
    explanation = f"""### Late Delivery Risk Assessment: **{risk_level} RISK** ({prob*100:.1f}% Confidence)

**Primary Factors Increasing Delay Risk:**
{pos_text}

**Primary Factors Mitigating Delay Risk:**
{neg_text}

**Operational Summary:**
This order is classified as **{risk_level}** risk with a predicted delay probability of **{prob*100:.1f}%**. """
    
    if risk_level == "HIGH":
        explanation += "Operational teams should immediately review the suggested interventions in the Operations Control Tower, such as upgrading the shipping mode or verifying customs documents for international routing."
    elif risk_level == "MEDIUM":
        explanation += "Monitoring is recommended. If this is a high-value customer, consider proactive shipping mode upgrade or customer communication."
    else:
        explanation += "No immediate action required. Order is on track to arrive within the scheduled shipping window."
        
    return explanation

def explain_single_order(raw_order_dict, model, pipeline_components, explainer):
    """
    Computes predicted probability and SHAP values for a single raw order row
    and returns a structured explanation dictionary.
    """
    fe = pipeline_components["feature_engineer"]
    preprocessor = pipeline_components["preprocessor"]
    feature_cols = pipeline_components["feature_cols"]
    
    # 1. Convert dict to DataFrame
    df_raw = pd.DataFrame([raw_order_dict])
    
    # 2. Apply Feature Engineering
    df_fe = fe.transform(df_raw)
    
    # 3. Apply Preprocessing
    # Drop columns not used by preprocessor
    from src.config import COLS_TO_DROP
    cols_to_drop = [c for c in COLS_TO_DROP if c in df_fe.columns]
    X_raw_input = df_fe.drop(columns=cols_to_drop + ["Late_delivery_risk", "Order Date"], errors="ignore")
    
    X_processed = preprocessor.transform(X_raw_input)
    
    # 4. Predict
    prob = model.predict_proba(X_processed)[0, 1]
    
    # 5. Compute SHAP Values for this single row
    shap_values = explainer(X_processed)
    shap_row = shap_values[0]
    
    # Get natural language explanation
    explanation = get_natural_language_explanation(prob, shap_row, feature_cols)
    
    return {
        "probability": prob,
        "risk_level": "High" if prob >= 0.7 else "Medium" if prob >= 0.3 else "Low",
        "explanation": explanation,
        "shap_values": shap_row,
        "processed_features": X_processed
    }
