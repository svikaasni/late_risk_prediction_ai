import os
import joblib
import pandas as pd
from src.config import BEST_MODEL_PATH, PREPROCESSOR_PATH, DATA_DIR
from src.data_loader import load_raw_data, preprocess_raw_data, get_train_test_split
from app.utils import predict_df_risks

def main():
    print("Generating pre-processed dashboard sample for Streamlit Cloud deployment...")
    
    # 1. Load model and pipeline
    if not os.path.exists(BEST_MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError("ML pipeline components are missing. Train the model first.")
        
    model = joblib.load(BEST_MODEL_PATH)
    pipeline = joblib.load(PREPROCESSOR_PATH)
    fe = pipeline["feature_engineer"]
    
    # 2. Load and split raw data
    raw_df = load_raw_data()
    clean_df = preprocess_raw_data(raw_df)
    _, test_df = get_train_test_split(clean_df, split_method="temporal")
    
    # 3. Take a representative sample of 2,000 rows from the test set
    print("Sampling 2,000 records from the out-of-time test set...")
    sample_df = test_df.sample(n=min(2000, len(test_df)), random_state=42).copy()
    
    # 4. Apply Feature Engineering
    print("Engineering features on the sample...")
    sample_fe = fe.transform(sample_df)
    
    # 5. Batch predict risks and compute operational metrics
    print("Predicting delay risks and calculating operational scores...")
    sample_predicted = predict_df_risks(sample_fe, model, pipeline)
    
    # 6. Save to CSV
    sample_path = os.path.join(DATA_DIR, "dashboard_sample.csv")
    sample_predicted.to_csv(sample_path, index=False)
    print(f"Successfully generated and saved dashboard sample to {sample_path}")
    print(f"Sample size: {sample_predicted.shape[0]} rows, {sample_predicted.shape[1]} columns.")

if __name__ == "__main__":
    main()
