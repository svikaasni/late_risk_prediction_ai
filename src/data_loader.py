import os
import shutil
import pandas as pd
import numpy as np
from src.config import RAW_DATA_PATH, TARGET_COL, COLS_TO_DROP, RANDOM_STATE, TEST_SIZE

def organize_dataset_file():
    """
    Moves the dataset from the workspace root to the data/ directory if needed.
    """
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root_csv_path = os.path.join(workspace_root, "APL_logisticsdata_risk_prediction.csv")
    
    if os.path.exists(root_csv_path) and not os.path.exists(RAW_DATA_PATH):
        print(f"Moving dataset from {root_csv_path} to {RAW_DATA_PATH}...")
        os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
        shutil.move(root_csv_path, RAW_DATA_PATH)
        print("Dataset successfully organized.")
    elif os.path.exists(RAW_DATA_PATH):
        print("Dataset already in correct data/ directory.")
    else:
        print("Warning: Dataset file not found.")

def load_raw_data():
    """
    Loads raw supply chain logistics dataset and returns a pandas DataFrame.
    """
    organize_dataset_file()
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {RAW_DATA_PATH}")
    
    print("Loading raw logistics dataset...")
    df = pd.read_csv(RAW_DATA_PATH)
    return df

def preprocess_raw_data(df):
    """
    Performs initial data cleaning and temporal date synthesis.
    """
    print("Performing initial cleaning and temporal date simulation...")
    df_clean = df.copy()
    
    # Impute missing values
    df_clean["Customer Lname"] = df_clean["Customer Lname"].fillna("Unknown")
    df_clean["Customer Zipcode"] = df_clean["Customer Zipcode"].fillna(0).astype(int)
    
    # Synthesize chronological Order Date spanning 2 years (2024-2025)
    # This enables seasonality, temporal splitting, and forecasting
    df_clean["Order Date"] = pd.date_range(start="2024-01-01", end="2025-12-31", periods=len(df_clean))
    df_clean["Year"] = df_clean["Order Date"].dt.year
    df_clean["Month"] = df_clean["Order Date"].dt.month
    df_clean["Day"] = df_clean["Order Date"].dt.day
    df_clean["DayOfWeek"] = df_clean["Order Date"].dt.dayofweek
    df_clean["Hour"] = 12  # Dummy hour for time-series context if needed
    
    return df_clean

def get_train_test_split(df, split_method="temporal"):
    """
    Splits the dataset into training and testing sets.
    - 'temporal': Out-of-time split (Train on 2024 to Oct 2025, Test on Nov-Dec 2025)
    - 'random': Standard stratified random split
    """
    if split_method == "temporal":
        print("Splitting dataset temporally (Out-of-time validation)...")
        cutoff_date = pd.Timestamp("2025-11-01")
        train_mask = df["Order Date"] < cutoff_date
        
        train_df = df[train_mask].reset_index(drop=True)
        test_df = df[~train_mask].reset_index(drop=True)
        
        print(f"Train set size: {len(train_df)} (before Nov 2025)")
        print(f"Test set size: {len(test_df)} (Nov-Dec 2025)")
    else:
        print("Splitting dataset randomly (Stratified split)...")
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(
            df, 
            test_size=TEST_SIZE, 
            random_state=RANDOM_STATE, 
            stratify=df[TARGET_COL]
        )
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        
    return train_df, test_df
