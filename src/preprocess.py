import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.config import RANDOM_STATE

class BayesianTargetEncoder:
    """
    A robust Bayesian target encoder with additive smoothing to prevent overfitting on low-count categories.
    Formula: (count * mean + smoothing * global_mean) / (count + smoothing)
    """
    def __init__(self, cols, smoothing=10.0):
        self.cols = cols
        self.smoothing = smoothing
        self.mappings = {}
        self.global_mean = 0.5

    def fit(self, X, y):
        self.global_mean = y.mean()
        for col in self.cols:
            stats = pd.DataFrame({"feat": X[col], "target": y}).groupby("feat")["target"].agg(["count", "mean"])
            counts = stats["count"]
            means = stats["mean"]
            
            # Bayesian smoothed mapping
            smoothed_vals = (counts * means + self.smoothing * self.global_mean) / (counts + self.smoothing)
            self.mappings[col] = smoothed_vals.to_dict()
        return self

    def transform(self, X):
        X_out = X.copy()
        for col in self.cols:
            mapping = self.mappings[col]
            X_out[col] = X_out[col].map(mapping).fillna(self.global_mean)
        return X_out

class LogisticsPreprocessor:
    def __init__(self, one_hot_cols, target_encode_cols, numeric_cols, smoothing=10.0):
        self.one_hot_cols = one_hot_cols
        self.target_encode_cols = target_encode_cols
        self.numeric_cols = numeric_cols
        self.smoothing = smoothing
        
        self.scaler = StandardScaler()
        self.target_encoder = BayesianTargetEncoder(cols=self.target_encode_cols, smoothing=self.smoothing)
        self.ohe_categories = {}
        self.feature_names = []

    def fit(self, X, y):
        # 1. Fit Target Encoder
        self.target_encoder.fit(X, y)
        
        # 2. Fit Numeric Scaler
        self.scaler.fit(X[self.numeric_cols])
        
        # 3. Fit One-Hot Categories
        for col in self.one_hot_cols:
            # Drop null values and get unique categories sorted
            cats = sorted(X[col].dropna().unique())
            self.ohe_categories[col] = cats
            
        # Compile feature names for the output DataFrame
        self.feature_names = []
        # Target encoded columns retain their names
        self.feature_names.extend(self.target_encode_cols)
        # Scaled numeric columns retain their names
        self.feature_names.extend(self.numeric_cols)
        # One-hot encoded columns expand
        for col in self.one_hot_cols:
            for cat in self.ohe_categories[col]:
                self.feature_names.append(f"{col}_{cat}")
                
        return self

    def transform(self, X):
        X_out = X.copy()
        
        # 1. Apply Target Encoding
        X_te = self.target_encoder.transform(X_out[self.target_encode_cols])
        
        # 2. Apply Standard Scaling
        X_scaled_vals = self.scaler.transform(X_out[self.numeric_cols])
        X_scaled = pd.DataFrame(X_scaled_vals, columns=self.numeric_cols, index=X_out.index)
        
        # 3. Apply One-Hot Encoding
        ohe_dfs = []
        for col in self.one_hot_cols:
            cats = self.ohe_categories[col]
            # Construct a DataFrame of binary columns
            col_dummies = pd.DataFrame(0, index=X_out.index, columns=[f"{col}_{cat}" for cat in cats])
            for cat in cats:
                col_dummies.loc[X_out[col] == cat, f"{col}_{cat}"] = 1
            ohe_dfs.append(col_dummies)
            
        # Concatenate all parts
        df_processed = pd.concat([X_te, X_scaled] + ohe_dfs, axis=1)
        
        # Ensure column ordering is correct and match feature_names
        df_processed = df_processed[self.feature_names]
        
        return df_processed
