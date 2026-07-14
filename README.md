# 🚚 APL Logistics Decision Intelligence Platform

An end-to-end Machine Learning and Generative AI system designed to predict, explain, and mitigate late delivery risks in global supply chains. The platform leverages advanced predictive models (LightGBM) to forecast delays, explainable AI (SHAP) to interpret risk drivers, prescriptive heuristics to recommend operational interventions, and an LLM Copilot (Gemini) to query logistics database metrics in natural language.

---

## 🌟 Key Features

### 1. 📊 Executive Overview
High-level KPIs summarizing total orders, weighted delay risk rate, overall **Revenue at Risk (RAR)**, and average delay probabilities. Includes dynamic charts tracking SLA compliance trends.

### 2. 🎮 Operations Control Tower
An operational dashboard to prioritize and triage pending shipments. Orders can be sorted by **Intervention Score** (balancing revenue value, delay risk, and scheduled buffer) to optimize operational efficiency.

### 3. 🔍 Order Risk Prediction & Explainable AI (XAI)
Look up specific order IDs to view their delay risk classification (Low, Medium, or High). Includes:
*   **SHAP Force Plots** showing exactly which features drove the model's decision.
*   **Natural Language Explanations** designed for logistics managers.

### 4. 🧪 What-If Simulator
Allows coordinators to adjust scheduled shipping days, sales value, and shipping mode in real time to simulate how changing variables impacts the late delivery risk before dispatch.

### 5. 💬 Generative AI Supply Chain Copilot
A Gemini-powered digital assistant connected directly to the active database. Query regional congestion, carrier performance, or request a triage action plan for high-risk accounts in natural language.

### 6. 🗺️ Global Risk Map
An interactive destination map of pending shipments, sizing data points by transaction value and coloring them by predicted delay risk level.

---

## 🛠️ Directory Structure

```text
├── app/
│   ├── app.py                      # Main Streamlit web application
│   └── utils.py                    # Helper utilities & Gemini LLM Copilot query engine
├── data/
│   ├── APL_logisticsdata_risk_prediction.csv # Raw historical dataset (gitignored)
│   └── dashboard_sample.csv        # Pre-processed representative test sample (2,000 rows)
├── models/                         # Serialized pipeline components
│   ├── best_model.joblib           # Trained LightGBM Classifier model
│   ├── preprocessor.joblib         # Column transformers and scaling pipeline
│   ├── shap_explainer.joblib       # Serialized SHAP TreeExplainer
│   └── shap_values_sample.joblib   # Pre-computed SHAP values for global dashboard plots
├── reports/
│   ├── drift_report.json           # Simulated data drift statistics
│   └── model_comparison.csv       # Training metrics for evaluated classifiers
├── src/                            # Machine learning pipeline modules
│   ├── config.py                   # Global constants and file paths
│   ├── data_loader.py              # Data loading, cleaning, and temporal split
│   ├── explainability.py           # SHAP computation modules
│   ├── features.py                 # Logistics feature engineering logic
│   ├── mlops.py                    # Production drift monitoring simulation
│   ├── preprocess.py               # One-hot/Target encoders and scalers
│   ├── prescriptive.py             # Heuristics for operational recommendations
│   └── train.py                    # Multi-model training and tuning execution
├── requirements.txt                # Python package dependencies
├── run_post_training.py            # Utility to generate SHAP values and run drift monitoring
└── generate_dashboard_sample.py    # Utility to compile the dashboard test sample
```

---

## 🚀 Quick Start Guide

### 1. Environment Setup
Clone the repository and set up a Python virtual environment:

```bash
# Clone the repository
git clone https://github.com/svikaasni/late_risk_prediction_ai.git
cd late_risk_prediction_ai

# Set up virtual environment
python -m venv venv

# Activate (PowerShell)
.\venv\Scripts\Activate.ps1
# Activate (CMD)
.\venv\Scripts\activate.bat
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Add Gemini API Credentials
To enable the **Generative AI Copilot**, create a `.streamlit/secrets.toml` file in the root directory:

```toml
GEMINI_API_KEY = "your_gemini_api_key_from_google_ai_studio"
```

### 4. Run the Streamlit Dashboard
```bash
streamlit run app/app.py
```

---

## 🤖 Running the ML Pipeline (Optional)

If you want to retrain the model, recalculate explainability values, or generate a fresh dashboard sample from raw data, use these commands:

### Retrain Models
Evaluates 6 different classifiers (Logistic Regression, Decision Trees, Random Forests, XGBoost, LightGBM, CatBoost), tunes the best performer (LightGBM), and serializes the updated pipeline artifacts to `models/`:
```bash
python -m src.train
```

### Run Post-Training Tasks
Recalculates SHAP explainers and runs the production monitoring simulation (which outputs data drift reports to `reports/drift_report.json`):
```bash
python run_post_training.py
```

### Regenerate Dashboard Data Sample
Extracts a fresh sample of 2,000 pending orders from the out-of-time test set, makes predictions, and saves them to `data/dashboard_sample.csv`:
```bash
python generate_dashboard_sample.py
```
