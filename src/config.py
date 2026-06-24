import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# File paths
RAW_DATA_PATH = os.path.join(DATA_DIR, "APL_logisticsdata_risk_prediction.csv")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.joblib")
MODEL_COMPARISON_PATH = os.path.join(REPORTS_DIR, "model_comparison.csv")

# Target Variable
TARGET_COL = "Late_delivery_risk"

# Target Leakage Columns & Identifiers to Drop
LEAKAGE_COLS = [
    "Days for shipping (real)",
    "Delivery Status",
    "Order Status"  # Highly correlated with cancel/fraud status and changes post-facto
]

IDENTIFIER_COLS = [
    "Customer Fname",
    "Customer Lname",
    "Customer Street",
    "Order Customer Id",
    "Customer Zipcode"
]

COLS_TO_DROP = LEAKAGE_COLS + IDENTIFIER_COLS

# Feature categorization for preprocessing
ONE_HOT_COLS = [
    "Type",
    "Customer Segment",
    "Shipping Mode",
    "Market"
]

TARGET_ENCODE_COLS = [
    "Category Name",
    "Customer City",
    "Customer State",
    "Department Name",
    "Order City",
    "Order Country",
    "Order Region",
    "Order State",
    "Product Name"
]

NUMERIC_COLS = [
    "Days for shipment (scheduled)",
    "Benefit per order",
    "Sales per customer",
    "Latitude",
    "Longitude",
    "Order Item Discount",
    "Order Item Discount Rate",
    "Order Item Product Price",
    "Order Item Profit Ratio",
    "Order Item Quantity",
    "Sales",
    "Order Item Total",
    "Order Profit Per Order"
]

# Random State
RANDOM_STATE = 42
TEST_SIZE = 0.2
