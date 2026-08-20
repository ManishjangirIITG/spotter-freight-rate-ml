from pathlib import Path

# Base Paths
PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"

# Dataset File Paths
TRAIN_PATH = DATA_DIR / "train-test.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
PRED_TEMPLATE_PATH = DATA_DIR / "validation-predictions-template.csv"
DECEMBER_PATH = DATA_DIR / "december-chart-inputs.csv"

# Target & Primary ID
TARGET_COL = "posted_rate"
ID_COL = "load_id"

# Column Groups
CATEGORICAL_COLS = ["pickup", "delivery", "equipment"]
NUMERICAL_COLS = [
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "weight",
    "market_index",
    "quote_signal",
]
DATE_COL = "date"

# Validation Requirements
EXPECTED_VAL_ROWS = 12000
EXPECTED_DECEMBER_ROWS = 31

# Internal Time-Series Split Cutoff (Jan-Aug for Train, Sep-Oct for Internal Val)
CV_SPLIT_DATE = "2025-09-01"