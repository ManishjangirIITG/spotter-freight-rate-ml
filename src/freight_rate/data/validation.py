import logging
import pandas as pd
from typing import Dict, Any, Tuple
from freight_rate import config

logger = logging.getLogger(__name__)

class DataValidationError(Exception):
    """Custom exception raised when dataset contract or validation rules fail."""
    pass

def validate_schema(df: pd.DataFrame, is_training: bool = True) -> Tuple[bool, Dict[str, Any]]:
    """
    Validates dataset schema, presence of mandatory columns, duplicate IDs, 
    and logs operational anomalies like negative weights.
    """
    issues = {}
    
    # 1. Required column presence
    required_cols = [config.ID_COL] + config.CATEGORICAL_COLS + config.NUMERICAL_COLS + [config.DATE_COL]
    if is_training:
        required_cols.append(config.TARGET_COL)
        
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        issues["missing_columns"] = missing_cols

    # 2. Duplicate ID validation
    if config.ID_COL in df.columns:
        duplicate_ids = df[config.ID_COL].duplicated().sum()
        if duplicate_ids > 0:
            issues["duplicate_ids"] = int(duplicate_ids)

    # 3. Anomaly flagging (does not halt execution unless critical)
    if "weight" in df.columns:
        negative_weights = (df["weight"] < 0).sum()
        if negative_weights > 0:
            issues["negative_weights_count"] = int(negative_weights)

    if is_training and config.TARGET_COL in df.columns:
        non_positive_targets = (df[config.TARGET_COL] <= 0).sum()
        if non_positive_targets > 0:
            issues["non_positive_targets"] = int(non_positive_targets)

    # Hard contract failure if missing required columns or duplicate IDs exist
    is_valid = len([k for k in issues.keys() if k in ["missing_columns", "duplicate_ids"]]) == 0
    return is_valid, issues