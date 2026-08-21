import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from freight_rate import config
from freight_rate.data.ingestion import load_train_data
from freight_rate.features.preprocessing import DataCleanerTransformer
from freight_rate.features.engineering import FeatureEngineerTransformer

logger = logging.getLogger(__name__)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculates regression metrics."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred) * 100)
    }


def time_based_split(df: pd.DataFrame, cutoff_date: str = config.CV_SPLIT_DATE) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits dataset temporally into train and internal validation sets."""
    df_sorted = df.sort_values(config.DATE_COL).reset_index(drop=True)
    train_mask = pd.to_datetime(df_sorted[config.DATE_COL]) < pd.to_datetime(cutoff_date)
    
    train_df = df_sorted[train_mask].reset_index(drop=True)
    val_df = df_sorted[~train_mask].reset_index(drop=True)
    
    logger.info(f"Time-based split at {cutoff_date} -> Train rows: {len(train_df)}, Val rows: {len(val_df)}")
    return train_df, val_df


class HeuristicRatePerMileBaseline:
    """Simple baseline predicting target based on median training rate-per-mile * distance."""
    
    def __init__(self):
        self.median_rpm_ = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series):
        rpm = y / (X["distance"] + 1e-5)
        self.median_rpm_ = float(rpm.median())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (X["distance"] * self.median_rpm_).values


def run_baseline_experiments() -> Dict[str, Dict[str, float]]:
    """Runs baseline experiments on time-based validation split."""
    raw_df = load_train_data()
    
    # 1. Pipeline Feature Engineering
    cleaner = DataCleanerTransformer()
    engineer = FeatureEngineerTransformer()
    
    processed_df = cleaner.fit_transform(raw_df)
    processed_df = engineer.fit_transform(processed_df)
    
    # 2. Temporal Split
    train_df, val_df = time_based_split(processed_df, cutoff_date=config.CV_SPLIT_DATE)
    
    X_train, y_train = train_df.drop(columns=[config.TARGET_COL, config.ID_COL]), train_df[config.TARGET_COL]
    X_val, y_val = val_df.drop(columns=[config.TARGET_COL, config.ID_COL]), val_df[config.TARGET_COL]
    
    results = {}

    # --- Baseline 1: Heuristic Rate-per-Mile ---
    heuristic = HeuristicRatePerMileBaseline()
    heuristic.fit(X_train, y_train)
    preds_h = heuristic.predict(X_val)
    results["Heuristic_RPM"] = calculate_metrics(y_val.values, preds_h)

    # --- Baseline 2: Ridge Regression ---
    num_cols = ["distance", "weight", "market_index", "quote_signal", "lat_delta", "lon_delta", "great_circle_distance"]
    cat_cols = ["pickup", "delivery", "equipment"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ]
    )
    
    ridge_pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", Ridge(alpha=1.0))
    ])
    
    ridge_pipeline.fit(X_train, y_train)
    preds_ridge = ridge_pipeline.predict(X_val)
    results["Ridge_Regression"] = calculate_metrics(y_val.values, preds_ridge)

    return results


if __name__ == "__main__":
    from freight_rate.logging_config import setup_logging
    setup_logging(log_filename="baseline_experiment.log")
    
    metrics = run_baseline_experiments()
    print("\n================ BASELINE EVALUATION RESULTS ================")
    for model_name, res in metrics.items():
        print(f"Model: {model_name:<20} | MAE: ${res['mae']:.2f} | RMSE: ${res['rmse']:.2f} | MAPE: {res['mape']:.2f}%")