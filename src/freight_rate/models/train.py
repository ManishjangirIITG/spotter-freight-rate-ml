import joblib
import json
import logging
from datetime import datetime
from typing import Dict, Any, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from freight_rate import config
from freight_rate.data.ingestion import load_train_data
from freight_rate.features.engineering import FeatureEngineerTransformer
from freight_rate.features.preprocessing import DataCleanerTransformer
from freight_rate.logging_config import setup_logging
from freight_rate.models.baseline import calculate_metrics, time_based_split

logger = logging.getLogger(__name__)


def prepare_features_and_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separates features and target variable."""
    drop_cols = [config.TARGET_COL, config.ID_COL, config.DATE_COL]
    drop_cols = [c for c in drop_cols if c in df.columns]

    X = df.drop(columns=drop_cols)
    y = df[config.TARGET_COL] if config.TARGET_COL in df.columns else None
    return X, y


def train_lightgbm_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: Dict[str, Any] = None,
) -> Tuple[lgb.LGBMRegressor, Dict[str, float]]:
    """Trains a LightGBM regressor with categorical support and returns evaluation metrics."""
    if params is None:
        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "n_estimators": 1000,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "random_state": 42,
            "verbose": -1,
        }

    # Ensure categorical features are explicitly formatted
    cat_features = [
        c
        for c in config.CATEGORICAL_COLS + ["route_id"]
        if c in X_train.columns
    ]
    for col in cat_features:
        X_train[col] = X_train[col].astype("category")
        X_val[col] = X_val[col].astype("category")

    logger.info(f"Training LightGBM model with features: {list(X_train.columns)}")

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train,
        y_train,
        eval_X=X_val,
        eval_y=y_val,
        # eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )

    preds_val = model.predict(X_val)
    metrics = calculate_metrics(y_val.values, preds_val)
    logger.info(f"LightGBM Val Metrics -> MAE: ${metrics['mae']:.2f} | RMSE: ${metrics['rmse']:.2f} | MAPE: {metrics['mape']:.2f}%")

    return model, metrics


def train_pipeline() -> Tuple[Pipeline, lgb.LGBMRegressor, Dict[str, float]]:
    """Executes end-to-end training pipeline on time-based train/val split."""
    raw_df = load_train_data()

    # 1. Fit Preprocessors
    cleaner = DataCleanerTransformer()
    engineer = FeatureEngineerTransformer()

    processed_df = cleaner.fit_transform(raw_df)
    processed_df = engineer.fit_transform(processed_df)

    # 2. Time-Based Split
    train_df, val_df = time_based_split(
        processed_df, cutoff_date=config.CV_SPLIT_DATE
    )

    X_train, y_train = prepare_features_and_target(train_df)
    X_val, y_val = prepare_features_and_target(val_df)

    # 3. Train Model
    model, metrics = train_lightgbm_model(X_train, y_train, X_val, y_val)

    # 4. Save Artifacts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = config.ARTIFACTS_DIR / "models"
    metrics_dir = config.ARTIFACTS_DIR / "metrics"
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(cleaner, model_dir / f"cleaner_{timestamp}.joblib")
    joblib.dump(engineer, model_dir / f"engineer_{timestamp}.joblib")
    joblib.dump(model, model_dir / f"lgbm_model_{timestamp}.joblib")

    # Save latest aliases for easy inference pipeline loading
    joblib.dump(cleaner, model_dir / "cleaner_latest.joblib")
    joblib.dump(engineer, model_dir / "engineer_latest.joblib")
    joblib.dump(model, model_dir / "lgbm_model_latest.joblib")

    with open(metrics_dir / f"metrics_{timestamp}.json", "w") as f:
        json.dump(metrics, f, indent=4)

    logger.info(f"Artifacts successfully exported to {config.ARTIFACTS_DIR}")
    return model, metrics


if __name__ == "__main__":
    setup_logging("train_pipeline.log")
    train_pipeline()