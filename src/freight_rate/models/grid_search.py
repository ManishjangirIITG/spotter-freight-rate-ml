import itertools
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from freight_rate import config
from freight_rate.data.ingestion import load_train_data
from freight_rate.features.engineering import FeatureEngineerTransformer
from freight_rate.features.preprocessing import DataCleanerTransformer
from freight_rate.models.baseline import calculate_metrics, time_based_split
from freight_rate.models.train import prepare_features_and_target

logger = logging.getLogger(__name__)


def apply_domain_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Adds interaction features to refine market signals."""
    df_ext = df.copy()
    if "market_index" in df_ext.columns and "route_frequency" in df_ext.columns:
        df_ext["market_x_route_freq"] = (
            df_ext["market_index"] * df_ext["route_frequency"]
        )
    if "distance" in df_ext.columns and "weight" in df_ext.columns:
        df_ext["distance_x_weight"] = df_ext["distance"] * df_ext["weight"]
    return df_ext


def run_experiment_grid() -> Dict[str, Any]:
    """Executes a 16-combination grid search across all 4 adjustments."""
    raw_df = load_train_data()

    cleaner = DataCleanerTransformer()
    engineer = FeatureEngineerTransformer()

    processed_df = cleaner.fit_transform(raw_df)
    processed_df = engineer.fit_transform(processed_df)

    # 4 Adjustment Flag Grids
    grid = {
        "objective": ["regression", "huber"],
        "log_transform_target": [False, True],
        "add_domain_interactions": [False, True],
        "tuned_params": [False, True],
    }

    keys, values = zip(*grid.items())
    permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    logger.info(
        f"Starting grid search across {len(permutations)} combinations..."
    )

    results = []
    best_mae = float("inf")
    best_run = None
    best_artifacts = None

    for idx, config_combination in enumerate(permutations, 1):
        df_run = processed_df.copy()

        if config_combination["add_domain_interactions"]:
            df_run = apply_domain_interactions(df_run)

        train_df, val_df = time_based_split(
            df_run, cutoff_date=config.CV_SPLIT_DATE
        )
        X_train, y_train = prepare_features_and_target(train_df)
        X_val, y_val = prepare_features_and_target(val_df)

        cat_cols = [
            c
            for c in config.CATEGORICAL_COLS + ["route_id"]
            if c in X_train.columns
        ]
        for col in cat_cols:
            X_train[col] = X_train[col].astype("category")
            X_val[col] = X_val[col].astype("category")

        # Log transformation handling
        y_train_fit = (
            np.log1p(y_train)
            if config_combination["log_transform_target"]
            else y_train
        )

        lgb_params = {
            "objective": config_combination["objective"],
            "n_estimators": 1000,
            "learning_rate": 0.03,
            "random_state": 42,
            "verbose": -1,
        }

        if config_combination["tuned_params"]:
            lgb_params.update(
                {
                    "num_leaves": 45,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "min_child_samples": 50,
                }
            )
        else:
            lgb_params["num_leaves"] = 31

        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(
            X_train,
            y_train_fit,
            eval_X=X_val,
            eval_y=y_val,
            # eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )

        preds_val = model.predict(X_val)
        if config_combination["log_transform_target"]:
            preds_val = np.expm1(preds_val)

        metrics = calculate_metrics(y_val.values, preds_val)

        run_summary = {
            "run_id": idx,
            "config": config_combination,
            "metrics": metrics,
        }
        results.append(run_summary)

        logger.info(
            f"Run {idx:02d}/16 | Objective: {config_combination['objective']:<10} | LogTarget: {str(config_combination['log_transform_target']):<5} | Interactions: {str(config_combination['add_domain_interactions']):<5} | Tuned: {str(config_combination['tuned_params']):<5} || MAE: ${metrics['mae']:.2f} | RMSE: ${metrics['rmse']:.2f} | MAPE: {metrics['mape']:.2f}%"
        )

        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            best_run = run_summary
            best_artifacts = {
                "cleaner": cleaner,
                "engineer": engineer,
                "model": model,
                "config": config_combination,
            }

    # Save outputs
    metrics_dir = config.ARTIFACTS_DIR / "metrics"
    models_dir = config.ARTIFACTS_DIR / "models"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    with open(metrics_dir / "grid_search_summary.json", "w") as f:
        json.dump(results, f, indent=4)

    # Persist best pipeline model
    joblib.dump(best_artifacts["cleaner"], models_dir / "cleaner_best.joblib")
    joblib.dump(best_artifacts["engineer"], models_dir / "engineer_best.joblib")
    joblib.dump(best_artifacts["model"], models_dir / "lgbm_model_best.joblib")
    
    with open(metrics_dir / "best_model_config.json", "w") as f:
        json.dump(best_run, f, indent=4)

    logger.info(
        f"Grid Search finished. Best MAE: ${best_mae:.2f} (Run ID: {best_run['run_id']})"
    )
    return best_run