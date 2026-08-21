import json
import logging
from typing import Dict, Any
import joblib
import numpy as np
import pandas as pd

from freight_rate import config
from freight_rate.data.ingestion import load_train_data
from freight_rate.logging_config import setup_logging
from freight_rate.models.baseline import calculate_metrics, time_based_split
from freight_rate.models.train import prepare_features_and_target

logger = logging.getLogger(__name__)


def evaluate_model_performance() -> Dict[str, Any]:
    """Loads latest model artifacts, evaluates on validation split, logs metrics, and exports report."""
    model_dir = config.ARTIFACTS_DIR / "models"
    metrics_dir = config.ARTIFACTS_DIR / "metrics"
    
    # 1. Load latest serialized artifacts
    cleaner = joblib.load(model_dir / "cleaner_latest.joblib")
    engineer = joblib.load(model_dir / "engineer_latest.joblib")
    model = joblib.load(model_dir / "lgbm_model_latest.joblib")

    # 2. Ingest and process raw data
    raw_df = load_train_data()
    processed_df = cleaner.transform(raw_df)
    processed_df = engineer.transform(processed_df)

    # 3. Apply time-based validation split
    _, val_df = time_based_split(processed_df, cutoff_date=config.CV_SPLIT_DATE)
    X_val, y_val = prepare_features_and_target(val_df)

    # 4. Generate predictions and calculate metrics
    for col in config.CATEGORICAL_COLS + ["route_id"]:
        if col in X_val.columns:
            X_val[col] = X_val[col].astype("category")

    preds = model.predict(X_val)
    eval_metrics = calculate_metrics(y_val.values, preds)

    # Log explicit evaluation metrics to file
    logger.info("=== VALIDATION METRICS EVALUATION ===")
    logger.info(f"Validation MAE:  ${eval_metrics['mae']:.2f}")
    logger.info(f"Validation RMSE: ${eval_metrics['rmse']:.2f}")
    logger.info(f"Validation MAPE: {eval_metrics['mape']:.2f}%")

    # 5. Extract Feature Importances
    feature_names = list(X_val.columns)
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    top_10 = importance_df.head(10).to_dict(orient="records")
    
    logger.info("=== TOP 10 FEATURE IMPORTANCES ===")
    for feat in top_10:
        logger.info(f"Feature: {feat['feature']:<25} | Importance: {feat['importance']}")

    report = {
        "metrics": eval_metrics,
        "top_10_features": top_10
    }

    # 6. Export report JSON
    report_path = metrics_dir / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    logger.info(f"Evaluation report successfully saved to {report_path}")
    return report


if __name__ == "__main__":
    setup_logging("evaluate.log")
    report = evaluate_model_performance()
    
    print("\n================ MODEL EVALUATION SUMMARY ================")
    print(f"Validation MAE:  ${report['metrics']['mae']:.2f}")
    print(f"Validation RMSE: ${report['metrics']['rmse']:.2f}")
    print(f"Validation MAPE: {report['metrics']['mape']:.2f}%\n")
    print("Top 10 Feature Importances:")
    for feat in report["top_10_features"]:
        print(f"  - {feat['feature']:<25}: {feat['importance']}")