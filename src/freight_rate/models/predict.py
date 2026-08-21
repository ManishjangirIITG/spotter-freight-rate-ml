import logging
from pathlib import Path
import joblib
import pandas as pd

from freight_rate import config
from freight_rate.data.ingestion import load_validation_data
from freight_rate.logging_config import setup_logging
from freight_rate.models.train import prepare_features_and_target

logger = logging.getLogger(__name__)


import joblib
import numpy as np
import pandas as pd
from freight_rate import config
from freight_rate.data.ingestion import load_validation_data

def generate_validation_predictions(output_path=None):
    if output_path is None:
        output_path = config.ARTIFACTS_DIR / "predictions" / "validation_predictions.csv"

    val_df = load_validation_data()
    
    cleaner = joblib.load(config.ARTIFACTS_DIR / "models" / "cleaner_best.joblib")
    engineer = joblib.load(config.ARTIFACTS_DIR / "models" / "engineer_best.joblib")
    model = joblib.load(config.ARTIFACTS_DIR / "models" / "lgbm_model_best.joblib")

    processed_df = cleaner.transform(val_df)
    processed_df = engineer.transform(processed_df)

    feature_cols = [c for c in processed_df.columns if c not in [config.ID_COL, config.TARGET_COL, "date"]]
    X_val = processed_df[feature_cols].copy()

    cat_cols = [c for c in config.CATEGORICAL_COLS + ["route_id"] if c in X_val.columns]
    for col in cat_cols:
        X_val[col] = X_val[col].astype("category")

    # Inverse transform predictions from log scale
    raw_preds = model.predict(X_val)
    final_preds = np.expm1(raw_preds)

    submission_df = pd.DataFrame({
        config.ID_COL: val_df[config.ID_COL],
        "posted_rate": final_preds
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    return submission_df


if __name__ == "__main__":
    setup_logging("predict.log")
    df_preds = generate_validation_predictions()

    print("\n================ PREDICTION GENERATION SUMMARY ================")
    print(f"Total rows predicted: {len(df_preds)}")
    print(f"Mean predicted rate:  ${df_preds['posted_rate'].mean():.2f}")
    print(f"Min predicted rate:   ${df_preds['posted_rate'].min():.2f}")
    print(f"Max predicted rate:   ${df_preds['posted_rate'].max():.2f}")
    print("\nFirst 5 Predictions:")
    print(df_preds.head())