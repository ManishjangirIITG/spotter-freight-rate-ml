import logging
from pathlib import Path
import joblib
import pandas as pd

from freight_rate import config
from freight_rate.data.ingestion import load_validation_data
from freight_rate.logging_config import setup_logging
from freight_rate.models.train import prepare_features_and_target

logger = logging.getLogger(__name__)


def generate_validation_predictions(
    output_path: Path = None,
) -> pd.DataFrame:
    """Loads latest trained model pipeline, processes external validation data,
    and exports formatted rate predictions."""
    if output_path is None:
        output_path = (
            config.ARTIFACTS_DIR / "predictions" / "validation_predictions.csv"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load latest trained artifacts
    model_dir = config.ARTIFACTS_DIR / "models"
    logger.info("Loading latest serialized pipeline artifacts...")
    cleaner = joblib.load(model_dir / "cleaner_latest.joblib")
    engineer = joblib.load(model_dir / "engineer_latest.joblib")
    model = joblib.load(model_dir / "lgbm_model_latest.joblib")

    # 2. Load external validation target data
    raw_val_df = load_validation_data()

    # 3. Apply feature transformation sequence
    processed_df = cleaner.transform(raw_val_df)
    processed_df = engineer.transform(processed_df)

    X_val, _ = prepare_features_and_target(processed_df)

    # Convert categorical variables
    for col in config.CATEGORICAL_COLS + ["route_id"]:
        if col in X_val.columns:
            X_val[col] = X_val[col].astype("category")

    # 4. Generate rate predictions
    logger.info("Generating predictions on validation set...")
    predictions = model.predict(X_val)

    # 5. Format output
    pred_df = pd.DataFrame(
        {config.ID_COL: raw_val_df[config.ID_COL], "posted_rate": predictions}
    )

    # Ensure output row count matches expected contract
    if len(pred_df) != config.EXPECTED_VAL_ROWS:
        logger.warning(
            f"Prediction row count ({len(pred_df)}) mismatch with expected target ({config.EXPECTED_VAL_ROWS})"
        )

    pred_df.to_csv(output_path, index=False)
    logger.info(
        f"Validation predictions exported to {output_path} | Shape: {pred_df.shape}"
    )

    return pred_df


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