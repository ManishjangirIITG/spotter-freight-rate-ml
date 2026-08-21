import joblib
import pandas as pd
from freight_rate import config
from freight_rate.models.predict import generate_validation_predictions

def test_latest_artifacts_exist():
    model_dir = config.ARTIFACTS_DIR / "models"
    assert (model_dir / "cleaner_latest.joblib").exists()
    assert (model_dir / "engineer_latest.joblib").exists()
    assert (model_dir / "lgbm_model_latest.joblib").exists()

def test_generate_validation_predictions_execution(tmp_path):
    output_file = tmp_path / "test_preds.csv"
    df_preds = generate_validation_predictions(output_path=output_file)
    
    assert output_file.exists()
    assert len(df_preds) == config.EXPECTED_VAL_ROWS
    assert list(df_preds.columns) == [config.ID_COL, "posted_rate"]