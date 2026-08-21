import pandas as pd
import pytest
from freight_rate import config

def test_submission_format():
    pred_path = config.ARTIFACTS_DIR / "predictions" / "validation_predictions.csv"
    assert pred_path.exists(), "Prediction file does not exist."

    df_preds = pd.read_csv(pred_path)
    assert list(df_preds.columns) == [config.ID_COL, "posted_rate"]
    assert len(df_preds) == config.EXPECTED_VAL_ROWS
    assert df_preds["posted_rate"].isna().sum() == 0
    assert (df_preds["posted_rate"] > 0).all()