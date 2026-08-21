import pandas as pd
import numpy as np
from freight_rate.features.preprocessing import DataCleanerTransformer
from freight_rate.features.engineering import FeatureEngineerTransformer, haversine_distance

def test_data_cleaner_transformer():
    data = {
        "weight": [-45000.0, np.nan, 30000.0],
        "market_index": [1.0, np.nan, 1.2],
        "date": ["2025-01-01", "2025-01-02", "2025-01-03"]
    }
    df = pd.DataFrame(data)
    cleaner = DataCleanerTransformer()
    cleaned_df = cleaner.fit_transform(df)

    assert (cleaned_df["weight"] >= 0).all()
    assert cleaned_df["weight_is_negative"].sum() == 1
    assert cleaned_df["weight_is_missing"].sum() == 1
    assert cleaned_df["market_index"].isna().sum() == 0

def test_haversine_distance():
    # Known distance approx: NYC to LAX ~ 2445 miles
    dist = haversine_distance(
        np.array([40.7128]), np.array([-74.0060]),
        np.array([34.0522]), np.array([-118.2437])
    )
    assert 2400 < dist[0] < 2500