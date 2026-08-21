import pandas as pd
import pytest
from freight_rate.data.validation import validate_schema, DataValidationError

def test_validate_schema_success():
    data = {
        "load_id": ["LD-1"],
        "pickup": ["NYC"],
        "delivery": ["LAX"],
        "equipment": ["V"],
        "pickup_lat": [40.7128],
        "pickup_lon": [-74.0060],
        "delivery_lat": [34.0522],
        "delivery_lon": [-118.2437],
        "distance": [2800.0],
        "weight": [30000.0],
        "market_index": [1.0],
        "quote_signal": [2.0],
        "date": ["2025-01-01"],
        "posted_rate": [2500.0]
    }
    df = pd.DataFrame(data)
    is_valid, issues = validate_schema(df, is_training=True)
    assert is_valid is True
    assert len(issues) == 0

def test_validate_schema_missing_columns():
    df = pd.DataFrame({"load_id": ["LD-1"]})
    is_valid, issues = validate_schema(df, is_training=True)
    assert is_valid is False
    assert "missing_columns" in issues