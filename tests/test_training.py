import pandas as pd
import numpy as np
from freight_rate import config
from freight_rate.models.baseline import calculate_metrics, HeuristicRatePerMileBaseline
from freight_rate.models.train import train_lightgbm_model

def test_calculate_metrics():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 300.0])
    metrics = calculate_metrics(y_true, y_pred)
    
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "mape" in metrics
    assert metrics["mae"] > 0

def test_heuristic_rpm_baseline():
    X = pd.DataFrame({"distance": [100.0, 200.0]})
    y = pd.Series([200.0, 400.0])  # $2.00 RPM
    
    model = HeuristicRatePerMileBaseline()
    model.fit(X, y)
    preds = model.predict(X)
    
    assert len(preds) == 2
    assert np.isclose(preds[0], 200.0)

def test_lightgbm_training_smoke():
    # Train set
    X_train = pd.DataFrame({
        "distance": [100.0, 200.0, 300.0, 400.0],
        "weight": [20000.0, 30000.0, 25000.0, 35000.0],
        "pickup": ["A", "B", "A", "B"],
        "delivery": ["C", "D", "C", "D"]
    })
    y_train = pd.Series([500.0, 1000.0, 800.0, 1400.0])

    # Distinct Val set to avoid early stopping warning
    X_val = pd.DataFrame({
        "distance": [150.0, 250.0],
        "weight": [22000.0, 32000.0],
        "pickup": ["A", "B"],
        "delivery": ["C", "D"]
    })
    y_val = pd.Series([650.0, 1100.0])
    
    model, metrics = train_lightgbm_model(
        X_train, y_train, X_val, y_val, 
        params={"objective": "regression", "n_estimators": 5, "verbose": -1, "learning_rate": 0.1}
    )
    
    assert model is not None
    assert "mae" in metrics