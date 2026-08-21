import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from freight_rate import config

logger = logging.getLogger(__name__)


class DataCleanerTransformer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compliant transformer that performs deterministic data cleaning,
    anomaly handling, and missing value imputation without target leakage.
    """

    def __init__(self):
        self.medians_ = {}

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Calculates medians on training data for numerical imputation."""
        X_df = X.copy()

        # Learn median of cleaned weight (ignoring negative/NaN)
        valid_weights = X_df["weight"].apply(
            lambda w: abs(w) if pd.notna(w) else np.nan
        )
        self.medians_["weight"] = float(valid_weights.median())

        # Learn median of market_index
        if "market_index" in X_df.columns:
            self.medians_["market_index"] = float(
                X_df["market_index"].median()
            )

        logger.info(f"Fitted cleaner medians: {self.medians_}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies data cleaning and learned imputations."""
        X_df = X.copy()

        # 1. Weight cleaning & missingness indicators
        if "weight" in X_df.columns:
            X_df["weight_is_missing"] = (
                X_df["weight"].isna().astype(int)
            )
            X_df["weight_is_negative"] = (
                (X_df["weight"] < 0).fillna(False).astype(int)
            )

            # Convert negative weight to absolute value
            X_df["weight"] = X_df["weight"].abs()

            # Impute missing weight with learned training median
            X_df["weight"] = X_df["weight"].fillna(
                self.medians_.get("weight", 31000.0)
            )

        # 2. Market Index cleaning & indicator
        if "market_index" in X_df.columns:
            X_df["market_index_is_missing"] = (
                X_df["market_index"].isna().astype(int)
            )
            X_df["market_index"] = X_df["market_index"].fillna(
                self.medians_.get("market_index", 1.0)
            )

        # 3. Ensure proper temporal ordering
        if config.DATE_COL in X_df.columns:
            X_df[config.DATE_COL] = pd.to_datetime(X_df[config.DATE_COL])

        return X_df