import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from freight_rate import config

logger = logging.getLogger(__name__)


def haversine_distance(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Calculates great-circle distance (in miles) between coordinate arrays."""
    R = 3958.8  # Earth radius in miles
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


class FeatureEngineerTransformer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compliant transformer for domain-specific feature extraction,
    geographic metrics, temporal encodings, and route interaction features.
    """

    def __init__(self):
        self.route_frequencies_ = {}

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Learns categorical frequency encodings on training data."""
        X_df = X.copy()
        if "pickup" in X_df.columns and "delivery" in X_df.columns:
            routes = X_df["pickup"] + "__" + X_df["delivery"]
            freqs = routes.value_counts(normalize=True).to_dict()
            self.route_frequencies_ = freqs
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies feature engineering transformations."""
        X_df = X.copy()

        # 1. Geographic & Spatial Features
        if all(
            col in X_df.columns
            for col in [
                "pickup_lat",
                "pickup_lon",
                "delivery_lat",
                "delivery_lon",
            ]
        ):
            X_df["lat_delta"] = (
                X_df["delivery_lat"] - X_df["pickup_lat"]
            ).abs()
            X_df["lon_delta"] = (
                X_df["delivery_lon"] - X_df["pickup_lon"]
            ).abs()

            gc_dist = haversine_distance(
                X_df["pickup_lat"].values,
                X_df["pickup_lon"].values,
                X_df["delivery_lat"].values,
                X_df["delivery_lon"].values,
            )
            X_df["great_circle_distance"] = gc_dist

            if "distance" in X_df.columns:
                # Avoid division by zero
                X_df["distance_ratio"] = X_df["distance"] / (gc_dist + 1.0)

        # 2. Route & Categorical Encodings
        if "pickup" in X_df.columns and "delivery" in X_df.columns:
            X_df["route_id"] = X_df["pickup"] + "__" + X_df["delivery"]
            X_df["route_frequency"] = X_df["route_id"].map(
                self.route_frequencies_
            ).fillna(0.0)

        # 3. Domain Interactions
        if "weight" in X_df.columns and "distance" in X_df.columns:
            X_df["weight_per_mile"] = X_df["weight"] / (
                X_df["distance"] + 1.0
            )

        # 4. Temporal & Cyclical Features
        if config.DATE_COL in X_df.columns:
            date_s = pd.to_datetime(X_df[config.DATE_COL])
            X_df["day_of_week"] = date_s.dt.dayofweek
            X_df["day_of_month"] = date_s.dt.day
            X_df["month"] = date_s.dt.month
            X_df["is_weekend"] = date_s.dt.dayofweek.isin([5, 6]).astype(int)
            X_df["day_of_year"] = date_s.dt.dayofyear

            # Cyclical Day of Week
            X_df["sin_day_of_week"] = np.sin(
                2 * np.pi * X_df["day_of_week"] / 7.0
            )
            X_df["cos_day_of_week"] = np.cos(
                2 * np.pi * X_df["day_of_week"] / 7.0
            )

            # Cyclical Day of Year
            X_df["sin_day_of_year"] = np.sin(
                2 * np.pi * X_df["day_of_year"] / 365.25
            )
            X_df["cos_day_of_year"] = np.cos(
                2 * np.pi * X_df["day_of_year"] / 365.25
            )

        return X_df