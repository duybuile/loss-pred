import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin


class TimeFeaturesTransformer(TransformerMixin):
    """Creating several time-based features"""

    def __init__(self, dt_cols=None):
        self.dt_cols = dt_cols

    @staticmethod
    def _time_features(X: pd.DataFrame, columns):
        for c in columns:
            X[f"{c}__hour_of_day"] = X[c].dt.hour.values
            X[f"{c}__day_of_month"] = X[c].dt.day.values
            X[f"{c}__day_of_week"] = X[c].dt.dayofweek.values
            X[f"{c}__week_of_year"] = X[c].dt.week.values
            X[f"{c}__month_of_year"] = X[c].dt.month.values

            # cyclical daytime features -
            # see https://ianlondon.github.io/blog/encoding-cyclical-features-24hour-time/
            h = X[c].dt.hour + X[c].dt.minute / 60
            d = X[c].dt.day + X[c].dt.hour / 24

            X[f"{c}__hour_sin"] = np.sin(2 * np.pi * h / 24)
            X[f"{c}__hour_cos"] = np.cos(2 * np.pi * h / 24)
            X[f"{c}__day_sin"] = np.sin(2 * np.pi * d / 365)
            X[f"{c}__day_cos"] = np.cos(2 * np.pi * d / 365)

            # cyclical month features
            m = X[c].dt.month + X[c].dt.day / 30 - 1
            X[f"{c}__month_sin"] = np.sin(2 * np.pi * m / 12)
            X[f"{c}__month_cos"] = np.cos(2 * np.pi * m / 12)

        return X

    def transform(self, X):
        """runs all datetime feature generation"""
        dt_cols = (
            self.dt_cols if self.dt_cols else X.select_dtypes(include=["datetime"])
        )
        return self._time_features(X, dt_cols)


class TimeFeatureExtraction:
    def __init__(self, dt_col: str):
        self.dt_col = dt_col

    @staticmethod
    def _time_of_day(hour: int):
        if 6 < hour < 18:
            return "day"
        else:
            return "night"

    def time_of_day(self, df: pd.DataFrame):
        df[f"{self.dt_col}_hour_of_day"] = df[self.dt_col].dt.hour.values
        df[f"time_of_day_{self.dt_col}"] = [
            self._time_of_day(t) for t in df[f"{self.dt_col}_hour_of_day"]
        ]
        return df.drop(columns=[f"{self.dt_col}_hour_of_day"])

    @staticmethod
    def extract_age(dob: pd.Series, then: pd.Series):
        if not then:
            then = pd.Timestamp("now")
        return (then - dob).astype("<m8[Y]")
