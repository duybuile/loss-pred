import logging
from datetime import datetime

import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer


logger = logging.getLogger(__name__)


class SimpleNumericImputer(TransformerMixin):
    """Utilises Sklearn Simple Imputer for dataframes"""

    def __init__(self, **kwargs):
        self.imp = SimpleImputer(**kwargs)
        self.numeric_cols = []

    def fit(self, X: pd.DataFrame):
        self.numeric_cols = list(X.select_dtypes(include=["number"]).columns)
        self.imp.fit(X[self.numeric_cols])
        return self

    def transform(self, X: pd.DataFrame):
        X[self.numeric_cols] = self.imp.transform(X[self.numeric_cols])
        return X


class SimpleDatetimeImputer(TransformerMixin):
    """Replaces all Null values with null_date=1970-01-01"""

    def __init__(self, imputation_value="1970-01-01"):
        self.imputation_value = imputation_value
        self.dt_cols = []

    def fit(self, X: pd.DataFrame):
        self.dt_cols = list(X.select_dtypes(include=["datetime"]).columns)
        return self

    def transform(self, X: pd.DataFrame):
        X[self.dt_cols] = X[self.dt_cols].fillna(
            datetime.strptime(self.imputation_value, "%Y-%m-%d")
        )
        return X


class SimpleCategoryImputer(TransformerMixin):
    """Replaces all None values with imputation_value='None'"""

    def __init__(self, imputation_value="None"):
        self.imputation_value = imputation_value
        self.cat_cols = []

    def fit(self, X: pd.DataFrame):
        self.cat_cols = list(X.select_dtypes(include=["object", "string"]).columns)
        return self

    def transform(self, X: pd.DataFrame):
        X[self.cat_cols] = X[self.cat_cols].fillna(self.imputation_value)
        return X
