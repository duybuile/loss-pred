# app/feature_engineering/transformers.py
"""
Sklearn transformer classes for the loss-prediction feature pipeline.

These classes are used both in the modelling notebook (notebooks/modelling.ipynb)
and at inference time in the app. Defining them here makes the serialised
feature_pipeline.pkl loadable outside the notebook context.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder


class StringCleaner(BaseEstimator, TransformerMixin):
    """Trim whitespace in all string columns and fill missing categoricals with a sentinel."""

    def __init__(self, categorical_cols, fill_value="Unknown"):
        self.categorical_cols = categorical_cols
        self.fill_value = fill_value

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X, y=None):
        X = X.copy()
        for col in X.select_dtypes(include=["object", "string"]).columns:
            X[col] = X[col].astype("string").str.strip()
        for col in self.categorical_cols:
            if col in X.columns:
                X[col] = (
                    X[col]
                    .astype("string")
                    .str.strip()
                    .replace({"": pd.NA, "nan": pd.NA, "<na>": pd.NA})
                    .fillna(self.fill_value)
                )
        return X


class NumericCoercer(BaseEstimator, TransformerMixin):
    """Coerce specified columns to numeric; non-parseable values become NaN."""

    def __init__(self, numeric_cols):
        self.numeric_cols = numeric_cols

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X, y=None):
        X = X.copy()
        for col in self.numeric_cols:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors="coerce")
        return X


class MissingFlagger(BaseEstimator, TransformerMixin):
    """Add binary flag columns recording missingness and known invalid values before imputation."""

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X, y=None):
        X = X.copy()
        if "prior_claims" in X.columns:
            X["prior_claims_missing"] = X["prior_claims"].isna().astype(int)
        if "years_trading" in X.columns:
            X["years_trading_invalid"] = (X["years_trading"] < 0).astype(int)
            X["years_trading_missing"] = X["years_trading"].isna().astype(int)
        return X


class NumericImputer(BaseEstimator, TransformerMixin):
    """
    Impute numeric columns using training-set statistics learned in fit().

    - prior_claims  : fill NaN with 0, clip negatives to 0.
    - years_trading : cap negatives at 0, fill NaN with training median.
    - premium/limit : fill NaN with training median; floor only.
    """

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        self.years_trading_median_ = (
            X["years_trading"].clip(lower=0).median() if "years_trading" in X.columns else 0.0
        )
        self.premium_median_ = X["premium"].median() if "premium" in X.columns else 0.0
        self.limit_median_ = X["limit"].median() if "limit" in X.columns else 1.0
        return self

    def transform(self, X, y=None):
        X = X.copy()
        if "prior_claims" in X.columns:
            X["prior_claims"] = X["prior_claims"].fillna(0).clip(lower=0)
        if "years_trading" in X.columns:
            X["years_trading"] = (
                X["years_trading"].clip(lower=0).fillna(self.years_trading_median_)
            )
        if "premium" in X.columns:
            X["premium"] = X["premium"].fillna(self.premium_median_).clip(lower=0)
        if "limit" in X.columns:
            X["limit"] = X["limit"].fillna(self.limit_median_).clip(lower=1)
        return X


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Create rate-style, log-scale, and ordinal features from cleaned columns."""

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X["premium_to_limit"] = X["premium"] / X["limit"]
        X["log_limit"] = np.log1p(X["limit"])
        X["log_premium"] = np.log1p(X["premium"])
        X["claims_per_year"] = X["prior_claims"] / X["years_trading"].clip(lower=1)
        X["experience_band"] = pd.cut(
            X["years_trading"],
            bins=[-0.1, 3, 10, 25, float("inf")],
            labels=["startup", "early", "established", "mature"],
        ).astype(str)
        return X


class OOFGroupEncoder(BaseEstimator, TransformerMixin):
    """
    Leakage-safe group-mean encoder driven by loss_ratio.

    fit_transform (training): each row's score is the mean loss_ratio of its group
    from the other K-1 folds. transform (inference): applies stored group means.
    """

    def __init__(
        self,
        group_col,
        score_col,
        loss_ratio_col="loss_ratio",
        n_splits=5,
        random_state=42,
    ):
        self.group_col = group_col
        self.score_col = score_col
        self.loss_ratio_col = loss_ratio_col
        self.n_splits = n_splits
        self.random_state = random_state

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        self.global_mean_ = X[self.loss_ratio_col].mean()
        self.group_means_ = X.groupby(self.group_col)[self.loss_ratio_col].mean()
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X[self.score_col] = X[self.group_col].map(self.group_means_).fillna(self.global_mean_)
        return X

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        X = X.copy()
        oof = pd.Series(np.nan, index=X.index)
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for fold_train_idx, fold_val_idx in kf.split(X):
            fold_train = X.iloc[fold_train_idx]
            fold_val = X.iloc[fold_val_idx]
            fold_means = fold_train.groupby(self.group_col)[self.loss_ratio_col].mean()
            fold_global = fold_train[self.loss_ratio_col].mean()
            oof.iloc[fold_val_idx] = fold_val[self.group_col].map(fold_means).fillna(fold_global)
        X[self.score_col] = oof
        return X


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Encode categorical features for model consumption.

    - Nominal (risk_type, territory): OneHotEncoding with drop='first'.
    - Ordered (experience_band): mapped to integer ranks 0-3.
    """

    def __init__(self, nominal_cols, ordinal_col_map):
        self.nominal_cols = nominal_cols
        self.ordinal_col_map = ordinal_col_map

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        try:
            self.ohe_ = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
            self.ohe_.fit(X[self.nominal_cols].astype(str))
        except TypeError:
            self.ohe_ = OneHotEncoder(drop="first", sparse=False, handle_unknown="ignore")
            self.ohe_.fit(X[self.nominal_cols].astype(str))
        self.ordinal_maps_ = {
            col: {cat: i for i, cat in enumerate(cats)}
            for col, cats in self.ordinal_col_map.items()
        }
        return self

    def transform(self, X, y=None):
        X = X.copy()
        ohe_values = self.ohe_.transform(X[self.nominal_cols].astype(str))
        ohe_col_names = self.ohe_.get_feature_names_out(self.nominal_cols)
        ohe_df = pd.DataFrame(ohe_values, columns=ohe_col_names, index=X.index)
        X = X.drop(columns=self.nominal_cols)
        X = pd.concat([X, ohe_df], axis=1)
        for col, mapping in self.ordinal_maps_.items():
            if col in X.columns:
                X[col] = X[col].map(mapping).fillna(-1).astype(int)
        return X


class LeakageDropper(BaseEstimator, TransformerMixin):
    """Remove columns that would leak the target into the feature matrix."""

    def __init__(self, cols_to_drop=("loss_ratio",)):
        self.cols_to_drop = list(cols_to_drop)

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X, y=None):
        return X.drop(columns=[c for c in self.cols_to_drop if c in X.columns])
