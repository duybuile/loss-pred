# app/model.py
"""
Model loader — loads trained artifacts and runs inference on a single record.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import app.feature_engineering  # noqa: F401 — registers transformer classes for unpickling

_MODEL_PATH = Path(__file__).parent / "artifacts" / "model.pkl"
_PIPELINE_PATH = Path(__file__).parent / "artifacts" / "feature_pipeline.pkl"

_REQUIRED_FIELDS = {"risk_type", "territory", "limit", "premium"}
_OPTIONAL_FIELDS = {"broker", "industry", "prior_claims", "years_trading"}


def load_model() -> dict:
    """Load the model artifact dict from app/artifacts/model.pkl."""
    if not _MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {_MODEL_PATH}. "
            "Complete the modelling notebook and save app/artifacts/model.pkl first."
        )
    with _MODEL_PATH.open("rb") as f:
        return pickle.load(f)


def load_pipeline() -> Any:
    """Load the feature pipeline from app/artifacts/feature_pipeline.pkl."""
    if not _PIPELINE_PATH.exists():
        raise FileNotFoundError(
            f"Pipeline artifact not found at {_PIPELINE_PATH}. "
            "Complete the modelling notebook and save app/artifacts/feature_pipeline.pkl first."
        )
    with _PIPELINE_PATH.open("rb") as f:
        return pickle.load(f)


def get_top_features(model: Any, feature_columns: list[str], n: int = 5) -> list[str]:
    """
    Return the top n most influential feature names.
    Handles both tree-based (feature_importances_) and linear (coef_) models.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return []
    ranked = sorted(zip(feature_columns, importances), key=lambda x: x[1], reverse=True)
    return [name for name, _ in ranked[:n]]


def _validate_record(record: dict) -> list[str]:
    """Return a list of input quality flags for the given record dict."""
    flags = []
    for field in sorted(_REQUIRED_FIELDS):
        if record.get(field) is None:
            flags.append(f"required_field_missing:{field}")
    for field in sorted(_OPTIONAL_FIELDS):
        if record.get(field) is None:
            flags.append(f"optional_field_missing:{field}")
    if record.get("years_trading") is not None and record["years_trading"] < 0:
        flags.append("invalid_value:years_trading_negative")
    if record.get("limit") is not None and record["limit"] <= 0:
        flags.append("invalid_value:limit_nonpositive")
    if record.get("premium") is not None and record["premium"] <= 0:
        flags.append("invalid_value:premium_nonpositive")
    return flags


def predict(pipeline: Any, artifact: dict, record: dict) -> dict:
    """
    Run the feature pipeline and model on a single record dict.

    Args:
        pipeline: Loaded feature pipeline from load_pipeline().
        artifact: Loaded model artifact dict from load_model().
        record:   A dict matching the fields in records.csv (without loss_ratio / is_loss_making).

    Returns:
        {
            "is_loss_making_prediction": bool,
            "probability_of_loss": float,
            "top_features": list[str],
            "input_quality_flags": list[str],
            "model_warnings": list[str],
        }
    """
    input_quality_flags = _validate_record(record)
    model_warnings: list[str] = []

    # Fill missing numeric fields with np.nan so the pipeline's NumericImputer
    # can handle them rather than crashing on missing column references.
    # Use float np.nan (not None/pd.NA) so comparisons like (col < 0) yield
    # False rather than pd.NA, which would cause .astype(int) to fail.
    _numeric_fields = {"limit", "premium", "prior_claims", "years_trading"}
    _all_fields = _REQUIRED_FIELDS | _OPTIONAL_FIELDS
    padded_record = {}
    for field in _all_fields:
        val = record.get(field)
        if val is None and field in _numeric_fields:
            padded_record[field] = np.nan
        else:
            padded_record[field] = val
    padded_record.update({k: v for k, v in record.items() if k not in _all_fields})

    df = pd.DataFrame([padded_record])
    X = pipeline.transform(df)
    feature_columns: list[str] = artifact["feature_columns"]
    # Keep only columns the model was trained on; fill any extras with 0
    missing_cols = [c for c in feature_columns if c not in X.columns]
    if missing_cols:
        model_warnings.append(f"feature_columns_missing_after_transform:{missing_cols}")
    for col in missing_cols:
        X[col] = 0
    X = X[feature_columns]

    model = artifact["model"]
    proba = float(model.predict_proba(X)[0, 1])
    prediction = proba >= 0.5
    top_features = get_top_features(model, feature_columns)

    return {
        "is_loss_making_prediction": bool(prediction),
        "probability_of_loss": round(proba, 4),
        "top_features": top_features,
        "input_quality_flags": input_quality_flags,
        "model_warnings": model_warnings,
    }
