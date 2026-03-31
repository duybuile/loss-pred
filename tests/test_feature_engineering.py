# tests/test_feature_engineering.py
import pickle
from pathlib import Path

import pandas as pd
import pytest


def test_transformer_classes_importable():
    from app.feature_engineering import (
        CategoricalEncoder,
        FeatureEngineer,
        LeakageDropper,
        MissingFlagger,
        NumericCoercer,
        NumericImputer,
        OOFGroupEncoder,
        StringCleaner,
    )
    classes = [StringCleaner, NumericCoercer, MissingFlagger, NumericImputer,
               FeatureEngineer, OOFGroupEncoder, CategoricalEncoder, LeakageDropper]
    assert all(cls is not None for cls in classes)


def test_feature_pipeline_unpicklable():
    path = Path(__file__).parent.parent / "app/artifacts/feature_pipeline.pkl"
    assert path.exists(), "feature_pipeline.pkl must exist"
    with path.open("rb") as f:
        pipeline = pickle.load(f)
    assert hasattr(pipeline, "transform")


def test_string_cleaner_strips_whitespace():
    from app.feature_engineering import StringCleaner

    df = pd.DataFrame({"risk_type": [" cyber ", "property "], "limit": [1000, 2000]})
    cleaner = StringCleaner(categorical_cols=["risk_type"])
    cleaner.fit(df)
    result = cleaner.transform(df)
    assert result["risk_type"].tolist() == ["cyber", "property"]


def test_leakage_dropper_removes_column():
    from app.feature_engineering import LeakageDropper

    df = pd.DataFrame({"loss_ratio": [1.2, 0.8], "premium": [1000, 2000]})
    dropper = LeakageDropper(cols_to_drop=["loss_ratio"])
    dropper.fit(df)
    result = dropper.transform(df)
    assert "loss_ratio" not in result.columns
    assert "premium" in result.columns
