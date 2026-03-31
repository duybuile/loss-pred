import pytest


VALID_RECORD = {
    "record_id": "NEW_0001",
    "risk_type": "cyber",
    "territory": "EU",
    "industry": "transport",
    "limit": 3438000,
    "premium": 26904,
    "broker": "Meridian Re",
    "prior_claims": 1,
    "years_trading": 4,
}


def test_predict_returns_required_keys():
    from app.model import load_model, load_pipeline, predict

    pipeline = load_pipeline()
    artifact = load_model()
    result = predict(pipeline, artifact, VALID_RECORD)

    assert "is_loss_making_prediction" in result
    assert "probability_of_loss" in result
    assert "top_features" in result
    assert "input_quality_flags" in result
    assert "model_warnings" in result


def test_predict_probability_in_range():
    from app.model import load_model, load_pipeline, predict

    pipeline = load_pipeline()
    artifact = load_model()
    result = predict(pipeline, artifact, VALID_RECORD)

    assert 0.0 <= result["probability_of_loss"] <= 1.0


def test_predict_top_features_is_list_of_strings():
    from app.model import load_model, load_pipeline, predict

    pipeline = load_pipeline()
    artifact = load_model()
    result = predict(pipeline, artifact, VALID_RECORD)

    assert isinstance(result["top_features"], list)
    assert all(isinstance(f, str) for f in result["top_features"])


def test_predict_flags_missing_optional_fields():
    from app.model import load_model, load_pipeline, predict

    pipeline = load_pipeline()
    artifact = load_model()
    sparse_record = {
        "record_id": "NEW_0002",
        "risk_type": "cyber",
        "territory": "EU",
        "limit": 1000000,
        "premium": 10000,
        # missing: broker, industry, prior_claims, years_trading
    }
    result = predict(pipeline, artifact, sparse_record)

    assert any("missing" in flag for flag in result["input_quality_flags"])


def test_predict_flags_missing_required_field():
    from app.model import load_model, load_pipeline, predict

    pipeline = load_pipeline()
    artifact = load_model()
    bad_record = {
        "record_id": "NEW_0003",
        "risk_type": "cyber",
        "territory": "EU",
        # missing: limit, premium
    }
    result = predict(pipeline, artifact, bad_record)

    required_flags = [f for f in result["input_quality_flags"] if "required" in f]
    assert len(required_flags) >= 1
