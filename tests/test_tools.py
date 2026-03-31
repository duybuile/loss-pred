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


def test_dispatch_predict_loss_returns_expected_keys():
    from app.tools import dispatch_tool

    result = dispatch_tool("predict_loss", {"record": VALID_RECORD})

    assert "is_loss_making_prediction" in result
    assert "probability_of_loss" in result
    assert "top_features" in result


def test_dispatch_predict_loss_graceful_on_bad_record(monkeypatch):
    """If predict raises unexpectedly, dispatch_tool returns an error dict, not an exception."""
    from app.tools import dispatch_tool

    def broken_predict(*args, **kwargs):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr("app.tools.predict", broken_predict)

    result = dispatch_tool("predict_loss", {"record": VALID_RECORD})
    assert "error" in result


def test_dispatch_unknown_tool_raises():
    from app.tools import dispatch_tool

    with pytest.raises(ValueError, match="Unknown tool"):
        dispatch_tool("nonexistent_tool", {})
