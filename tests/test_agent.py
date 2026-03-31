# tests/test_agent.py
import json
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

SYNTHESIS_RESPONSE = {
    "recommendation": "Decline this record.",
    "summary": "High risk based on prior claims and premium rate.",
    "key_factors": ["prior_claims", "premium_to_limit"],
    "similar_records_summary": None,
    "review_guidance": "Refer to senior underwriter.",
}


def test_run_agent_returns_required_keys(monkeypatch):
    from app import agent as agent_module

    monkeypatch.setattr(
        agent_module,
        "_adapter",
        _make_fake_adapter(SYNTHESIS_RESPONSE),
    )

    result = agent_module.run_agent(VALID_RECORD)

    for key in ["recommendation", "risk_assessment", "confidence_level",
                "key_factors", "summary", "second_opinion_recommended",
                "review_guidance", "tools_used", "warnings"]:
        assert key in result, f"Missing key: {key}"


def test_run_agent_escalation_skips_llm(monkeypatch):
    """When escalation is forced, the LLM adapter should not be called."""
    from app import agent as agent_module

    call_count = {"n": 0}

    class TrackingAdapter:
        def synthesize(self, *args, **kwargs):
            call_count["n"] += 1
            return SYNTHESIS_RESPONSE

    monkeypatch.setattr(agent_module, "_adapter", TrackingAdapter())

    # Force escalation by providing a near-boundary record with missing fields
    sparse_record = {"record_id": "X", "risk_type": "cyber", "territory": "EU"}
    result = agent_module.run_agent(sparse_record)

    # With required fields missing → confidence forced low → escalation → no LLM
    assert call_count["n"] == 0
    assert result["second_opinion_recommended"] is True


def test_run_agent_tools_used_populated(monkeypatch):
    from app import agent as agent_module

    monkeypatch.setattr(agent_module, "_adapter", _make_fake_adapter(SYNTHESIS_RESPONSE))
    result = agent_module.run_agent(VALID_RECORD)

    assert "predict_loss" in result["tools_used"]


def _make_fake_adapter(synthesis_response: dict):
    class FakeAdapter:
        def synthesize(self, *args, **kwargs):
            return synthesis_response
    return FakeAdapter()
