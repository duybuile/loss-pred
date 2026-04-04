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

    monkeypatch.setattr(agent_module, "_adapter", _make_fake_adapter(SYNTHESIS_RESPONSE))
    monkeypatch.setattr(agent_module._controller, "compute_confidence", lambda pred: "high")
    monkeypatch.setattr(agent_module._controller, "should_escalate", lambda **kw: False)

    result = agent_module.run_agent(VALID_RECORD)

    for key in ["recommendation", "risk_assessment", "confidence_level",
                "key_factors", "summary", "similar_records_summary",
                "second_opinion_recommended", "review_guidance", "tools_used", "warnings"]:
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
    monkeypatch.setattr(agent_module._controller, "compute_confidence", lambda pred: "low")
    monkeypatch.setattr(agent_module._controller, "should_escalate", lambda **kw: True)

    sparse_record = {"record_id": "X", "risk_type": "cyber", "territory": "EU"}
    result = agent_module.run_agent(sparse_record)

    assert call_count["n"] == 0
    assert result["second_opinion_recommended"] is True


def test_run_agent_tools_used_populated(monkeypatch):
    from app import agent as agent_module

    monkeypatch.setattr(agent_module, "_adapter", _make_fake_adapter(SYNTHESIS_RESPONSE))
    monkeypatch.setattr(agent_module._controller, "compute_confidence", lambda pred: "high")
    monkeypatch.setattr(agent_module._controller, "should_escalate", lambda **kw: False)

    result = agent_module.run_agent(VALID_RECORD)

    assert "predict_loss" in result["tools_used"]


def test_retrieve_similar_records_not_in_tools_used_when_high_confidence(monkeypatch):
    """Retrieval tool should not appear in tools_used when confidence is high."""
    from app import agent as agent_module

    monkeypatch.setattr(agent_module, "_adapter", _make_fake_adapter(SYNTHESIS_RESPONSE))
    # Force high confidence so retrieval is skipped
    monkeypatch.setattr(agent_module._controller, "compute_confidence", lambda pred: "high")
    monkeypatch.setattr(agent_module._controller, "should_escalate", lambda **kw: False)

    result = agent_module.run_agent(VALID_RECORD)

    assert "retrieve_similar_records" not in result["tools_used"]


def test_load_system_prompt_uses_configured_base_name_and_version(tmp_path, monkeypatch):
    from app import agent as agent_module

    prompt_dir = tmp_path / "prompt"
    prompt_dir.mkdir()
    prompt_file = prompt_dir / "orchestrator_v99.txt"
    prompt_file.write_text("prompt from file")

    monkeypatch.setattr(
        agent_module.cfg,
        "get",
        lambda key: {
            "prompt.orchestrator": "prompt/orchestrator.txt",
            "prompt.orchestrator_version": "v99",
        }[key],
    )

    assert agent_module._load_system_prompt(repo_root=tmp_path) == "prompt from file"


def _make_fake_adapter(synthesis_response: dict):
    class FakeAdapter:
        def synthesize(self, *args, **kwargs):
            return synthesis_response
    return FakeAdapter()
