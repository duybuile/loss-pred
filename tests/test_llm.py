import json
import pytest


def test_get_client_anthropic(monkeypatch):
    import anthropic
    from app.llm import get_client

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    client = get_client("anthropic")
    assert isinstance(client, anthropic.Anthropic)


def test_get_client_openai(monkeypatch):
    import openai
    from app.llm import get_client

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = get_client("openai")
    assert isinstance(client, openai.OpenAI)


def test_get_client_unknown_raises():
    from app.llm import get_client

    with pytest.raises(ValueError, match="Unknown provider"):
        get_client("gemini")


def test_llm_adapter_synthesize_anthropic(monkeypatch):
    """LLMAdapter.synthesize() calls Anthropic and returns parsed dict."""
    import anthropic
    from app.llm import LLMAdapter

    fake_json = json.dumps({
        "recommendation": "Decline this record.",
        "summary": "High risk.",
        "key_factors": ["prior_claims", "premium_to_limit"],
        "similar_records_summary": None,
        "review_guidance": "Refer to senior underwriter.",
    })

    class FakeTextBlock:
        text = fake_json

    class FakeResponse:
        content = [FakeTextBlock()]

    class FakeMessages:
        def create(self, **kwargs):
            return FakeResponse()

    class FakeClient:
        messages = FakeMessages()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr("app.llm.get_client", lambda provider: FakeClient())

    adapter = LLMAdapter(provider="anthropic", model="claude-sonnet-4-6", max_tokens=2048)
    result = adapter.synthesize(evidence={"foo": "bar"}, system_prompt="You are helpful.")

    assert result["recommendation"] == "Decline this record."
    assert isinstance(result["key_factors"], list)
