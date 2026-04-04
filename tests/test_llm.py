import json
import pytest


def test_llm_client_call_anthropic(monkeypatch):
    from utils.llms.llm_client import LLMClient

    fake_json = json.dumps({"answer": "ok"})

    class FakeTextBlock:
        text = fake_json

    class FakeResponse:
        content = [FakeTextBlock()]
        usage = None

    captured = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return FakeResponse()

    class FakeAnthropicClient:
        def __init__(self, api_key):
            captured["api_key"] = api_key
            self.messages = FakeMessages()

    monkeypatch.setattr("utils.llms.llm_client.Anthropic", FakeAnthropicClient)

    client = LLMClient(api="anthropic", model="claude-sonnet-4-6", api_key="test-key")
    result = client.call(
        input=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": '{"foo":"bar"}'},
        ],
        max_tokens=2048,
    )

    assert result == fake_json
    assert captured["api_key"] == "test-key"
    assert captured["model"] == "claude-sonnet-4-6"
    assert captured["max_tokens"] == 2048
    assert captured["system"] == "You are helpful."
    assert captured["messages"] == [{"role": "user", "content": '{"foo":"bar"}'}]


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
    """LLMAdapter.synthesize() calls the shared LLM client and returns parsed dict."""
    from app.llm import LLMAdapter

    fake_json = json.dumps({
        "recommendation": "Decline this record.",
        "summary": "High risk.",
        "key_factors": ["prior_claims", "premium_to_limit"],
        "similar_records_summary": None,
        "review_guidance": "Refer to senior underwriter.",
    })

    captured = {}

    class FakeLLMClient:
        def __init__(self, api, model, api_key):
            captured["init"] = {
                "api": api,
                "model": model,
                "api_key": api_key,
            }

        def call(self, **kwargs):
            captured["call"] = kwargs
            return fake_json

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr("app.llm.LLMClient", FakeLLMClient)

    adapter = LLMAdapter(provider="anthropic", model="claude-sonnet-4-6", max_tokens=2048)
    result = adapter.synthesize(evidence={"foo": "bar"}, system_prompt="You are helpful.")

    assert result["recommendation"] == "Decline this record."
    assert isinstance(result["key_factors"], list)
    assert captured["init"] == {
        "api": "anthropic",
        "model": "claude-sonnet-4-6",
        "api_key": "test-key",
    }
    assert captured["call"]["max_tokens"] == 2048
    assert captured["call"]["input"] == [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": json.dumps({"foo": "bar"}, indent=2)},
    ]
