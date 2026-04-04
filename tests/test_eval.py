# tests/test_eval.py
import pytest


def test_call_judge_llm_uses_anthropic_provider(monkeypatch):
    from evals import eval as eval_module

    monkeypatch.setattr(eval_module, "JUDGE_PROVIDER", "anthropic")
    monkeypatch.setattr(eval_module, "JUDGE_MODEL", "claude-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    captured = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return type(
                "Response",
                (),
                {"content": [type("Chunk", (), {"text": '{"outcome_aligned": true}'})()]},
            )()

    class FakeAnthropicClient:
        def __init__(self, api_key):
            captured["api_key"] = api_key
            self.messages = FakeMessages()

    monkeypatch.setattr(eval_module.anthropic, "Anthropic", FakeAnthropicClient)

    result = eval_module._call_judge_llm(
        recommendation="Decline this record.",
        ground_truth={"is_loss_making": True, "loss_ratio": 1.3, "case_type": "clear"},
    )

    assert result["outcome_aligned"] is True
    assert captured["api_key"] == "test-key"
    assert captured["model"] == "claude-test"
    assert captured["max_tokens"] == eval_module.JUDGE_MAX_TOKENS
    assert captured["system"] == eval_module._JUDGE_SYSTEM_PROMPT
    assert captured["messages"][0]["role"] == "user"


def test_call_judge_llm_uses_openai_provider(monkeypatch):
    from evals import eval as eval_module

    monkeypatch.setattr(eval_module, "JUDGE_PROVIDER", "openai")
    monkeypatch.setattr(eval_module, "JUDGE_MODEL", "gpt-test")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return type(
                "Response",
                (),
                {
                    "choices": [
                        type(
                            "Choice",
                            (),
                            {
                                "message": type(
                                    "Message",
                                    (),
                                    {"content": '{"outcome_aligned": false}'},
                                )()
                            },
                        )()
                    ]
                },
            )()

    class FakeOpenAIClient:
        def __init__(self, api_key):
            captured["api_key"] = api_key
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr(eval_module.openai, "OpenAI", FakeOpenAIClient)

    result = eval_module._call_judge_llm(
        recommendation="Accept this record.",
        ground_truth={"is_loss_making": False, "loss_ratio": 0.7, "case_type": "clear"},
    )

    assert result["outcome_aligned"] is False
    assert captured["api_key"] == "test-key"
    assert captured["model"] == "gpt-test"
    assert captured["max_tokens"] == eval_module.JUDGE_MAX_TOKENS
    assert captured["messages"][0]["role"] == "system"
    assert captured["messages"][1]["role"] == "user"


def test_call_judge_llm_rejects_unknown_provider(monkeypatch):
    from evals import eval as eval_module

    monkeypatch.setattr(eval_module, "JUDGE_PROVIDER", "unsupported")

    with pytest.raises(ValueError, match="Unknown judge provider"):
        eval_module._call_judge_llm(
            recommendation="Review this record.",
            ground_truth={"is_loss_making": True, "loss_ratio": 1.0, "case_type": "borderline"},
        )


def test_score_recommendation_correct_outcome(monkeypatch):
    """Judge returns correct=True when recommendation aligns with ground truth."""
    from evals import eval as eval_module

    fake_score = {
        "correct": True,
        "outcome_aligned": True,
        "calibration_ok": True,
        "actionable": True,
        "grounded": True,
        "safe_deferral": True,
        "reasoning": "Recommendation correctly identifies loss-making record.",
    }
    monkeypatch.setattr(eval_module, "_call_judge_llm", lambda *a, **kw: fake_score)

    ground_truth = {"is_loss_making": True, "loss_ratio": 1.5, "case_type": "clear"}
    result = eval_module.score_recommendation(
        recommendation="This record is very likely to result in a loss. Decline.",
        ground_truth=ground_truth,
    )

    assert result["correct"] is True
    assert "reasoning" in result


def test_score_recommendation_wrong_outcome(monkeypatch):
    from evals import eval as eval_module

    fake_score = {
        "correct": False,
        "outcome_aligned": False,
        "calibration_ok": False,
        "actionable": True,
        "grounded": False,
        "safe_deferral": False,
        "reasoning": "Recommendation says unlikely loss but record was loss-making.",
    }
    monkeypatch.setattr(eval_module, "_call_judge_llm", lambda *a, **kw: fake_score)

    ground_truth = {"is_loss_making": True, "loss_ratio": 1.5, "case_type": "clear"}
    result = eval_module.score_recommendation(
        recommendation="This record is unlikely to result in a loss. Accept.",
        ground_truth=ground_truth,
    )

    assert result["correct"] is False


def test_score_correct_requires_both_aligned_and_calibrated(monkeypatch):
    """correct=True only if both outcome_aligned AND calibration_ok are True."""
    from evals import eval as eval_module

    fake_score = {
        "correct": True,      # judge initially marks correct
        "outcome_aligned": True,
        "calibration_ok": False,  # but calibration failed
        "actionable": True,
        "grounded": True,
        "safe_deferral": True,
        "reasoning": "Got direction right but overconfident on borderline case.",
    }
    monkeypatch.setattr(eval_module, "_call_judge_llm", lambda *a, **kw: fake_score)

    ground_truth = {"is_loss_making": True, "loss_ratio": 1.01, "case_type": "borderline"}
    result = eval_module.score_recommendation(
        recommendation="This is definitely going to be a loss. Decline immediately.",
        ground_truth=ground_truth,
    )

    # score_recommendation enforces: correct = outcome_aligned AND calibration_ok
    assert result["correct"] is False
