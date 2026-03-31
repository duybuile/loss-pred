# tests/test_eval.py
import pytest


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
