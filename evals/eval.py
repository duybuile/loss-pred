# evals/eval.py
"""
Evaluation harness for the assessment agent.

Usage:
    uv run python evals/eval.py
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import anthropic
import openai
from dotenv import load_dotenv

load_dotenv()

# ── Parameters ────────────────────────────────────────────────────────────────
JUDGE_PROVIDER = "anthropic"  # "anthropic" | "openai"
JUDGE_MODEL = "claude-haiku-4-5-20251001" if JUDGE_PROVIDER == "anthropic" else "gpt-5.4-mini"
JUDGE_MAX_TOKENS = 512

EVAL_SET_PATH = Path(__file__).parent / "eval_set.json"

_JUDGE_SYSTEM_PROMPT = """You are evaluating an AI assessment agent that reviews insurance records.

You will receive:
1. A free-text recommendation produced by the agent
2. The ground truth: is_loss_making (bool), loss_ratio (float), case_type ("clear" or "borderline")

Score the recommendation on these dimensions:
- outcome_aligned: Does the recommendation point toward the correct outcome?
- calibration_ok: Is the expressed certainty proportionate to how clear-cut the case was?
  A confident wrong answer OR a confident answer on a borderline case (loss_ratio near 1.0) fails calibration.
  A borderline case (case_type="borderline") that hedges or recommends review passes calibration.
- actionable: Would a non-technical reviewer know what to do next?
- grounded: Does the response cite specific factors or make unsupported claims?
- safe_deferral: On borderline cases, does the system appropriately hedge or escalate?

Return ONLY a JSON object:
{
  "outcome_aligned": true/false,
  "calibration_ok": true/false,
  "actionable": true/false,
  "grounded": true/false,
  "safe_deferral": true/false,
  "reasoning": "one sentence explanation"
}
"""


def load_eval_set() -> list[dict]:
    return json.loads(EVAL_SET_PATH.read_text())


def _call_judge_llm(recommendation: str, ground_truth: dict) -> dict:
    """Call the LLM judge and return parsed scoring dict."""
    user_content = json.dumps({
        "recommendation": recommendation,
        "ground_truth": ground_truth,
    }, indent=2)

    if JUDGE_PROVIDER == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY is not set.")
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=JUDGE_MAX_TOKENS,
            system=_JUDGE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        raw = response.content[0].text
    elif JUDGE_PROVIDER == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set.")
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            max_tokens=JUDGE_MAX_TOKENS,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        raw = response.choices[0].message.content
    else:
        raise ValueError(
            f"Unknown judge provider: {JUDGE_PROVIDER!r}. Must be 'anthropic' or 'openai'."
        )

    # Parse JSON, tolerating markdown fences
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if match:
        raw = match.group(1)
    else:
        start, end = raw.find("{"), raw.rfind("}") + 1
        if start != -1 and end > start:
            raw = raw[start:end]
    return json.loads(raw)


def score_recommendation(recommendation: str, ground_truth: dict) -> dict:
    """
    Score a single agent recommendation against ground truth using an LLM judge.

    correct = True only when outcome_aligned AND calibration_ok are both True.
    This ensures a confident wrong answer on a borderline case is still marked incorrect.
    """
    scores = _call_judge_llm(recommendation, ground_truth)
    scores["correct"] = bool(scores.get("outcome_aligned") and scores.get("calibration_ok"))
    return scores


def run_eval() -> None:
    from app.agent import run_agent

    eval_set = load_eval_set()
    results = []

    for i, item in enumerate(eval_set):
        record = {k: v for k, v in item.items() if k != "ground_truth"}
        ground_truth = item["ground_truth"]

        print(f"[{i + 1}/{len(eval_set)}] Assessing {record['record_id']}...", end=" ", flush=True)

        start = time.time()
        agent_result = run_agent(record)
        recommendation = agent_result["recommendation"]
        elapsed = time.time() - start

        score = score_recommendation(recommendation, ground_truth)
        results.append({
            "record_id": record["record_id"],
            "ground_truth": ground_truth,
            "recommendation": recommendation,
            "confidence_level": agent_result.get("confidence_level"),
            "risk_assessment": agent_result.get("risk_assessment"),
            "second_opinion_recommended": agent_result.get("second_opinion_recommended"),
            "score": score,
            "latency_s": round(elapsed, 2),
        })
        status = "✓" if score.get("correct") else "✗"
        print(f"{status} ({elapsed:.1f}s)")

    # ── Summary ───────────────────────────────────────────────────────────────
    n = len(results)
    n_correct = sum(1 for r in results if r["score"].get("correct"))
    avg_latency = sum(r["latency_s"] for r in results) / n

    loss_making = [r for r in results if r["ground_truth"]["is_loss_making"]]
    not_loss_making = [r for r in results if not r["ground_truth"]["is_loss_making"]]
    borderline = [r for r in results if r["ground_truth"].get("case_type") == "borderline"]
    clear = [r for r in results if r["ground_truth"].get("case_type") == "clear"]

    print(f"\n{'─' * 45}")
    print(f"Overall:          {n_correct}/{n} correct ({n_correct / n:.0%})")
    print(f"Average latency:  {avg_latency:.1f}s")
    if loss_making:
        tp = sum(1 for r in loss_making if r["score"].get("correct"))
        print(f"Loss-making:      {tp}/{len(loss_making)} correct")
    if not_loss_making:
        tn = sum(1 for r in not_loss_making if r["score"].get("correct"))
        print(f"Non-loss-making:  {tn}/{len(not_loss_making)} correct")
    if borderline:
        bp = sum(1 for r in borderline if r["score"].get("correct"))
        print(f"Borderline cases: {bp}/{len(borderline)} correct")
    if clear:
        cp = sum(1 for r in clear if r["score"].get("correct"))
        print(f"Clear cases:      {cp}/{len(clear)} correct")

    out = Path(__file__).parent / "eval_results.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nFull results → {out}")


if __name__ == "__main__":
    run_eval()
