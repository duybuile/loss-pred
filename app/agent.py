# app/agent.py
"""
Assessment agent — hybrid controller orchestrator.

The controller owns tool routing and policy decisions.
The LLM is called once for prose synthesis when evidence is coherent.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from app import cfg
from app.controller import ControllerPolicy
from app.llm import LLMAdapter
from app.tools import run_predict_loss, run_retrieve_similar_records

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MAX_ITERATIONS: int = cfg.get("agent.max_iterations")
# MAX_ITERATIONS is read from config. The current orchestrator is single-pass;
# this constant is retained for future extension to an agentic tool-call loop.

# ── Module-level singletons ───────────────────────────────────────────────────

def _load_record_outcomes() -> dict[str, bool]:
    repo_root = Path(__file__).parent.parent

    records_path = repo_root / cfg.get("data.records_path")
    logger.info(f"Loading records from {records_path}")
    df = pd.read_csv(records_path)
    return dict(zip(df["record_id"], df["is_loss_making"].astype(bool)))


_controller = ControllerPolicy(
    config=cfg.get("controller"),
    record_outcomes=_load_record_outcomes(),
)

def _make_adapter() -> LLMAdapter:
    try:
        return LLMAdapter(
            provider=cfg.get("llm.provider"),
            model=cfg.get("llm.model"),
        )
    except Exception as e:
        logger.error(f"No LLMAdapter was initialised: {e}")
        return None  # type: ignore[return-value]

_adapter: LLMAdapter | None = _make_adapter()

# ── System prompt for synthesis ───────────────────────────────────────────────

def _load_system_prompt(repo_root: Path | None = None) -> str:
    if repo_root is None:
        repo_root = Path(__file__).parent.parent

    prompt_path = Path(cfg.get("prompt.orchestrator"))
    version = cfg.get("prompt.orchestrator_version")
    versioned_prompt_path = prompt_path.with_name(
        f"{prompt_path.stem}_{version}{prompt_path.suffix}"
    )

    full_path = repo_root / versioned_prompt_path
    logger.info(f"Loading system prompt from {full_path}")
    return full_path.read_text(encoding="utf-8")


SYSTEM_PROMPT = _load_system_prompt()

# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_agent(record: dict) -> dict[str, Any]:
    """
    Run the assessment agent on a single record.

    1. Call predict_loss (always first)
    2. Compute confidence via controller
    3. Conditionally call retrieve_similar_records
    4. Detect conflicts and determine escalation
    5. Either build deterministic response (escalation) or call LLM synthesis

    Returns a dict matching AssessResponse fields.
    """
    tools_used: list[str] = []
    warnings: list[str] = []

    # Step 1: Predict
    prediction = run_predict_loss(record)
    tools_used.append("predict_loss")
    predict_failed = "error" in prediction

    if predict_failed:
        warnings.append(f"Prediction failed: {prediction.get('error', 'unknown error')}")
        # Provide a minimal stub so downstream logic has something to work with
        prediction = {
            "is_loss_making_prediction": False,
            "probability_of_loss": 0.5,
            "top_features": [],
            "input_quality_flags": [],
            "model_warnings": [err for err in [prediction.get("error", "")] if err],
        }

    # Step 2: Confidence
    confidence_level = _controller.compute_confidence(prediction)

    # Propagate input quality flags as warnings
    for flag in prediction.get("input_quality_flags", []):
        warnings.append(flag)

    # Step 3: Retrieval (only when warranted)
    retrieval_results: list[dict] = []
    if _controller.should_retrieve(confidence_level=confidence_level, predict_failed=predict_failed):
        query = _build_retrieval_query(record)
        try:
            retrieval_results = run_retrieve_similar_records(
                query, n_results=cfg.get("controller.retrieval_n_results")
            )
            tools_used.append("retrieve_similar_records")
        except Exception as e:
            warnings.append(f"Retrieval failed: {e}")

    # Step 4: Conflict detection and escalation
    conflict_detected = False
    if retrieval_results:
        conflict_detected = _controller.detect_conflict(
            is_loss_making_prediction=prediction["is_loss_making_prediction"],
            retrieval_results=retrieval_results,
        )

    second_opinion_recommended = _controller.should_escalate(
        confidence_level=confidence_level,
        conflict_detected=conflict_detected,
    )

    risk_assessment = _controller.compute_risk_assessment(
        is_loss_making=prediction["is_loss_making_prediction"],
        confidence_level=confidence_level,
    )

    if conflict_detected:
        warnings.append("Model prediction conflicts with majority of similar historical records.")

    # Step 5: Build response
    # Bypass the LLM only when escalation is forced AND confidence is low AND retrieval
    # produced no results. When retrieval did run (even on a low-confidence record), the
    # LLM synthesises the retrieved evidence into a useful narrative — skipping it would
    # discard valuable context the reviewer needs.
    if second_opinion_recommended and confidence_level == "low" and not retrieval_results:
        # Fully deterministic response — no LLM
        synthesis = _build_deterministic_response(prediction, confidence_level, warnings)
    else:
        # LLM synthesis pass
        if _adapter is None:
            raise RuntimeError(
                "LLM adapter is not configured. "
                "Set the ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable."
            )
        evidence = {
            "record": {k: v for k, v in record.items() if k != "record_id"},
            "prediction": {
                "is_loss_making": prediction["is_loss_making_prediction"],
                "probability_of_loss": prediction["probability_of_loss"],
                "top_features": prediction["top_features"],
            },
            "confidence_level": confidence_level,
            "risk_assessment": risk_assessment,
            "second_opinion_recommended": second_opinion_recommended,
            "retrieved_records": retrieval_results or None,
            "warnings": warnings,
        }
        synthesis = _adapter.synthesize(evidence=evidence, system_prompt=SYSTEM_PROMPT)

    return {
        "recommendation": synthesis.get("recommendation", "No recommendation produced."),
        "risk_assessment": risk_assessment,
        "confidence_level": confidence_level,
        "key_factors": synthesis.get("key_factors", prediction.get("top_features", [])),
        "summary": synthesis.get("summary", ""),
        "similar_records_summary": synthesis.get("similar_records_summary"),
        "second_opinion_recommended": second_opinion_recommended,
        "review_guidance": synthesis.get("review_guidance", ""),
        "tools_used": tools_used,
        "warnings": warnings,
    }


def _build_retrieval_query(record: dict) -> str:
    """Construct a natural-language query for the vector store from record fields."""
    parts = []
    if record.get("risk_type"):
        parts.append(record["risk_type"])
    if record.get("industry"):
        parts.append(f"{record['industry']} company")
    if record.get("territory"):
        parts.append(f"in {record['territory']}")
    if record.get("prior_claims"):
        parts.append(f"with {record['prior_claims']} prior claim(s)")
    if record.get("limit"):
        parts.append(f"limit {record['limit']:,.0f}")
    return " ".join(parts) if parts else "insurance record"


def _build_deterministic_response(prediction: dict, confidence_level: str, warnings: list[str]) -> dict:
    """Build a safe fallback response without calling the LLM."""
    return {
        "recommendation": (
            "Insufficient information to produce a confident recommendation. "
            "Manual review is required."
        ),
        "summary": (
            f"Model confidence is {confidence_level}. "
            + (f"Issues: {'; '.join(warnings[:3])}." if warnings else "")
        ),
        "key_factors": prediction.get("top_features", []),
        "similar_records_summary": None,
        "review_guidance": "Refer to a senior underwriter for manual assessment.",
    }
