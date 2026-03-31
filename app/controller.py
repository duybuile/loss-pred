"""
ControllerPolicy — owns all deterministic decisions in the assessment loop.

Confidence computation, retrieval gating, conflict detection, and escalation
are implemented here as pure functions on structured data. No LLM calls.
"""
from __future__ import annotations


class ControllerPolicy:
    """
    Deterministic policy controller for the assessment agent.

    Args:
        config:          Dict from conf/agent.toml [controller] section.
        record_outcomes: Dict mapping record_id → is_loss_making (bool),
                         loaded from records.csv for conflict detection.
    """

    def __init__(self, config: dict, record_outcomes: dict[str, bool]):
        self.high_threshold: float = config["high_confidence_threshold"]
        self.medium_threshold: float = config["medium_confidence_threshold"]
        self.retrieval_n_results: int = config["retrieval_n_results"]
        self.distance_threshold: float = config.get("retrieval_distance_threshold", 0.5)
        self.record_outcomes = record_outcomes

    # ── Confidence ────────────────────────────────────────────────────────────

    def compute_confidence(self, prediction: dict) -> str:
        """
        Compute confidence level from prediction output.

        Base band from probability margin, then adjusted down for quality issues.
        """
        flags: list[str] = prediction.get("input_quality_flags", [])
        warnings: list[str] = prediction.get("model_warnings", [])

        # Force low on serious issues
        has_required_missing = any(f.startswith("required_field_missing") for f in flags)
        has_invalid_value = any(f.startswith("invalid_value") for f in flags)
        has_model_warning = len(warnings) > 0

        if has_required_missing or has_invalid_value or has_model_warning:
            return "low"

        prob = prediction["probability_of_loss"]
        margin = abs(prob - 0.5)

        if margin >= self.high_threshold:
            base = "high"
        elif margin >= self.medium_threshold:
            base = "medium"
        else:
            return "low"

        # Downgrade one level for non-critical quality flags
        has_optional_missing = any(f.startswith("optional_field_missing") for f in flags)
        if has_optional_missing:
            return {"high": "medium", "medium": "low"}.get(base, "low")

        return base

    # ── Risk Assessment ───────────────────────────────────────────────────────

    def compute_risk_assessment(self, is_loss_making: bool, confidence_level: str) -> str:
        """Return a directional risk label. Low confidence always yields 'unclear'."""
        if confidence_level == "low":
            return "unclear"
        return "likely_loss" if is_loss_making else "unlikely_loss"

    # ── Retrieval Gating ──────────────────────────────────────────────────────

    def should_retrieve(self, confidence_level: str, predict_failed: bool) -> bool:
        """Retrieval runs only when confidence is low or prediction failed."""
        return confidence_level == "low" or predict_failed

    # ── Conflict Detection ────────────────────────────────────────────────────

    def detect_conflict(self, is_loss_making_prediction: bool, retrieval_results: list[dict]) -> bool:
        """
        Return True if the model prediction conflicts with the majority outcome
        of retrieved similar records.

        Conflict detection is skipped when all retrieved records have distance
        above the threshold (poor retrieval quality).

        Tie-break rule: On even splits (e.g., 1 loss-making, 1 non-loss),
        the majority is treated as "not loss-making" (sum(outcomes) > len/2 is False
        for ties). This asymmetric tie-break favors "no conflict" when outcomes are split.
        """
        good_results = [
            r for r in retrieval_results if r["distance"] <= self.distance_threshold
        ]
        if not good_results:
            return False

        known_outcomes = [
            self.record_outcomes[r["record_id"]]
            for r in good_results
            if r["record_id"] in self.record_outcomes
        ]
        if not known_outcomes:
            return False

        majority_loss_making = sum(known_outcomes) > len(known_outcomes) / 2
        return majority_loss_making != is_loss_making_prediction

    # ── Escalation ────────────────────────────────────────────────────────────

    def should_escalate(
        self,
        confidence_level: str,
        conflict_detected: bool,
    ) -> bool:
        """Return True when the system should recommend human review."""
        if conflict_detected:
            return True
        if confidence_level == "low":
            return True
        return False
