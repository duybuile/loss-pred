# app/schemas.py
"""
Request and response schemas for the assessment API.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class AssessRequest(BaseModel):
    record: dict[str, Any]


class AssessResponse(BaseModel):
    record_id: str
    recommendation: str           # Core free-text output for the reviewer
    risk_assessment: str          # "likely_loss" | "unlikely_loss" | "unclear"
    confidence_level: str         # "high" | "medium" | "low"
    key_factors: list[str]        # Plain-English descriptions of main risk drivers
    summary: str                  # 2-3 sentence explanation of evidence
    similar_records_summary: str | None  # Narrative from retrieval; None if retrieval did not run
    second_opinion_recommended: bool     # Explicit escalation flag
    review_guidance: str          # What the reviewer should do next
    tools_used: list[str]         # Which tools the agent called (auditability)
    warnings: list[str]           # Data quality issues, model limitations, conflicts
