# tests/test_schemas.py
import pytest
from pydantic import ValidationError


def test_assess_request_accepts_partial_typed_record():
    from app.schemas import AssessRequest

    req = AssessRequest(
        record={
            "record_id": "NEW_0001",
            "risk_type": "cyber",
            "territory": "EU",
            "limit": 3438000,
            "premium": 26904,
        }
    )

    assert req.record.record_id == "NEW_0001"
    assert req.record.risk_type == "cyber"
    assert req.record.industry is None


def test_assess_request_rejects_invalid_field_type():
    from app.schemas import AssessRequest

    with pytest.raises(ValidationError):
        AssessRequest(
            record={
                "risk_type": "cyber",
                "territory": "EU",
                "limit": "not-a-number",
                "premium": 26904,
            }
        )


def test_assess_request_preserves_extra_fields():
    from app.schemas import AssessRequest

    req = AssessRequest(
        record={
            "risk_type": "cyber",
            "territory": "EU",
            "limit": 3438000,
            "premium": 26904,
            "custom_field": "kept-for-downstream-compatibility",
        }
    )

    assert req.record.model_dump()["custom_field"] == "kept-for-downstream-compatibility"


def test_assess_response_valid():
    from app.schemas import AssessResponse

    resp = AssessResponse(
        record_id="NEW_0001",
        recommendation="Likely a loss.",
        risk_assessment="likely_loss",
        confidence_level="high",
        key_factors=["prior_claims", "premium_to_limit"],
        summary="High risk based on model.",
        similar_records_summary=None,
        second_opinion_recommended=False,
        review_guidance="Decline or price up.",
        tools_used=["predict_loss"],
        warnings=[],
    )
    assert resp.record_id == "NEW_0001"
    assert resp.risk_assessment == "likely_loss"


def test_assess_response_missing_required_field():
    from app.schemas import AssessResponse

    with pytest.raises(ValidationError):
        AssessResponse(
            record_id="NEW_0001",
            # missing recommendation and other required fields
        )


def test_assess_response_similar_records_summary_optional():
    from app.schemas import AssessResponse

    resp = AssessResponse(
        record_id="NEW_0001",
        recommendation="Likely a loss.",
        risk_assessment="likely_loss",
        confidence_level="high",
        key_factors=[],
        summary="High risk.",
        similar_records_summary=None,
        second_opinion_recommended=False,
        review_guidance="Decline.",
        tools_used=["predict_loss"],
        warnings=[],
    )
    assert resp.similar_records_summary is None
