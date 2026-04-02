"""
FastAPI application — HTTP layer only.

Endpoints:
    GET  /health   — liveness check
    POST /assess   — run the assessment agent on a new record
"""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI, HTTPException

from app import cfg
from app.agent import run_agent
from app.schemas import AssessRequest, AssessResponse
from utils.config.log_handler import setup_logger

logger = setup_logger(
    level=cfg.get("logging.level")
)

app = FastAPI(
    title="Loss Prediction API",
    description="Prototype assessment agent for incoming insurance records.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/assess", response_model=AssessResponse)
def assess(request: AssessRequest) -> AssessResponse:
    """Run the assessment agent on a new record and return a recommendation."""
    logger.info("Assessing record: %s", request.record.record_id)
    record = request.record.model_dump()
    try:
        result = run_agent(record)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except EnvironmentError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return AssessResponse(
        record_id=request.record.record_id or "unknown",
        recommendation=result["recommendation"],
        risk_assessment=result["risk_assessment"],
        confidence_level=result["confidence_level"],
        key_factors=result["key_factors"],
        summary=result["summary"],
        similar_records_summary=result.get("similar_records_summary"),
        second_opinion_recommended=result["second_opinion_recommended"],
        review_guidance=result["review_guidance"],
        tools_used=result.get("tools_used", []),
        warnings=result.get("warnings", []),
    )


if __name__ == '__main__':
    uvicorn.run(app, host=cfg.get("fastapi.host"), port=cfg.get("fastapi.port"))
