"""
FastAPI application — HTTP layer only.

Endpoints:
    GET  /health   — liveness check
    POST /assess   — run the assessment agent on a new record
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.agent import run_agent
from app.schemas import AssessRequest, AssessResponse
from app.vectorstore import init as init_vectorstore


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise shared resources at startup before serving any requests."""
    init_vectorstore()
    yield


app = FastAPI(
    title="Torch Assessment API",
    description="Prototype assessment agent for incoming insurance records.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/assess", response_model=AssessResponse)
def assess(request: AssessRequest) -> AssessResponse:
    """Run the assessment agent on a new record and return a recommendation."""
    try:
        result = run_agent(request.record)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except EnvironmentError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return AssessResponse(
        record_id=request.record.get("record_id", "unknown"),
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
