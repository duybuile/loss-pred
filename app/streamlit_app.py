from __future__ import annotations

import json
import os
from typing import Any

import httpx

from app import cfg
from app.schemas import AssessResponse


def get_default_api_base_url() -> str:
    return os.getenv("ASSESS_API_BASE_URL", cfg.get("streamlit.api_base_url"))


def get_streamlit_host() -> str:
    return os.getenv("STREAMLIT_SERVER_ADDRESS", cfg.get("streamlit.host"))


def get_streamlit_port() -> int:
    value = os.getenv("STREAMLIT_SERVER_PORT")
    if value is not None:
        return int(value)
    return int(cfg.get("streamlit.port"))


def load_record_json_bytes(raw_bytes: bytes) -> dict[str, Any]:
    payload = json.loads(raw_bytes.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("JSON file must contain an object.")

    if "record" in payload:
        record = payload["record"]
        if not isinstance(record, dict):
            raise ValueError("The 'record' field must contain an object.")
        return record

    return payload


def build_assess_payload(record: dict[str, Any]) -> dict[str, Any]:
    return {"record": record}


def build_assess_url(api_base_url: str) -> str:
    return f"{api_base_url.rstrip('/')}/assess"


def request_assessment(
    record: dict[str, Any],
    *,
    api_base_url: str,
    timeout_seconds: float = 60.0,
) -> AssessResponse:
    response = httpx.post(
        build_assess_url(api_base_url),
        json=build_assess_payload(record),
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    return AssessResponse.model_validate(response.json())


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Loss Prediction Demo", page_icon=":clipboard:", layout="wide")
    st.title("Loss Prediction Demo")
    st.caption("Upload a JSON record, review it, then execute the FastAPI assessment.")

    api_base_url = st.sidebar.text_input("FastAPI base URL", value=get_default_api_base_url())
    uploaded_file = st.file_uploader("Select a JSON record", type=["json"])

    record: dict[str, Any] | None = None
    if uploaded_file is not None:
        try:
            record = load_record_json_bytes(uploaded_file.getvalue())
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            st.error(f"Invalid JSON file: {exc}")
        else:
            st.subheader("Selected Record")
            st.json(record)

    if st.button("Execute", type="primary", disabled=record is None):
        if record is None:
            st.warning("Select a JSON record first.")
            return

        with st.spinner("Calling /assess ..."):
            try:
                assessment = request_assessment(record, api_base_url=api_base_url)
            except httpx.HTTPStatusError as exc:
                detail = exc.response.text
                st.error(f"API returned {exc.response.status_code}: {detail}")
                return
            except httpx.HTTPError as exc:
                st.error(f"Could not reach the API: {exc}")
                return
            except ValueError as exc:
                st.error(f"Could not parse API response: {exc}")
                return

        st.success("Assessment completed.")

        col1, col2, col3 = st.columns(3)
        col1.metric("Risk Assessment", assessment.risk_assessment)
        col2.metric("Confidence", assessment.confidence_level)
        col3.metric(
            "Second Opinion",
            "Yes" if assessment.second_opinion_recommended else "No",
        )

        st.subheader("Recommendation")
        st.write(assessment.recommendation)

        st.subheader("Summary")
        st.write(assessment.summary)

        if assessment.similar_records_summary:
            st.subheader("Similar Records")
            st.write(assessment.similar_records_summary)

        st.subheader("Key Factors")
        if assessment.key_factors:
            for factor in assessment.key_factors:
                st.write(f"- {factor}")
        else:
            st.write("No key factors returned.")

        st.subheader("Review Guidance")
        st.write(assessment.review_guidance)

        if assessment.tools_used:
            st.subheader("Tools Used")
            st.write(", ".join(assessment.tools_used))

        if assessment.warnings:
            st.subheader("Warnings")
            for warning in assessment.warnings:
                st.warning(warning)

        with st.expander("Raw Response"):
            st.json(assessment.model_dump())


if __name__ == "__main__":
    main()
