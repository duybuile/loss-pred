import json

import pytest


def test_load_record_json_bytes_accepts_plain_record():
    from app.streamlit_app import load_record_json_bytes

    payload = {"record_id": "NEW_0001", "risk_type": "cyber"}

    assert load_record_json_bytes(json.dumps(payload).encode("utf-8")) == payload


def test_load_record_json_bytes_unwraps_record_envelope():
    from app.streamlit_app import load_record_json_bytes

    payload = {"record": {"record_id": "NEW_0001", "risk_type": "cyber"}}

    assert load_record_json_bytes(json.dumps(payload).encode("utf-8")) == payload["record"]


def test_load_record_json_bytes_rejects_non_object():
    from app.streamlit_app import load_record_json_bytes

    with pytest.raises(ValueError, match="JSON file must contain an object"):
        load_record_json_bytes(json.dumps(["not", "a", "record"]).encode("utf-8"))


def test_build_assess_payload_wraps_record():
    from app.streamlit_app import build_assess_payload

    record = {"record_id": "NEW_0001", "risk_type": "cyber"}

    assert build_assess_payload(record) == {"record": record}


def test_build_assess_url_trims_trailing_slash():
    from app.streamlit_app import build_assess_url

    assert build_assess_url("http://localhost:8000/") == "http://localhost:8000/assess"


def test_get_default_api_base_url_prefers_env(monkeypatch):
    from app import streamlit_app

    monkeypatch.setenv("ASSESS_API_BASE_URL", "http://api:8000")

    assert streamlit_app.get_default_api_base_url() == "http://api:8000"
