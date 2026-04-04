from fastapi.testclient import TestClient
import pytest


def test_health_endpoint_initialises_vectorstore_at_startup(monkeypatch):
    from app import main as main_module
    from app import vectorstore_mgt as vectorstore_module

    calls = {"count": 0}

    def track_vectorstore_init():
        calls["count"] += 1

    monkeypatch.setattr(vectorstore_module, "init", track_vectorstore_init)

    with TestClient(main_module.app) as client:
        response = client.get("/health")

    assert calls["count"] == 1
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_startup_failure_propagates_when_vectorstore_init_fails(monkeypatch):
    from app import main as main_module
    from app import vectorstore_mgt as vectorstore_module

    def fail_vectorstore_init():
        raise RuntimeError("vectorstore unavailable")

    monkeypatch.setattr(vectorstore_module, "init", fail_vectorstore_init)

    with pytest.raises(RuntimeError, match="vectorstore unavailable"):
        with TestClient(main_module.app):
            pass
