from fastapi.testclient import TestClient


def test_health_endpoint_does_not_depend_on_vectorstore_startup(monkeypatch):
    from app import main as main_module
    from app import vectorstore as vectorstore_module

    def fail_vectorstore_init():
        raise RuntimeError("vectorstore init should not run at app startup")

    monkeypatch.setattr(vectorstore_module, "init", fail_vectorstore_init)

    with TestClient(main_module.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
