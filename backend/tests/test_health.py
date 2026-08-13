from fastapi.testclient import TestClient

from app.main import app


def test_healthcheck_returns_service_metadata() -> None:
    with TestClient(app) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "echora-api",
        "environment": "development",
    }
