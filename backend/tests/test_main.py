from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_healthcheck() -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_version() -> None:
    response = client.get("/api/version")

    assert response.status_code == 200
    assert response.json() == {"version": "0.0.1"}


def test_status() -> None:
    response = client.get("/api/status")

    assert response.status_code == 200
    assert response.json() == {
        "service": "menu-intelligence-api",
        "status": "ok",
        "version": "0.0.1",
    }
