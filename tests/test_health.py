from fastapi.testclient import TestClient

from gateway.app import create_app
from gateway.config import Settings


def test_health_endpoint() -> None:
    with TestClient(create_app(Settings(_env_file=None))) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readiness_endpoint() -> None:
    with TestClient(create_app(Settings(_env_file=None))) as client:
        response = client.get("/readyz")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}
