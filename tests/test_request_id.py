from uuid import UUID

from fastapi.testclient import TestClient

from gateway.app import create_app
from gateway.config import Settings


def test_preserves_supplied_request_id() -> None:
    with TestClient(create_app(Settings(_env_file=None))) as client:
        response = client.get("/healthz", headers={"X-Request-ID": "caller-request-id"})

    assert response.headers["X-Request-ID"] == "caller-request-id"


def test_generates_request_id_when_missing() -> None:
    with TestClient(create_app(Settings(_env_file=None))) as client:
        response = client.get("/healthz")

    generated_request_id = response.headers["X-Request-ID"]
    assert str(UUID(generated_request_id)) == generated_request_id


def test_uses_configured_request_id_header() -> None:
    settings = Settings(_env_file=None, request_id_header="Trace-ID")
    with TestClient(create_app(settings)) as client:
        response = client.get("/healthz", headers={"Trace-ID": "trace-123"})

    assert response.headers["Trace-ID"] == "trace-123"
