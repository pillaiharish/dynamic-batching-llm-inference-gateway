from fastapi import FastAPI
from fastapi.testclient import TestClient

from gateway.app import create_app
from gateway.config import Settings
from gateway.core.errors import GatewayError


def make_error_app() -> FastAPI:
    app = create_app(Settings(_env_file=None))

    @app.get("/controlled-error")
    async def controlled_error() -> None:
        raise GatewayError(
            "The test request was rejected",
            code="test_error",
            status_code=422,
        )

    @app.get("/unexpected-error")
    async def unexpected_error() -> None:
        raise RuntimeError("sensitive implementation detail")

    return app


def test_gateway_error_has_consistent_response() -> None:
    with TestClient(make_error_app()) as client:
        response = client.get(
            "/controlled-error",
            headers={"X-Request-ID": "error-request-id"},
        )

    assert response.status_code == 422
    assert response.json() == {
        "error": {
            "code": "test_error",
            "message": "The test request was rejected",
            "request_id": "error-request-id",
        }
    }
    assert response.headers["X-Request-ID"] == "error-request-id"


def test_unexpected_error_does_not_leak_details() -> None:
    with TestClient(make_error_app(), raise_server_exceptions=False) as client:
        response = client.get("/unexpected-error")

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert response.json()["error"]["message"] == "Internal gateway error"
    assert "sensitive implementation detail" not in response.text
    assert response.json()["error"]["request_id"] == response.headers["X-Request-ID"]
