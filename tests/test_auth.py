from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast

import httpx2
import pytest
from fastapi.testclient import TestClient

from gateway.admission.controller import AdmissionController, AdmissionLease
from gateway.app import create_app
from gateway.auth.tenants import TenantRegistry
from gateway.backends.fake import FakeBackend
from gateway.backends.vllm import VLLMBackend
from gateway.config import Settings
from gateway.core.errors import (
    AdmissionTimeoutError,
    GatewayError,
    GatewayQueueFullError,
    TenantQueueFullError,
)

VALID_REQUEST = {
    "model": "test-model",
    "messages": [{"role": "user", "content": "Hello"}],
}
TENANTS: dict[str, dict[str, Any]] = {
    "tenant-a": {
        "api_key": "tenant-secret",
        "max_inflight": 1,
        "max_queue": 2,
    }
}


def make_settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, tenants_json=TENANTS, **overrides)


@pytest.mark.parametrize(
    "authorization",
    [None, "Basic credentials", "Bearer", "Bearer ", "Bearer unknown-key", "Bearer  token"],
)
def test_invalid_tenant_credentials_return_safe_401(authorization: str | None) -> None:
    headers = {"X-Request-ID": "auth-request-id"}
    if authorization is not None:
        headers["Authorization"] = authorization

    app = create_app(make_settings(), backend=FakeBackend())
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=VALID_REQUEST, headers=headers)

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"
    assert response.headers["X-Request-ID"] == "auth-request-id"
    assert response.json() == {
        "error": {
            "code": "unauthorized",
            "message": "Invalid or missing tenant credentials",
            "request_id": "auth-request-id",
        }
    }
    assert "tenant-secret" not in response.text
    assert "unknown-key" not in response.text


def test_valid_tenant_credentials_resolve_safe_identity() -> None:
    settings = make_settings()
    registry = TenantRegistry(settings.tenants_json)

    tenant = registry.authenticate("Bearer tenant-secret")

    assert tenant.tenant_id == "tenant-a"
    assert tenant.max_inflight == 1
    assert tenant.max_queue == 2
    assert "tenant-secret" not in repr(tenant)


def test_valid_tenant_credentials_allow_chat_request() -> None:
    app = create_app(make_settings(), backend=FakeBackend())

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json=VALID_REQUEST,
            headers={"Authorization": "Bearer tenant-secret"},
        )

    assert response.status_code == 200
    assert response.json()["object"] == "chat.completion"


def test_health_endpoints_remain_unauthenticated() -> None:
    app = create_app(make_settings(), backend=FakeBackend())

    with TestClient(app) as client:
        health = client.get("/healthz")
        readiness = client.get("/readyz")

    assert health.status_code == 200
    assert readiness.status_code == 200


def test_tenant_authorization_is_never_forwarded_to_vllm() -> None:
    observed_authorization: list[str] = []

    async def handler(request: httpx2.Request) -> httpx2.Response:
        observed_authorization.append(request.headers["Authorization"])
        return httpx2.Response(
            200,
            json={
                "id": "chatcmpl-upstream",
                "object": "chat.completion",
                "created": 1,
                "model": "test-model",
                "choices": [],
            },
        )

    settings = make_settings(
        backends_json={
            "upstream": {
                "base_url": "https://vllm.example.test",
                "api_key": "backend-secret",
            }
        },
    )
    backend = VLLMBackend(
        "upstream",
        settings.backends_json["upstream"],
        connect_timeout_seconds=settings.vllm_connect_timeout_seconds,
        request_timeout_seconds=settings.vllm_request_timeout_seconds,
        health_timeout_seconds=settings.backend_health_timeout_seconds,
        transport=httpx2.MockTransport(handler),
    )
    app = create_app(settings, backend=backend)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json=VALID_REQUEST,
            headers={"Authorization": "Bearer tenant-secret"},
        )

    assert response.status_code == 200
    assert observed_authorization == ["Bearer backend-secret"]


def test_credentials_do_not_appear_in_application_logs(capsys: pytest.CaptureFixture[str]) -> None:
    settings = make_settings(
        backends_json={
            "upstream": {
                "base_url": "https://vllm.example.test",
                "api_key": "backend-secret",
            }
        }
    )
    app = create_app(settings, backend=FakeBackend())

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json=VALID_REQUEST,
            headers={"Authorization": "Bearer tenant-secret"},
        )

    assert response.status_code == 200
    captured = capsys.readouterr()
    assert "tenant-secret" not in captured.out
    assert "backend-secret" not in captured.out
    assert "tenant-secret" not in captured.err
    assert "backend-secret" not in captured.err
    assert '"tenant_id":"tenant-a"' in captured.out


class RejectingAdmissionController:
    def __init__(self, error: GatewayError) -> None:
        self.error = error

    @asynccontextmanager
    async def admit(self, _tenant: Any) -> AsyncIterator[AdmissionLease]:
        raise self.error
        yield  # pragma: no cover

    async def shutdown(self) -> None:
        return None


@pytest.mark.parametrize(
    "error,expected_code",
    [
        (TenantQueueFullError(), "tenant_queue_full"),
        (GatewayQueueFullError(), "gateway_queue_full"),
        (AdmissionTimeoutError(), "admission_timeout"),
    ],
)
def test_admission_rejections_have_normalized_429_responses(
    error: GatewayError,
    expected_code: str,
) -> None:
    admission = cast(AdmissionController, RejectingAdmissionController(error))
    app = create_app(
        make_settings(),
        backend=FakeBackend(),
        admission_controller=admission,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json=VALID_REQUEST,
            headers={
                "Authorization": "Bearer tenant-secret",
                "X-Request-ID": "admission-request-id",
            },
        )

    assert response.status_code == 429
    assert response.json()["error"]["code"] == expected_code
    assert response.json()["error"]["request_id"] == "admission-request-id"
