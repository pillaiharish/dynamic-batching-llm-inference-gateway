from typing import Any

import pytest
from fastapi.testclient import TestClient

from gateway.app import create_app
from gateway.backends.fake import FakeBackend
from gateway.config import Settings
from gateway.routing.pool import BackendPool


class UnhealthyFakeBackend(FakeBackend):
    async def check_health(self) -> bool:
        return False


def make_pool(backend: FakeBackend) -> BackendPool:
    return BackendPool(
        {"gpu-a": backend},
        health_interval_seconds=60,
        health_timeout_seconds=1,
    )


def test_health_endpoint() -> None:
    with TestClient(create_app(Settings(_env_file=None), backend=FakeBackend())) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readiness_endpoint() -> None:
    with TestClient(create_app(Settings(_env_file=None), backend=FakeBackend())) as client:
        response = client.get("/readyz")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_pool_readiness_requires_one_healthy_backend() -> None:
    app = create_app(Settings(_env_file=None), backend=make_pool(FakeBackend()))

    with TestClient(app) as client:
        health = client.get("/healthz")
        readiness = client.get("/readyz")

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert readiness.status_code == 200
    assert readiness.json() == {"status": "ready"}


def test_pool_readiness_is_503_when_all_backends_are_unhealthy() -> None:
    app = create_app(Settings(_env_file=None), backend=make_pool(UnhealthyFakeBackend()))

    with TestClient(app) as client:
        health = client.get("/healthz")
        readiness = client.get("/readyz")

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert readiness.status_code == 503
    assert readiness.json() == {"status": "not_ready"}


def test_production_lifespan_builds_and_probes_each_configured_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: dict[str, FakeBackend] = {}
    captured: dict[str, tuple[str, dict[str, Any]]] = {}

    def fake_vllm_backend(backend_id: str, config: Any, **kwargs: Any) -> FakeBackend:
        backend = FakeBackend(prefix=backend_id)
        created[backend_id] = backend
        captured[backend_id] = (config.base_url, kwargs)
        return backend

    monkeypatch.setattr("gateway.app.VLLMBackend", fake_vllm_backend)
    settings = Settings(
        _env_file=None,
        backends_json={
            "gpu-a": {"base_url": "http://vllm-a:8000"},
            "gpu-b": {"base_url": "http://vllm-b:8000"},
        },
    )
    app = create_app(settings)

    with TestClient(app) as client:
        readiness = client.get("/readyz")
        assert client.portal is not None
        snapshot = client.portal.call(app.state.backend_pool.snapshot)

    assert readiness.status_code == 200
    assert set(snapshot.backends) == {"gpu-a", "gpu-b"}
    assert all(slot.healthy for slot in snapshot.backends.values())
    assert set(created) == {"gpu-a", "gpu-b"}
    assert created["gpu-a"] is not created["gpu-b"]
    assert captured["gpu-a"][0] == "http://vllm-a:8000"
    assert captured["gpu-b"][0] == "http://vllm-b:8000"
    assert captured["gpu-a"][1] == {
        "connect_timeout_seconds": 5.0,
        "request_timeout_seconds": 120.0,
        "health_timeout_seconds": 2.0,
    }
    assert all(backend.closed for backend in created.values())


def test_no_healthy_backend_errors_are_json_and_release_admission() -> None:
    settings = Settings(
        _env_file=None,
        tenants_json={
            "tenant-a": {
                "api_key": "tenant-a-key",
                "max_inflight": 1,
                "max_queue": 1,
            }
        },
        global_max_inflight=1,
    )
    backend = UnhealthyFakeBackend()
    pool = make_pool(backend)
    app = create_app(settings, backend=pool)

    with TestClient(
        app,
        headers={"Authorization": "Bearer tenant-a-key"},
    ) as client:
        json_response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
            },
            headers={"X-Request-ID": "no-json-backend"},
        )
        stream_response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
            headers={"X-Request-ID": "no-stream-backend"},
        )
        assert client.portal is not None
        admission_snapshot = client.portal.call(app.state.admission_controller.snapshot)
        routing_snapshot = client.portal.call(pool.snapshot)

    for response, request_id in (
        (json_response, "no-json-backend"),
        (stream_response, "no-stream-backend"),
    ):
        assert response.status_code == 503
        assert response.headers["Content-Type"].startswith("application/json")
        assert response.headers["X-Request-ID"] == request_id
        assert response.json()["error"] == {
            "code": "no_healthy_backend",
            "message": "No healthy inference backend is available",
            "request_id": request_id,
        }
    assert admission_snapshot.global_inflight == 0
    assert admission_snapshot.tenants["tenant-a"].inflight == 0
    assert routing_snapshot.backends["gpu-a"].inflight == 0
    assert backend.last_stream is None
