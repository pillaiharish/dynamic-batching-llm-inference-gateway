from typing import Any

import pytest
from fastapi.testclient import TestClient

from gateway.app import create_app
from gateway.backends.base import BackendBatchResult, BackendStream, InferenceBackend
from gateway.backends.fake import FakeBackend
from gateway.batching.models import BatchItemResult
from gateway.config import Settings
from gateway.core.errors import BackendTimeoutError

VALID_REQUEST: dict[str, Any] = {
    "model": "test-model",
    "messages": [{"role": "user", "content": "Hello"}],
}
TENANTS = {
    "tenant-a": {
        "api_key": "tenant-a-key",
        "max_inflight": 2,
        "max_queue": 4,
    }
}
AUTH_HEADERS = {"Authorization": "Bearer tenant-a-key"}


def make_settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, tenants_json=TENANTS, **overrides)


def make_client(
    *,
    settings: Settings | None = None,
    backend: InferenceBackend | None = None,
) -> TestClient:
    return TestClient(
        create_app(
            settings or make_settings(),
            backend=backend or FakeBackend(),
        ),
        headers=AUTH_HEADERS,
    )


def assert_invalid_request(response: Any, *, message: str | None = None) -> None:
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"
    assert response.json()["error"]["request_id"] == response.headers["X-Request-ID"]
    if message is not None:
        assert response.json()["error"]["message"] == message


def test_valid_chat_completion_request() -> None:
    payload = {
        **VALID_REQUEST,
        "temperature": 0.5,
        "top_p": 0.9,
        "max_tokens": 32,
        "stop": ["done"],
        "seed": 7,
        "n": 2,
        "stream": False,
    }

    with make_client() as client:
        response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    assert response.json()["object"] == "chat.completion"
    assert response.json()["model"] == "test-model"
    assert len(response.json()["choices"]) == 2


class ResponseBackend:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response

    async def generate(self, _request: Any) -> dict[str, Any]:
        return self.response

    async def stream(self, _request: Any) -> BackendStream:
        raise NotImplementedError

    async def generate_batch(self, _requests: list[Any]) -> list[Any]:
        raise NotImplementedError

    async def close(self) -> None:
        return None


def test_success_response_preserves_upstream_fields() -> None:
    upstream_response = {
        "id": "chatcmpl-upstream",
        "object": "chat.completion",
        "created": 123,
        "model": "test-model",
        "choices": [],
        "usage": {"total_tokens": 0},
        "vllm_extension": {"preserved": True},
    }

    with make_client(backend=ResponseBackend(upstream_response)) as client:
        response = client.post("/v1/chat/completions", json=VALID_REQUEST)

    assert response.status_code == 200
    assert response.json() == upstream_response


@pytest.mark.parametrize(
    "payload",
    [
        {"messages": VALID_REQUEST["messages"]},
        {**VALID_REQUEST, "model": "  "},
        {"model": "test-model"},
        {**VALID_REQUEST, "messages": []},
    ],
)
def test_required_field_validation(payload: dict[str, Any]) -> None:
    with make_client() as client:
        response = client.post("/v1/chat/completions", json=payload)

    assert_invalid_request(response)


@pytest.mark.parametrize(
    "message",
    [
        {"role": "tool", "content": "result"},
        {"role": "user", "content": ["not", "text"]},
        {"role": "user", "content": "  "},
        {"role": "user"},
        {"role": "user", "content": "hello", "name": "unsupported"},
    ],
)
def test_message_validation(message: dict[str, Any]) -> None:
    payload = {**VALID_REQUEST, "messages": [message]}

    with make_client() as client:
        response = client.post("/v1/chat/completions", json=payload)

    assert_invalid_request(response)


@pytest.mark.parametrize(
    "field,value",
    [
        ("temperature", -0.1),
        ("temperature", 2.1),
        ("top_p", -0.1),
        ("top_p", 1.1),
        ("max_tokens", 0),
        ("max_tokens", -1),
        ("n", 0),
        ("n", -1),
        ("n", "2"),
        ("stream", 1),
        ("stream", "true"),
    ],
)
def test_generation_schema_validation(field: str, value: object) -> None:
    with make_client() as client:
        response = client.post(
            "/v1/chat/completions",
            json={**VALID_REQUEST, field: value},
        )

    assert_invalid_request(response)


@pytest.mark.parametrize(
    "field,value,expected_message",
    [
        ("max_tokens", 9, "max_tokens must not exceed 8"),
        ("n", 3, "n must not exceed 2"),
    ],
)
def test_configured_generation_limits(
    field: str,
    value: int,
    expected_message: str,
) -> None:
    settings = make_settings(max_completion_tokens=8, max_choices=2)

    with make_client(settings=settings) as client:
        response = client.post(
            "/v1/chat/completions",
            json={**VALID_REQUEST, field: value},
        )

    assert_invalid_request(response, message=expected_message)


def test_streaming_chat_completion_returns_sse() -> None:
    with make_client() as client:
        response = client.post(
            "/v1/chat/completions",
            json={**VALID_REQUEST, "stream": True},
            headers={"X-Request-ID": "stream-request-id"},
        )

    assert response.status_code == 200
    assert response.headers["Content-Type"] == "text/event-stream"
    assert response.headers["Cache-Control"] == "no-cache"
    assert response.headers["X-Accel-Buffering"] == "no"
    assert response.headers["X-Request-ID"] == "stream-request-id"
    assert response.content.count(b"data: [DONE]\n\n") == 1


def test_unsupported_request_field_is_rejected() -> None:
    with make_client() as client:
        response = client.post(
            "/v1/chat/completions",
            json={**VALID_REQUEST, "tools": []},
        )

    assert_invalid_request(response)


def test_malformed_json_is_normalized() -> None:
    with make_client() as client:
        response = client.post(
            "/v1/chat/completions",
            content=b'{"model":',
            headers={"Content-Type": "application/json"},
        )

    assert_invalid_request(response, message="Invalid chat completion request")
    assert "json_invalid" not in response.text


def test_injected_backend_is_closed_on_shutdown() -> None:
    backend = FakeBackend()

    with make_client(backend=backend) as client:
        assert backend.closed is False
        assert client.get("/readyz").status_code == 200

    assert backend.closed is True


class FailOnceBackend:
    def __init__(self) -> None:
        self.calls = 0
        self.fake = FakeBackend()

    async def generate(self, _request: Any) -> Any:
        self.calls += 1
        if self.calls == 1:
            raise BackendTimeoutError()
        return await self.fake.generate(_request)

    async def stream(self, _request: Any) -> BackendStream:
        raise NotImplementedError

    async def generate_batch(self, _requests: list[Any]) -> list[Any]:
        raise NotImplementedError

    async def close(self) -> None:
        await self.fake.close()


def test_backend_error_is_normalized_and_releases_admission_slot() -> None:
    backend = FailOnceBackend()
    settings = make_settings(global_max_inflight=1)
    app = create_app(settings, backend=backend)

    with TestClient(app, headers=AUTH_HEADERS) as client:
        failed = client.post("/v1/chat/completions", json=VALID_REQUEST)
        succeeded = client.post("/v1/chat/completions", json=VALID_REQUEST)

    assert failed.status_code == 504
    assert failed.json()["error"]["code"] == "backend_timeout"
    assert failed.json()["error"]["message"] == "Inference backend timed out"
    assert succeeded.status_code == 200
    assert backend.calls == 2


class TrackingBackend(FakeBackend):
    def __init__(self) -> None:
        super().__init__()
        self.generate_count = 0
        self.stream_count = 0
        self.batch_count = 0

    async def generate(self, request: Any) -> dict[str, Any]:
        self.generate_count += 1
        return await super().generate(request)

    async def stream(self, request: Any) -> BackendStream:
        self.stream_count += 1
        return await super().stream(request)

    async def generate_batch(self, requests: list[Any]) -> BackendBatchResult:
        self.batch_count += 1
        return await super().generate_batch(requests)


class SpyBatcher:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.admission_inflight: list[int] = []
        self.app: Any = None
        self.shutdown_called = False

    async def submit(self, tenant_id: str, request: Any) -> BatchItemResult:
        self.calls.append((tenant_id, request))
        snapshot = await self.app.state.admission_controller.snapshot()
        self.admission_inflight.append(snapshot.global_inflight)
        return BatchItemResult(
            response={
                "id": "chatcmpl-batched",
                "object": "chat.completion",
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "batched response"},
                        "finish_reason": "stop",
                    }
                ],
            }
        )

    async def shutdown(self) -> None:
        self.shutdown_called = True


def test_route_batches_only_eligible_requests_after_admission() -> None:
    backend = TrackingBackend()
    batcher = SpyBatcher()
    settings = make_settings(dynamic_batching_enabled=True)
    app = create_app(settings, backend=backend, dynamic_batcher=batcher)  # type: ignore[arg-type]
    batcher.app = app

    with TestClient(app, headers=AUTH_HEADERS) as client:
        eligible = client.post("/v1/chat/completions", json=VALID_REQUEST)
        n_gt_one = client.post("/v1/chat/completions", json={**VALID_REQUEST, "n": 2})
        streaming = client.post("/v1/chat/completions", json={**VALID_REQUEST, "stream": True})

    assert eligible.status_code == 200
    assert n_gt_one.status_code == 200
    assert eligible.json()["choices"][0]["message"]["content"] == "batched response"
    assert batcher.admission_inflight == [1]
    assert [tenant_id for tenant_id, _request in batcher.calls] == ["tenant-a"]
    assert backend.generate_count == 1
    assert backend.stream_count == 1
    assert backend.batch_count == 0
    assert streaming.headers["Content-Type"].startswith("text/event-stream")
    assert batcher.shutdown_called is True
    metrics = app.state.metrics
    assert (
        metrics.registry.get_sample_value(
            "gateway_batch_eligibility_total",
            {"decision": "eligible", "reason": "eligible"},
        )
        == 1
    )
    assert (
        metrics.registry.get_sample_value(
            "gateway_batch_eligibility_total",
            {"decision": "bypass", "reason": "n_gt_1"},
        )
        == 1
    )
    assert (
        metrics.registry.get_sample_value(
            "gateway_batch_eligibility_total",
            {"decision": "bypass", "reason": "streaming"},
        )
        == 1
    )


def test_batching_disabled_uses_direct_generation_without_batcher_infrastructure() -> None:
    backend = TrackingBackend()
    app = create_app(make_settings(dynamic_batching_enabled=False), backend=backend)

    with TestClient(app, headers=AUTH_HEADERS) as client:
        assert app.state.dynamic_batcher is None
        response = client.post("/v1/chat/completions", json=VALID_REQUEST)

    assert response.status_code == 200
    assert backend.generate_count == 1
    assert backend.batch_count == 0
    assert (
        app.state.metrics.registry.get_sample_value(
            "gateway_batch_eligibility_total",
            {"decision": "bypass", "reason": "disabled"},
        )
        == 1
    )
