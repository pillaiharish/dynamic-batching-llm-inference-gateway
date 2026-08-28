import asyncio
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Any

import pytest
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families
from starlette.types import Message, Receive, Scope

from gateway.admission.controller import AdmissionController
from gateway.app import create_app
from gateway.auth.tenants import TenantContext
from gateway.backends.base import BackendStream
from gateway.backends.fake import FakeBackend
from gateway.config import Settings
from gateway.core.errors import (
    BackendTimeoutError,
    BackendUnavailableError,
    TenantQueueFullError,
)
from gateway.observability.metrics import GatewayMetrics
from gateway.observability.middleware import InferenceMetricsMiddleware
from gateway.routing.pool import BackendPool

VALID_REQUEST: dict[str, Any] = {
    "model": "test-model",
    "messages": [{"role": "user", "content": "private-prompt-content"}],
}
AUTH_HEADERS = {
    "Authorization": "Bearer tenant-a-secret",
    "X-Request-ID": "private-request-id",
}
EXPECTED_METRICS = {
    "gateway_requests_total",
    "gateway_request_duration_seconds",
    "gateway_admission_queue_wait_seconds",
    "gateway_client_ttft_seconds",
    "gateway_backend_ttft_seconds",
    "gateway_ttft_observations_total",
    "gateway_observed_output_tokens_total",
    "gateway_token_accounting_requests_total",
    "gateway_errors_total",
    "gateway_admission_inflight",
    "gateway_admission_queued",
    "gateway_tenant_admission_inflight",
    "gateway_tenant_admission_queued",
    "gateway_backend_healthy",
    "gateway_backend_inflight",
    "gateway_backend_requests_total",
}


def make_settings(**overrides: Any) -> Settings:
    return Settings(
        _env_file=None,
        tenants_json={
            "tenant-a": {
                "api_key": "tenant-a-secret",
                "max_inflight": 1,
                "max_queue": 2,
            }
        },
        global_max_inflight=1,
        global_max_queue=2,
        **overrides,
    )


def sample(
    metrics: GatewayMetrics,
    name: str,
    labels: dict[str, str] | None = None,
) -> float | None:
    return metrics.registry.get_sample_value(name, labels or {})


def test_metrics_endpoint_uses_isolated_registry_and_excludes_sensitive_values() -> None:
    app1 = create_app(make_settings(), backend=FakeBackend(prefix="private-model-output"))
    app2 = create_app(make_settings(), backend=FakeBackend())

    with TestClient(app1, headers=AUTH_HEADERS) as client1:
        assert client1.post("/v1/chat/completions", json=VALID_REQUEST).status_code == 200
        first_scrape = client1.get("/metrics")
        second_scrape = client1.get("/metrics")
    with TestClient(app2, headers=AUTH_HEADERS) as client2:
        isolated_scrape = client2.get("/metrics")

    assert first_scrape.status_code == 200
    assert first_scrape.headers["Content-Type"].startswith("text/plain")
    parsed_families = {
        family.name: family for family in text_string_to_metric_families(first_scrape.text)
    }
    expected_family_names = {metric_name.removesuffix("_total") for metric_name in EXPECTED_METRICS}
    assert expected_family_names <= parsed_families.keys()
    request_duration_samples = {
        metric.name for metric in parsed_families["gateway_request_duration_seconds"].samples
    }
    assert {
        "gateway_request_duration_seconds_bucket",
        "gateway_request_duration_seconds_count",
        "gateway_request_duration_seconds_sum",
    } <= request_duration_samples
    assert (
        sample(
            app1.state.metrics,
            "gateway_requests_total",
            {"mode": "non_streaming", "status_code": "200", "outcome": "completed"},
        )
        == 1
    )
    assert (
        sample(
            app2.state.metrics,
            "gateway_requests_total",
            {"mode": "non_streaming", "status_code": "200", "outcome": "completed"},
        )
        is None
    )
    assert first_scrape.text == second_scrape.text
    assert "gateway_requests_total{" not in isolated_scrape.text
    for sensitive_value in (
        "tenant-a-secret",
        "private-request-id",
        "private-prompt-content",
        "private-model-output",
    ):
        assert sensitive_value not in first_scrape.text


class ResponseBackend(FakeBackend):
    def __init__(self, response: dict[str, Any]) -> None:
        super().__init__()
        self.response = response

    async def generate(self, _request: Any) -> dict[str, Any]:
        return self.response


@pytest.mark.parametrize(
    ("response", "result", "tokens"),
    [
        ({"usage": {"completion_tokens": 11}}, "observed", 11),
        ({"choices": []}, "missing", None),
        ({"usage": {"completion_tokens": "eleven"}}, "invalid", None),
        ({"usage": {"completion_tokens": -1}}, "invalid", None),
    ],
)
def test_non_streaming_token_accounting_is_authoritative_and_non_fatal(
    response: dict[str, Any],
    result: str,
    tokens: int | None,
) -> None:
    upstream = {"id": "unchanged", **response}
    app = create_app(make_settings(), backend=ResponseBackend(upstream))

    with TestClient(app, headers=AUTH_HEADERS) as client:
        downstream = client.post("/v1/chat/completions", json=VALID_REQUEST)

    assert downstream.status_code == 200
    assert downstream.json() == upstream
    assert (
        sample(
            app.state.metrics,
            "gateway_token_accounting_requests_total",
            {"mode": "non_streaming", "result": result},
        )
        == 1
    )
    assert (
        sample(
            app.state.metrics,
            "gateway_observed_output_tokens_total",
            {"mode": "non_streaming"},
        )
        == tokens
    )
    assert (
        sample(
            app.state.metrics,
            "gateway_client_ttft_seconds_count",
            {"backend_id": "fake"},
        )
        is None
    )


class FixedStream:
    backend_id = "gpu-a"

    def __init__(self, chunks: tuple[bytes, ...]) -> None:
        self.chunks = chunks
        self.upstream_request_started_at = perf_counter()
        self.closed = False

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


class StreamingUsageBackend(FakeBackend):
    def __init__(self, chunks: tuple[bytes, ...]) -> None:
        super().__init__()
        self.chunks = chunks
        self.last_payload: dict[str, object] | None = None

    async def stream(self, request: Any) -> BackendStream:
        self.last_payload = request.to_upstream_payload()
        return FixedStream(self.chunks)


def test_streaming_usage_is_counted_and_downstream_bytes_are_identical() -> None:
    chunks = (
        b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n',
        b'data: {"choices":[],"usage":{"completion_tokens":7}}\n\n',
        b"data: [DONE]\n\n",
    )
    backend = StreamingUsageBackend(chunks)
    app = create_app(make_settings(), backend=backend)
    request = {
        **VALID_REQUEST,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    with TestClient(app, headers=AUTH_HEADERS) as client:
        response = client.post("/v1/chat/completions", json=request)

    assert response.status_code == 200
    assert response.content == b"".join(chunks)
    assert backend.last_payload == request
    assert (
        sample(
            app.state.metrics,
            "gateway_observed_output_tokens_total",
            {"mode": "streaming"},
        )
        == 7
    )
    assert (
        sample(
            app.state.metrics,
            "gateway_token_accounting_requests_total",
            {"mode": "streaming", "result": "observed"},
        )
        == 1
    )


def test_streaming_without_usage_is_missing_and_options_are_not_injected() -> None:
    chunks = (
        b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n',
        b"data: [DONE]\n\n",
    )
    backend = StreamingUsageBackend(chunks)
    app = create_app(make_settings(), backend=backend)

    with TestClient(app, headers=AUTH_HEADERS) as client:
        response = client.post(
            "/v1/chat/completions",
            json={**VALID_REQUEST, "stream": True},
        )

    assert response.content == b"".join(chunks)
    assert backend.last_payload is not None
    assert "stream_options" not in backend.last_payload
    assert (
        sample(
            app.state.metrics,
            "gateway_observed_output_tokens_total",
            {"mode": "streaming"},
        )
        is None
    )
    assert (
        sample(
            app.state.metrics,
            "gateway_token_accounting_requests_total",
            {"mode": "streaming", "result": "missing"},
        )
        == 1
    )


@pytest.mark.parametrize(
    "stream_options",
    [
        {"include_usage": True},
        {"continuous_usage_stats": True},
        {"include_usage": 1},
    ],
)
def test_invalid_stream_options_are_normalized(stream_options: dict[str, object]) -> None:
    payload = {**VALID_REQUEST, "stream_options": stream_options}
    if "continuous_usage_stats" in stream_options:
        payload["stream"] = True
    app = create_app(make_settings(), backend=FakeBackend())

    with TestClient(app, headers=AUTH_HEADERS) as client:
        response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"


@pytest.mark.asyncio
async def test_streaming_request_duration_stays_open_until_final_body() -> None:
    metrics = GatewayMetrics()
    first_body_sent = asyncio.Event()
    allow_finish = asyncio.Event()
    now = 10.0

    async def streaming_app(scope: Scope, _receive: Receive, send: Any) -> None:
        scope["state"]["metrics_mode"] = "streaming"
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"first", "more_body": True})
        first_body_sent.set()
        await allow_finish.wait()
        await send({"type": "http.response.body", "body": b"last", "more_body": False})

    middleware = InferenceMetricsMiddleware(
        streaming_app,
        metrics=metrics,
        clock=lambda: now,
    )
    scope: Scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 1),
        "server": ("test", 80),
        "state": {},
    }

    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    sent: list[Message] = []

    async def send(message: Message) -> None:
        sent.append(message)

    task = asyncio.create_task(middleware(scope, receive, send))
    await first_body_sent.wait()

    assert (
        sample(
            metrics,
            "gateway_request_duration_seconds_count",
            {"mode": "streaming", "outcome": "completed"},
        )
        is None
    )

    now = 15.0
    allow_finish.set()
    await task

    assert (
        sample(
            metrics,
            "gateway_request_duration_seconds_count",
            {"mode": "streaming", "outcome": "completed"},
        )
        == 1
    )
    assert sample(
        metrics,
        "gateway_request_duration_seconds_sum",
        {"mode": "streaming", "outcome": "completed"},
    ) == pytest.approx(5.0)


@pytest.mark.asyncio
async def test_cancelled_request_duration_is_recorded_exactly_once() -> None:
    metrics = GatewayMetrics()
    first_body_sent = asyncio.Event()

    async def streaming_app(scope: Scope, _receive: Receive, send: Any) -> None:
        scope["state"]["metrics_mode"] = "streaming"
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"first", "more_body": True})
        first_body_sent.set()
        await asyncio.Event().wait()

    middleware = InferenceMetricsMiddleware(
        streaming_app,
        metrics=metrics,
        clock=lambda: 10.0,
    )
    scope: Scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 1),
        "server": ("test", 80),
        "state": {},
    }

    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(_message: Message) -> None:
        return None

    task = asyncio.create_task(middleware(scope, receive, send))
    await first_body_sent.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert (
        sample(
            metrics,
            "gateway_request_duration_seconds_count",
            {"mode": "streaming", "outcome": "cancelled"},
        )
        == 1
    )
    assert (
        sample(
            metrics,
            "gateway_requests_total",
            {"mode": "streaming", "status_code": "200", "outcome": "cancelled"},
        )
        == 1
    )


class ManualClock:
    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


class SignalingMetrics(GatewayMetrics):
    def __init__(self) -> None:
        self.queued = asyncio.Event()
        super().__init__()

    def admission_state_changed(
        self,
        global_inflight: int,
        global_queued: int,
        tenants: Mapping[str, tuple[int, int]],
    ) -> None:
        super().admission_state_changed(global_inflight, global_queued, tenants)
        if global_queued == 1:
            self.queued.set()


@pytest.mark.asyncio
async def test_admission_gauges_and_queue_wait_follow_real_transitions() -> None:
    tenant = TenantContext("tenant-a", max_inflight=1, max_queue=2)
    metrics = SignalingMetrics()
    clock = ManualClock(10.0)
    controller = AdmissionController(
        [tenant],
        global_max_inflight=1,
        global_max_queue=2,
        queue_timeout_seconds=1,
        observer=metrics,
        clock=clock,
    )
    first = await controller.acquire(tenant)

    clock.value = 10.1
    queued_task = asyncio.create_task(controller.acquire(tenant))
    await metrics.queued.wait()

    assert sample(metrics, "gateway_admission_inflight") == 1
    assert sample(metrics, "gateway_admission_queued") == 1
    assert (
        sample(
            metrics,
            "gateway_tenant_admission_inflight",
            {"tenant_id": "tenant-a"},
        )
        == 1
    )
    assert (
        sample(
            metrics,
            "gateway_tenant_admission_queued",
            {"tenant_id": "tenant-a"},
        )
        == 1
    )

    clock.value = 10.4
    await first.release()
    clock.value = 99.0
    second = await queued_task

    assert sample(metrics, "gateway_admission_queued") == 0
    assert (
        sample(
            metrics,
            "gateway_admission_queue_wait_seconds_count",
            {"outcome": "admitted"},
        )
        == 2
    )
    assert sample(
        metrics,
        "gateway_admission_queue_wait_seconds_sum",
        {"outcome": "admitted"},
    ) == pytest.approx(0.3)

    await second.release()
    await controller.shutdown()
    assert sample(metrics, "gateway_admission_inflight") == 0
    assert sample(metrics, "gateway_admission_queued") == 0


class PoolStream:
    def __init__(self, backend_id: str) -> None:
        self.backend_id = backend_id
        self.upstream_request_started_at = perf_counter()
        self.first_sent = asyncio.Event()
        self.allow_finish = asyncio.Event()
        self.closed = False

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[bytes]:
        self.first_sent.set()
        yield b"first"
        await self.allow_finish.wait()
        yield b"last"

    async def aclose(self) -> None:
        self.closed = True


class ControlledPoolBackend(FakeBackend):
    def __init__(self, backend_id: str, *, healthy: bool) -> None:
        super().__init__(prefix=backend_id)
        self.backend_id = backend_id
        self.healthy = healthy
        self.generate_started = asyncio.Event()
        self.allow_generate = asyncio.Event()
        self.streams: list[PoolStream] = []

    async def check_health(self) -> bool:
        return self.healthy

    async def generate(self, _request: Any) -> dict[str, Any]:
        self.generate_started.set()
        await self.allow_generate.wait()
        return {"usage": {"completion_tokens": 1}}

    async def stream(self, _request: Any) -> BackendStream:
        stream = PoolStream(self.backend_id)
        self.streams.append(stream)
        return stream

    async def close(self) -> None:
        self.allow_generate.set()
        for stream in self.streams:
            await stream.aclose()
        await super().close()


@pytest.mark.asyncio
async def test_backend_health_inflight_and_operation_outcomes_follow_pool_lifecycle() -> None:
    metrics = GatewayMetrics()
    backend_a = ControlledPoolBackend("A", healthy=True)
    backend_b = ControlledPoolBackend("B", healthy=False)
    pool = BackendPool(
        {"A": backend_a, "B": backend_b},
        health_interval_seconds=60,
        health_timeout_seconds=1,
        observer=metrics,
    )
    await pool.probe_once()

    assert sample(metrics, "gateway_backend_healthy", {"backend_id": "A"}) == 1
    assert sample(metrics, "gateway_backend_healthy", {"backend_id": "B"}) == 0

    generate_task = asyncio.create_task(pool.generate("request"))
    await backend_a.generate_started.wait()
    assert sample(metrics, "gateway_backend_inflight", {"backend_id": "A"}) == 1
    assert sample(metrics, "gateway_backend_inflight", {"backend_id": "B"}) == 0

    backend_b.healthy = True
    await pool.probe_once()
    assert sample(metrics, "gateway_backend_healthy", {"backend_id": "B"}) == 1
    backend_a.allow_generate.set()
    await generate_task
    assert sample(metrics, "gateway_backend_inflight", {"backend_id": "A"}) == 0
    assert (
        sample(
            metrics,
            "gateway_backend_requests_total",
            {"backend_id": "A", "operation": "generate", "outcome": "success"},
        )
        == 1
    )

    stream = await pool.stream("stream")
    assert stream.upstream_request_started_at == backend_b.streams[-1].upstream_request_started_at

    async def consume() -> None:
        async for _chunk in stream:
            pass

    consume_task = asyncio.create_task(consume())
    leaf_stream = backend_b.streams[-1]
    await leaf_stream.first_sent.wait()
    assert sample(metrics, "gateway_backend_inflight", {"backend_id": "B"}) == 1
    leaf_stream.allow_finish.set()
    await consume_task
    assert sample(metrics, "gateway_backend_inflight", {"backend_id": "B"}) == 0
    assert (
        sample(
            metrics,
            "gateway_backend_requests_total",
            {"backend_id": "B", "operation": "stream", "outcome": "success"},
        )
        == 1
    )

    cancelled_stream = await pool.stream("cancelled")
    cancelled_backend_id = cancelled_stream.backend_id
    await cancelled_stream.aclose()
    assert (
        sample(
            metrics,
            "gateway_backend_requests_total",
            {
                "backend_id": cancelled_backend_id,
                "operation": "stream",
                "outcome": "cancelled",
            },
        )
        == 1
    )
    assert (
        sample(
            metrics,
            "gateway_backend_inflight",
            {"backend_id": cancelled_backend_id},
        )
        == 0
    )

    await pool.close()
    assert sample(metrics, "gateway_backend_healthy", {"backend_id": "A"}) == 0
    assert sample(metrics, "gateway_backend_healthy", {"backend_id": "B"}) == 0


class TimeoutBackend(FakeBackend):
    async def generate(self, _request: Any) -> dict[str, Any]:
        raise BackendTimeoutError()


class RejectingAdmission:
    @asynccontextmanager
    async def admit(self, _tenant: TenantContext) -> AsyncIterator[None]:
        raise TenantQueueFullError()
        yield

    async def shutdown(self) -> None:
        return None


def test_error_counter_uses_stable_bounded_codes() -> None:
    app = create_app(make_settings(), backend=TimeoutBackend())
    with TestClient(app) as client:
        assert client.post("/v1/chat/completions", json=VALID_REQUEST).status_code == 401
        assert (
            client.post(
                "/v1/chat/completions",
                json={"messages": VALID_REQUEST["messages"]},
                headers=AUTH_HEADERS,
            ).status_code
            == 400
        )
        assert (
            client.post(
                "/v1/chat/completions",
                json=VALID_REQUEST,
                headers=AUTH_HEADERS,
            ).status_code
            == 504
        )

    for code in ("unauthorized", "invalid_request", "backend_timeout"):
        assert sample(app.state.metrics, "gateway_errors_total", {"code": code}) == 1

    rejected_app = create_app(
        make_settings(),
        backend=FakeBackend(),
        admission_controller=RejectingAdmission(),  # type: ignore[arg-type]
    )
    with TestClient(rejected_app, headers=AUTH_HEADERS) as client:
        assert client.post("/v1/chat/completions", json=VALID_REQUEST).status_code == 429
    assert (
        sample(
            rejected_app.state.metrics,
            "gateway_errors_total",
            {"code": "tenant_queue_full"},
        )
        == 1
    )

    unhealthy = ControlledPoolBackend("A", healthy=False)
    no_healthy_metrics = GatewayMetrics()
    pool = BackendPool(
        {"A": unhealthy},
        health_interval_seconds=60,
        health_timeout_seconds=1,
        observer=no_healthy_metrics,
    )
    no_healthy_app = create_app(
        make_settings(),
        backend=pool,
        metrics=no_healthy_metrics,
    )
    with TestClient(no_healthy_app, headers=AUTH_HEADERS) as client:
        assert client.post("/v1/chat/completions", json=VALID_REQUEST).status_code == 503
    assert (
        sample(
            no_healthy_metrics,
            "gateway_errors_total",
            {"code": "no_healthy_backend"},
        )
        == 1
    )


class FailingStream(FixedStream):
    def __init__(self) -> None:
        super().__init__((b'data: {"choices":[{"delta":{"content":"first"}}]}\n\n',))

    async def _iterate(self) -> AsyncIterator[bytes]:
        yield self.chunks[0]
        raise BackendUnavailableError()


class MidstreamFailureBackend(FakeBackend):
    async def stream(self, _request: Any) -> BackendStream:
        return FailingStream()


def test_midstream_error_has_one_gateway_error_and_one_backend_outcome() -> None:
    metrics = GatewayMetrics()
    pool = BackendPool(
        {"gpu-a": MidstreamFailureBackend()},
        health_interval_seconds=60,
        health_timeout_seconds=1,
        observer=metrics,
    )
    app = create_app(make_settings(), backend=pool, metrics=metrics)

    with TestClient(app, headers=AUTH_HEADERS) as client:
        response = client.post(
            "/v1/chat/completions",
            json={**VALID_REQUEST, "stream": True},
        )

    assert response.status_code == 200
    assert response.content.startswith(b"data:")
    assert sample(metrics, "gateway_errors_total", {"code": "stream_upstream_error"}) == 1
    assert (
        sample(
            metrics,
            "gateway_backend_requests_total",
            {"backend_id": "gpu-a", "operation": "stream", "outcome": "error"},
        )
        == 1
    )
    assert (
        sample(
            metrics,
            "gateway_requests_total",
            {"mode": "streaming", "status_code": "200", "outcome": "error"},
        )
        == 1
    )
    assert sample(metrics, "gateway_backend_healthy", {"backend_id": "gpu-a"}) == 0
