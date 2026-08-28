from collections.abc import AsyncIterator

import pytest

from gateway.admission.controller import AdmissionController
from gateway.auth.tenants import TenantContext
from gateway.observability.metrics import GatewayMetrics
from gateway.observability.sse import SSEMetricsObserver
from gateway.streaming.relay import StreamingRelay


def sample(
    metrics: GatewayMetrics,
    name: str,
    labels: dict[str, str] | None = None,
) -> float | None:
    return metrics.registry.get_sample_value(name, labels or {})


def test_fragmented_first_content_records_client_and_backend_ttft_once() -> None:
    metrics = GatewayMetrics()
    observed_times = iter((11.0,))
    observer = SSEMetricsObserver(
        metrics,
        request_started_at=10.0,
        backend_id="gpu-a",
        upstream_request_started_at=10.5,
        clock=lambda: next(observed_times),
    )

    observer.observe_bytes(b": keepalive\n\n")
    observer.observe_bytes(b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n')
    observer.observe_bytes(b'data: {"choices":[{"del')

    assert sample(metrics, "gateway_client_ttft_seconds_count", {"backend_id": "gpu-a"}) is None

    observer.observe_bytes(b'ta":{"content":"Hello"}}]}\n\n')
    observer.observe_bytes(b'data: {"choices":[{"delta":{"content":" again"}}]}\n\n')
    observer.finalize()

    assert (
        sample(
            metrics,
            "gateway_client_ttft_seconds_count",
            {"backend_id": "gpu-a"},
        )
        == 1
    )
    assert sample(
        metrics,
        "gateway_client_ttft_seconds_sum",
        {"backend_id": "gpu-a"},
    ) == pytest.approx(1.0)
    assert (
        sample(
            metrics,
            "gateway_backend_ttft_seconds_count",
            {"backend_id": "gpu-a"},
        )
        == 1
    )
    assert sample(
        metrics,
        "gateway_backend_ttft_seconds_sum",
        {"backend_id": "gpu-a"},
    ) == pytest.approx(0.5)
    assert sample(metrics, "gateway_ttft_observations_total", {"result": "observed"}) == 1


def test_multiple_events_per_chunk_and_authoritative_usage_are_idempotent() -> None:
    metrics = GatewayMetrics()
    observer = SSEMetricsObserver(
        metrics,
        request_started_at=1.0,
        backend_id="gpu-a",
        upstream_request_started_at=1.5,
        clock=lambda: 2.0,
    )
    observer.observe_bytes(
        b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        b'data: {"choices":[],"usage":{"completion_tokens":7}}\n\n'
        b'data: {"choices":[],"usage":{"completion_tokens":99}}\n\n'
        b"data: [DONE]\n\n"
    )
    observer.finalize()
    observer.finalize()

    assert (
        sample(
            metrics,
            "gateway_observed_output_tokens_total",
            {"mode": "streaming"},
        )
        == 7
    )
    assert (
        sample(
            metrics,
            "gateway_token_accounting_requests_total",
            {"mode": "streaming", "result": "observed"},
        )
        == 1
    )


@pytest.mark.parametrize(
    ("event", "result"),
    [
        (b"data: [DONE]\n\n", "missing"),
        (b'data: {"usage":{"completion_tokens":"seven"}}\n\n', "invalid"),
        (b'data: {"usage":{"completion_tokens":-1}}\n\n', "invalid"),
    ],
)
def test_streaming_token_coverage_never_invents_tokens(event: bytes, result: str) -> None:
    metrics = GatewayMetrics()
    observer = SSEMetricsObserver(
        metrics,
        request_started_at=1.0,
        backend_id="gpu-a",
        upstream_request_started_at=1.0,
    )

    observer.observe_bytes(event)
    observer.finalize()

    assert (
        sample(
            metrics,
            "gateway_observed_output_tokens_total",
            {"mode": "streaming"},
        )
        is None
    )
    assert (
        sample(
            metrics,
            "gateway_token_accounting_requests_total",
            {"mode": "streaming", "result": result},
        )
        == 1
    )
    assert sample(metrics, "gateway_ttft_observations_total", {"result": "missing"}) == 1


def test_malformed_or_oversized_event_disables_parsing_without_raising() -> None:
    malformed_metrics = GatewayMetrics()
    malformed = SSEMetricsObserver(
        malformed_metrics,
        request_started_at=1.0,
        backend_id="gpu-a",
        upstream_request_started_at=1.0,
    )
    malformed.observe_bytes(b"data: {not-json}\n\n")
    malformed.observe_bytes(b'data: {"choices":[{"delta":{"content":"ignored"}}]}\n\n')
    malformed.finalize()

    oversized_metrics = GatewayMetrics()
    oversized = SSEMetricsObserver(
        oversized_metrics,
        request_started_at=1.0,
        backend_id="gpu-a",
        upstream_request_started_at=1.0,
        max_event_bytes=8,
    )
    oversized.observe_bytes(b"data: 123")
    oversized.finalize()

    assert malformed.disabled is True
    assert oversized.disabled is True
    assert (
        sample(
            malformed_metrics,
            "gateway_token_accounting_requests_total",
            {"mode": "streaming", "result": "invalid"},
        )
        == 1
    )
    assert (
        sample(
            oversized_metrics,
            "gateway_token_accounting_requests_total",
            {"mode": "streaming", "result": "invalid"},
        )
        == 1
    )


class ByteStream:
    backend_id = "gpu-a"
    upstream_request_started_at = 10.5

    def __init__(self, chunks: tuple[bytes, ...]) -> None:
        self.chunks = chunks
        self.closed = False

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_streaming_relay_observes_a_copy_and_preserves_raw_bytes() -> None:
    tenant = TenantContext("tenant-a", max_inflight=1, max_queue=1)
    admission = AdmissionController(
        [tenant],
        global_max_inflight=1,
        global_max_queue=1,
        queue_timeout_seconds=1,
    )
    lease = await admission.acquire(tenant)
    chunks = (
        b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n',
        b'data: {"choices":[],"usage":{"completion_tokens":7}}\n\n',
        b"data: [DONE]\n\n",
    )
    stream = ByteStream(chunks)
    metrics = GatewayMetrics()
    observer = SSEMetricsObserver(
        metrics,
        request_started_at=10.0,
        backend_id=stream.backend_id,
        upstream_request_started_at=stream.upstream_request_started_at,
        clock=lambda: 11.0,
    )
    outcomes: list[str] = []
    relay = StreamingRelay(
        stream,
        lease,
        tenant_id=tenant.tenant_id,
        request_id="request-id",
        metrics_observer=observer,
        on_outcome=outcomes.append,
    )

    downstream = b"".join([chunk async for chunk in relay])

    assert downstream == b"".join(chunks)
    assert outcomes == ["completed"]
    assert stream.closed is True
    assert (await admission.snapshot()).global_inflight == 0
    assert (
        sample(
            metrics,
            "gateway_observed_output_tokens_total",
            {"mode": "streaming"},
        )
        == 7
    )
