"""Per-application Prometheus collectors and safe update helpers."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Literal

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

logger = logging.getLogger(__name__)

Mode = Literal["streaming", "non_streaming", "unknown"]
RequestOutcome = Literal["completed", "cancelled", "error"]
AdmissionOutcome = Literal["admitted", "timeout", "rejected", "cancelled"]
BackendOperation = Literal["generate", "stream", "batch"]
BackendOutcome = Literal["success", "error", "cancelled"]
AccountingResult = Literal["observed", "missing", "invalid", "aggregate_only"]

REQUEST_DURATION_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120)
QUEUE_WAIT_BUCKETS = (
    0.001,
    0.0025,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1,
    2,
    5,
    10,
)
TTFT_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1, 2, 5, 10, 30)
BATCH_SIZE_BUCKETS = (1, 2, 4, 8, 16, 32, 64)
BATCH_WAIT_BUCKETS = (0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 1)


class GatewayMetrics:
    """Own all collectors for exactly one gateway application instance."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "gateway_requests_total",
            "Inference requests by validated mode, HTTP status, and lifecycle outcome.",
            ("mode", "status_code", "outcome"),
            registry=self.registry,
        )
        self.request_duration = Histogram(
            "gateway_request_duration_seconds",
            "End-to-end inference request duration from T0 through T4.",
            ("mode", "outcome"),
            buckets=REQUEST_DURATION_BUCKETS,
            registry=self.registry,
        )
        self.admission_queue_wait = Histogram(
            "gateway_admission_queue_wait_seconds",
            "Admission wait from immediately before acquire until grant or failure.",
            ("outcome",),
            buckets=QUEUE_WAIT_BUCKETS,
            registry=self.registry,
        )
        self.client_ttft = Histogram(
            "gateway_client_ttft_seconds",
            "Stream-observed time from gateway receipt T0 to first content T3.",
            ("backend_id",),
            buckets=TTFT_BUCKETS,
            registry=self.registry,
        )
        self.backend_ttft = Histogram(
            "gateway_backend_ttft_seconds",
            "Stream-observed time from upstream request start T2 to first content T3.",
            ("backend_id",),
            buckets=TTFT_BUCKETS,
            registry=self.registry,
        )
        self.ttft_observations = Counter(
            "gateway_ttft_observations_total",
            "Streaming requests with observed or missing first generated content.",
            ("result",),
            registry=self.registry,
        )
        self.observed_output_tokens = Counter(
            "gateway_observed_output_tokens_total",
            "Authoritative observed generated completion tokens.",
            ("mode",),
            registry=self.registry,
        )
        self.token_accounting_requests = Counter(
            "gateway_token_accounting_requests_total",
            "Requests by authoritative output-token accounting coverage.",
            ("mode", "result"),
            registry=self.registry,
        )
        self.errors = Counter(
            "gateway_errors_total",
            "Client-visible gateway lifecycle errors by stable code.",
            ("code",),
            registry=self.registry,
        )
        self.admission_inflight = Gauge(
            "gateway_admission_inflight",
            "Current process-local global admission inflight count.",
            registry=self.registry,
        )
        self.admission_queued = Gauge(
            "gateway_admission_queued",
            "Current process-local global admission queued count.",
            registry=self.registry,
        )
        self.tenant_admission_inflight = Gauge(
            "gateway_tenant_admission_inflight",
            "Current process-local tenant admission inflight count.",
            ("tenant_id",),
            registry=self.registry,
        )
        self.tenant_admission_queued = Gauge(
            "gateway_tenant_admission_queued",
            "Current process-local tenant admission queued count.",
            ("tenant_id",),
            registry=self.registry,
        )
        self.backend_healthy = Gauge(
            "gateway_backend_healthy",
            "Whether a configured backend is currently healthy (1 or 0).",
            ("backend_id",),
            registry=self.registry,
        )
        self.backend_inflight = Gauge(
            "gateway_backend_inflight",
            "Current gateway logical requests assigned to a backend operation.",
            ("backend_id",),
            registry=self.registry,
        )
        self.backend_requests = Counter(
            "gateway_backend_requests_total",
            "Selected backend operations by full lifecycle outcome.",
            ("backend_id", "operation", "outcome"),
            registry=self.registry,
        )
        self.batch_eligibility = Counter(
            "gateway_batch_eligibility_total",
            "Validated requests by dynamic-batching eligibility decision.",
            ("decision", "reason"),
            registry=self.registry,
        )
        self.batches = Counter(
            "gateway_batches_total",
            "Detached upstream batches by flush reason and lifecycle outcome.",
            ("flush_reason", "outcome"),
            registry=self.registry,
        )
        self.batch_size = Histogram(
            "gateway_batch_size",
            "Logical requests dispatched in one upstream batch operation.",
            buckets=BATCH_SIZE_BUCKETS,
            registry=self.registry,
        )
        self.batch_wait = Histogram(
            "gateway_batch_wait_seconds",
            "Per-member wait from dynamic-batch submission until dispatch.",
            buckets=BATCH_WAIT_BUCKETS,
            registry=self.registry,
        )
        self.batch_pending = Gauge(
            "gateway_batch_pending",
            "Admitted logical requests waiting for dynamic-batch dispatch.",
            registry=self.registry,
        )
        self.batch_inflight = Gauge(
            "gateway_batch_inflight",
            "Detached upstream batch HTTP operations currently executing.",
            registry=self.registry,
        )

    def render(self) -> bytes:
        """Render this application's registry in Prometheus text format."""
        return generate_latest(self.registry)

    def record_request(
        self,
        mode: Mode,
        status_code: int | str,
        outcome: RequestOutcome,
        duration_seconds: float,
    ) -> None:
        """Record one complete inference HTTP/SSE lifecycle."""
        try:
            self.requests.labels(mode, str(status_code), outcome).inc()
            self.request_duration.labels(mode, outcome).observe(max(0.0, duration_seconds))
        except Exception:
            logger.warning("metrics_update_error", exc_info=True)

    def admission_wait_observed(
        self,
        duration_seconds: float,
        outcome: AdmissionOutcome,
    ) -> None:
        """Observe one admission acquire attempt without affecting admission."""
        try:
            self.admission_queue_wait.labels(outcome).observe(max(0.0, duration_seconds))
        except Exception:
            logger.warning("metrics_update_error", exc_info=True)

    def admission_state_changed(
        self,
        global_inflight: int,
        global_queued: int,
        tenants: Mapping[str, tuple[int, int]],
    ) -> None:
        """Mirror one consistent admission state transition into gauges."""
        try:
            self.admission_inflight.set(max(0, global_inflight))
            self.admission_queued.set(max(0, global_queued))
            for tenant_id, (inflight, queued) in tenants.items():
                self.tenant_admission_inflight.labels(tenant_id).set(max(0, inflight))
                self.tenant_admission_queued.labels(tenant_id).set(max(0, queued))
        except Exception:
            logger.warning("metrics_update_error", exc_info=True)

    def backend_state_changed(
        self,
        backends: Mapping[str, tuple[bool, int]],
    ) -> None:
        """Mirror one consistent routing state transition into gauges."""
        try:
            for backend_id, (healthy, inflight) in backends.items():
                self.backend_healthy.labels(backend_id).set(1 if healthy else 0)
                self.backend_inflight.labels(backend_id).set(max(0, inflight))
        except Exception:
            logger.warning("metrics_update_error", exc_info=True)

    def backend_operation_observed(
        self,
        backend_id: str,
        operation: BackendOperation,
        outcome: BackendOutcome,
    ) -> None:
        """Record one selected backend operation after its lifecycle ends."""
        try:
            self.backend_requests.labels(backend_id, operation, outcome).inc()
        except Exception:
            logger.warning("metrics_update_error", exc_info=True)

    def observe_ttft(
        self,
        *,
        backend_id: str | None,
        client_seconds: float,
        backend_seconds: float | None,
    ) -> None:
        """Record the first generated-content observation exactly once."""
        safe_backend_id = backend_id or "unknown"
        try:
            self.client_ttft.labels(safe_backend_id).observe(max(0.0, client_seconds))
            if backend_seconds is not None:
                self.backend_ttft.labels(safe_backend_id).observe(max(0.0, backend_seconds))
            self.ttft_observations.labels("observed").inc()
        except Exception:
            logger.warning("metrics_update_error", exc_info=True)

    def record_missing_ttft(self) -> None:
        """Record a streaming lifecycle that ended without generated content."""
        try:
            self.ttft_observations.labels("missing").inc()
        except Exception:
            logger.warning("metrics_update_error", exc_info=True)

    def record_token_accounting(
        self,
        mode: Literal["streaming", "non_streaming"],
        result: AccountingResult,
        completion_tokens: int | None = None,
    ) -> None:
        """Record authoritative token coverage and, when valid, token count."""
        try:
            if result == "observed" and completion_tokens is not None:
                self.observed_output_tokens.labels(mode).inc(completion_tokens)
            self.token_accounting_requests.labels(mode, result).inc()
        except Exception:
            logger.warning("metrics_update_error", exc_info=True)

    def observe_non_streaming_usage(self, response: Mapping[str, object]) -> None:
        """Account for a successful JSON response without changing that response."""
        if "usage" not in response:
            self.record_token_accounting("non_streaming", "missing")
            return
        usage = response.get("usage")
        if not isinstance(usage, Mapping) or "completion_tokens" not in usage:
            self.record_token_accounting("non_streaming", "invalid")
            return
        completion_tokens = usage.get("completion_tokens")
        if (
            not isinstance(completion_tokens, int)
            or isinstance(completion_tokens, bool)
            or completion_tokens < 0
        ):
            self.record_token_accounting("non_streaming", "invalid")
            return
        self.record_token_accounting(
            "non_streaming",
            "observed",
            completion_tokens,
        )

    def record_error(self, code: str) -> None:
        """Increment one stable, bounded gateway error code."""
        try:
            self.errors.labels(code).inc()
        except Exception:
            logger.warning("metrics_update_error", exc_info=True)

    def batch_eligibility_observed(self, decision: str, reason: str) -> None:
        """Record one bounded dynamic-batching eligibility decision."""
        try:
            self.batch_eligibility.labels(decision, reason).inc()
        except Exception:
            logger.warning("metrics_update_error", exc_info=True)

    def batch_state_changed(self, pending: int, inflight: int) -> None:
        """Mirror current process-local batch pending and operation counts."""
        try:
            self.batch_pending.set(max(0, pending))
            self.batch_inflight.set(max(0, inflight))
        except Exception:
            logger.warning("metrics_update_error", exc_info=True)

    def batch_dispatched(
        self,
        _flush_reason: Literal["size", "timeout"],
        size: int,
        wait_seconds: tuple[float, ...],
    ) -> None:
        """Observe one actual dispatch and every member's batch-only wait."""
        try:
            self.batch_size.observe(size)
            for duration in wait_seconds:
                self.batch_wait.observe(max(0.0, duration))
        except Exception:
            logger.warning("metrics_update_error", exc_info=True)

    def batch_completed(
        self,
        flush_reason: Literal["size", "timeout"],
        outcome: Literal["success", "error", "cancelled"],
    ) -> None:
        """Record one detached upstream batch at full operation completion."""
        try:
            self.batches.labels(flush_reason, outcome).inc()
        except Exception:
            logger.warning("metrics_update_error", exc_info=True)

    def observe_batch_usage(
        self,
        usage_result: Literal["observed", "missing", "invalid"],
        completion_tokens: int | None,
        member_count: int,
    ) -> None:
        """Count aggregate tokens once and member coverage without false attribution."""
        try:
            if usage_result == "observed" and completion_tokens is not None:
                self.observed_output_tokens.labels("non_streaming").inc(completion_tokens)
                self.token_accounting_requests.labels("non_streaming", "aggregate_only").inc(
                    member_count
                )
            else:
                self.token_accounting_requests.labels("non_streaming", usage_result).inc(
                    member_count
                )
        except Exception:
            logger.warning("metrics_update_error", exc_info=True)
