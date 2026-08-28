"""Bounded, fail-open observation of copied SSE bytes."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Mapping
from time import perf_counter
from typing import Any

from gateway.observability.metrics import GatewayMetrics

logger = logging.getLogger(__name__)

_EVENT_BOUNDARY = re.compile(rb"\r?\n\r?\n")
DEFAULT_MAX_EVENT_BYTES = 1024 * 1024


class SSEMetricsObserver:
    """Passively detect first content and authoritative final usage."""

    def __init__(
        self,
        metrics: GatewayMetrics,
        *,
        request_started_at: float,
        backend_id: str | None,
        upstream_request_started_at: float | None,
        clock: Callable[[], float] = perf_counter,
        max_event_bytes: int = DEFAULT_MAX_EVENT_BYTES,
    ) -> None:
        if max_event_bytes <= 0:
            raise ValueError("max_event_bytes must be positive")
        self._metrics = metrics
        self._request_started_at = request_started_at
        self._backend_id = backend_id
        self._upstream_request_started_at = upstream_request_started_at
        self._clock = clock
        self._max_event_bytes = max_event_bytes
        self._buffer = bytearray()
        self._first_content_seen = False
        self._usage_result: str | None = None
        self._completion_tokens: int | None = None
        self._disabled = False
        self._finalized = False

    @property
    def disabled(self) -> bool:
        """Whether malformed or oversized input disabled further parsing."""
        return self._disabled

    def observe_bytes(self, chunk: bytes) -> None:
        """Inspect copied upstream bytes; never propagate parser failures."""
        if self._disabled or self._finalized or not chunk:
            return
        try:
            offset = 0
            while offset < len(chunk):
                available = self._max_event_bytes - len(self._buffer)
                if available <= 0:
                    self._disable()
                    return
                end = min(len(chunk), offset + available)
                self._buffer.extend(chunk[offset:end])
                offset = end
                self._drain_events()
                if self._disabled:
                    return
        except Exception:
            self._disable()

    def finalize(self) -> None:
        """Record TTFT/token coverage exactly once at stream lifecycle end."""
        if self._finalized:
            return
        self._finalized = True
        if not self._first_content_seen:
            self._metrics.record_missing_ttft()
        if self._usage_result == "observed":
            self._metrics.record_token_accounting(
                "streaming",
                "observed",
                self._completion_tokens,
            )
        elif self._usage_result == "invalid" or self._disabled:
            self._metrics.record_token_accounting("streaming", "invalid")
        else:
            self._metrics.record_token_accounting("streaming", "missing")
        self._buffer.clear()

    def _drain_events(self) -> None:
        while match := _EVENT_BOUNDARY.search(self._buffer):
            event = bytes(self._buffer[: match.start()])
            del self._buffer[: match.end()]
            if len(event) > self._max_event_bytes:
                self._disable()
                return
            self._observe_event(event)
            if self._disabled:
                return

    def _observe_event(self, event: bytes) -> None:
        data_lines: list[bytes] = []
        for line in event.splitlines():
            if line.startswith(b":"):
                continue
            if line == b"data":
                data_lines.append(b"")
            elif line.startswith(b"data:"):
                value = line[5:]
                data_lines.append(value[1:] if value.startswith(b" ") else value)
        if not data_lines:
            return
        data = b"\n".join(data_lines)
        if data.strip() == b"[DONE]":
            return
        try:
            payload = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._disable()
            return
        if not isinstance(payload, Mapping):
            self._disable()
            return
        self._observe_first_content(payload)
        self._observe_usage(payload)

    def _observe_first_content(self, payload: Mapping[str, Any]) -> None:
        if self._first_content_seen:
            return
        choices = payload.get("choices")
        if not isinstance(choices, list):
            return
        for choice in choices:
            if not isinstance(choice, Mapping):
                continue
            delta = choice.get("delta")
            if not isinstance(delta, Mapping):
                continue
            content = delta.get("content")
            if isinstance(content, str) and content:
                observed_at = self._clock()
                backend_seconds = (
                    None
                    if self._upstream_request_started_at is None
                    else observed_at - self._upstream_request_started_at
                )
                self._metrics.observe_ttft(
                    backend_id=self._backend_id,
                    client_seconds=observed_at - self._request_started_at,
                    backend_seconds=backend_seconds,
                )
                self._first_content_seen = True
                return

    def _observe_usage(self, payload: Mapping[str, Any]) -> None:
        if self._usage_result is not None or "usage" not in payload:
            return
        usage = payload.get("usage")
        if not isinstance(usage, Mapping) or "completion_tokens" not in usage:
            self._usage_result = "invalid"
            return
        completion_tokens = usage.get("completion_tokens")
        if (
            not isinstance(completion_tokens, int)
            or isinstance(completion_tokens, bool)
            or completion_tokens < 0
        ):
            self._usage_result = "invalid"
            return
        self._usage_result = "observed"
        self._completion_tokens = completion_tokens

    def _disable(self) -> None:
        if self._disabled:
            return
        self._disabled = True
        self._buffer.clear()
        logger.warning("metrics_event_parse_error")
