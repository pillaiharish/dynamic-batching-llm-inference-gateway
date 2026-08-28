"""Pure ASGI inference lifecycle timing."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from time import perf_counter
from typing import Any, cast

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from gateway.observability.metrics import GatewayMetrics, Mode, RequestOutcome


class InferenceMetricsMiddleware:
    """Observe T0 through the final response body or downstream cancellation."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        metrics: GatewayMetrics,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self.app = app
        self.metrics = metrics
        self.clock = clock

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not self._is_inference_request(scope):
            await self.app(scope, receive, send)
            return

        started_at = self.clock()
        state = cast(dict[str, Any], scope.setdefault("state", {}))
        state["request_started_at"] = started_at
        status_code: int | None = None
        finalized = False

        def finalize(outcome: RequestOutcome) -> None:
            nonlocal finalized
            if finalized:
                return
            finalized = True
            mode = cast(Mode, state.get("metrics_mode", "unknown"))
            self.metrics.record_request(
                mode,
                status_code if status_code is not None else "unknown",
                outcome,
                self.clock() - started_at,
            )

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                stream_outcome = state.get("stream_outcome")
                if stream_outcome == "cancelled":
                    outcome: RequestOutcome = "cancelled"
                elif stream_outcome == "upstream_error" or (status_code or 500) >= 400:
                    outcome = "error"
                else:
                    outcome = "completed"
                finalize(outcome)

        try:
            await self.app(scope, receive, send_wrapper)
        except asyncio.CancelledError:
            finalize("cancelled")
            raise
        except BaseException:
            finalize("error")
            raise
        else:
            if not finalized:
                finalize("cancelled")

    @staticmethod
    def _is_inference_request(scope: Scope) -> bool:
        return scope.get("method") == "POST" and scope.get("path") == "/v1/chat/completions"
