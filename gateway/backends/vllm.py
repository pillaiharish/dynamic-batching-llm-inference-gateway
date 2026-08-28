"""Pooled asynchronous client for a vLLM OpenAI-compatible server."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from time import perf_counter
from typing import Any

import httpx2

from gateway.backends.base import BackendStream
from gateway.config import BackendConfig
from gateway.core.errors import (
    BackendCapabilityError,
    BackendConfigurationError,
    BackendHTTPError,
    BackendProtocolError,
    BackendRequestRejectedError,
    BackendTimeoutError,
    BackendUnavailableError,
)
from gateway.schemas.chat import ChatCompletionRequest

logger = logging.getLogger(__name__)


class _VLLMBackendStream:
    """Own one unbuffered vLLM HTTP response."""

    def __init__(
        self,
        response: httpx2.Response,
        on_close: Callable[[_VLLMBackendStream], None],
        *,
        backend_id: str,
        upstream_request_started_at: float,
    ) -> None:
        self._response = response
        self._on_close = on_close
        self.backend_id = backend_id
        self.upstream_request_started_at = upstream_request_started_at
        self._closed = False

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[bytes]:
        try:
            async for chunk in self._response.aiter_bytes():
                yield chunk
        except httpx2.TimeoutException as exc:
            raise BackendTimeoutError() from exc
        except httpx2.RequestError as exc:
            raise BackendUnavailableError() from exc
        except RuntimeError as exc:
            raise BackendUnavailableError() from exc
        finally:
            await self.aclose()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self._response.aclose()
        finally:
            self._on_close(self)


class VLLMBackend:
    """Forward validated chat requests to one configured vLLM server."""

    def __init__(
        self,
        backend_id: str,
        config: BackendConfig,
        *,
        connect_timeout_seconds: float,
        request_timeout_seconds: float,
        health_timeout_seconds: float,
        transport: httpx2.AsyncBaseTransport | None = None,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        headers = {"Content-Type": "application/json"}
        if config.api_key is not None:
            api_key = config.api_key.get_secret_value()
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

        timeout = httpx2.Timeout(
            request_timeout_seconds,
            connect=connect_timeout_seconds,
        )
        self.backend_id = backend_id
        self._clock = clock
        self._health_timeout_seconds = health_timeout_seconds
        self._client = httpx2.AsyncClient(
            base_url=config.base_url,
            headers=headers,
            timeout=timeout,
            transport=transport,
        )
        self._active_streams: set[_VLLMBackendStream] = set()

    @property
    def is_closed(self) -> bool:
        """Expose client lifecycle state for diagnostics and lifecycle tests."""
        return self._client.is_closed

    async def generate(self, request: ChatCompletionRequest) -> dict[str, Any]:
        """Forward one validated request and return its JSON response unchanged."""
        try:
            response = await self._client.post(
                "v1/chat/completions",
                json=request.to_upstream_payload(),
            )
        except httpx2.TimeoutException as exc:
            raise BackendTimeoutError() from exc
        except httpx2.ConnectError as exc:
            raise BackendUnavailableError() from exc
        except httpx2.RequestError as exc:
            raise BackendUnavailableError() from exc

        logger.info(
            "vLLM request completed",
            extra={
                "backend": "vllm",
                "backend_id": self.backend_id,
                "upstream_status": response.status_code,
            },
        )
        self._raise_for_upstream_status(response.status_code)

        try:
            payload = response.json()
        except ValueError as exc:
            raise BackendProtocolError() from exc
        if not isinstance(payload, dict):
            raise BackendProtocolError()
        return payload

    async def stream(self, request: ChatCompletionRequest) -> BackendStream:
        """Open and validate an unbuffered vLLM SSE response."""
        upstream_request = self._client.build_request(
            "POST",
            "v1/chat/completions",
            json=request.to_upstream_payload(),
            headers={"Accept": "text/event-stream"},
        )
        upstream_request_started_at = self._clock()
        try:
            response = await self._client.send(upstream_request, stream=True)
        except httpx2.TimeoutException as exc:
            raise BackendTimeoutError() from exc
        except httpx2.ConnectError as exc:
            raise BackendUnavailableError() from exc
        except httpx2.RequestError as exc:
            raise BackendUnavailableError() from exc
        except RuntimeError as exc:
            raise BackendUnavailableError() from exc

        logger.info(
            "vLLM streaming response opened",
            extra={
                "backend": "vllm",
                "backend_id": self.backend_id,
                "upstream_status": response.status_code,
            },
        )
        try:
            self._raise_for_upstream_status(response.status_code)
        except Exception:
            await response.aclose()
            raise

        media_type = response.headers.get("Content-Type", "").partition(";")[0]
        if media_type.strip().casefold() != "text/event-stream":
            await response.aclose()
            raise BackendProtocolError()

        stream = _VLLMBackendStream(
            response,
            self._active_streams.discard,
            backend_id=self.backend_id,
            upstream_request_started_at=upstream_request_started_at,
        )
        self._active_streams.add(stream)
        return stream

    async def generate_batch(self, requests: list[Any]) -> list[Any]:
        """Reject backend batching until the batching milestone is implemented."""
        raise BackendCapabilityError("Batch generation is not supported")

    async def check_health(self) -> bool:
        """Probe vLLM's control-plane health endpoint without generation traffic."""
        try:
            response = await self._client.get(
                "health",
                timeout=self._health_timeout_seconds,
            )
        except (httpx2.RequestError, RuntimeError):
            return False
        return 200 <= response.status_code < 300

    async def close(self) -> None:
        """Close the long-lived pooled HTTP client."""
        for stream in tuple(self._active_streams):
            await stream.aclose()
        await self._client.aclose()

    @staticmethod
    def _raise_for_upstream_status(status_code: int) -> None:
        if 200 <= status_code < 300:
            return
        if status_code in {400, 404, 422}:
            raise BackendRequestRejectedError(status_code=status_code)
        if status_code in {401, 403}:
            raise BackendConfigurationError()
        raise BackendHTTPError()
