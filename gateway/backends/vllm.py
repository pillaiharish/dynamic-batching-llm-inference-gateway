"""Pooled asynchronous client for a vLLM OpenAI-compatible server."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from time import perf_counter
from typing import Any, cast

import httpx2

from gateway.backends.base import BackendBatchResult, BackendStream
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

    async def generate_batch(self, requests: list[Any]) -> BackendBatchResult:
        """Send one compatible batch request and strictly demultiplex its response."""
        if not requests or any(
            not isinstance(request, ChatCompletionRequest) for request in requests
        ):
            raise BackendCapabilityError("Batch generation requires chat completion requests")

        typed_requests = cast(list[ChatCompletionRequest], requests)
        if any(request.stream or request.n != 1 for request in typed_requests):
            raise BackendCapabilityError("Batch generation requires stream=false and n=1")
        shared_payload = typed_requests[0].to_batch_shared_payload()
        if any(
            request.to_batch_shared_payload() != shared_payload for request in typed_requests[1:]
        ):
            raise BackendCapabilityError("Batch generation requires compatible shared fields")

        batch_payload = dict(shared_payload)
        batch_payload["messages"] = [
            request.to_upstream_payload()["messages"] for request in typed_requests
        ]
        try:
            response = await self._client.post(
                "v1/chat/completions/batch",
                json=batch_payload,
            )
        except httpx2.TimeoutException as exc:
            raise BackendTimeoutError() from exc
        except httpx2.ConnectError as exc:
            raise BackendUnavailableError() from exc
        except httpx2.RequestError as exc:
            raise BackendUnavailableError() from exc

        logger.info(
            "vLLM batch request completed",
            extra={
                "backend": "vllm",
                "backend_id": self.backend_id,
                "upstream_status": response.status_code,
                "batch_size": len(typed_requests),
            },
        )
        self._raise_for_upstream_status(response.status_code)
        try:
            payload = response.json()
        except ValueError as exc:
            raise BackendProtocolError() from exc
        if not isinstance(payload, dict):
            raise BackendProtocolError()
        return self._demultiplex_batch_response(payload, len(typed_requests))

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

    @staticmethod
    def _demultiplex_batch_response(
        payload: dict[str, Any],
        batch_size: int,
    ) -> BackendBatchResult:
        choices = payload.get("choices")
        if not isinstance(choices, list) or len(choices) != batch_size:
            raise BackendProtocolError()

        by_index: dict[int, dict[str, Any]] = {}
        for choice in choices:
            if not isinstance(choice, dict):
                raise BackendProtocolError()
            index = choice.get("index")
            if not isinstance(index, int) or isinstance(index, bool):
                raise BackendProtocolError()
            if index < 0 or index >= batch_size or index in by_index:
                raise BackendProtocolError()
            by_index[index] = choice
        if set(by_index) != set(range(batch_size)):
            raise BackendProtocolError()

        safe_batch_fields = {
            key: payload[key]
            for key in ("id", "object", "created", "model", "system_fingerprint")
            if key in payload
        }
        responses: list[dict[str, Any]] = []
        for index in range(batch_size):
            member_choice = dict(by_index[index])
            member_choice["index"] = 0
            responses.append({**safe_batch_fields, "choices": [member_choice]})

        if "usage" not in payload:
            return BackendBatchResult(responses, None, "missing")
        usage = payload.get("usage")
        if not isinstance(usage, dict) or "completion_tokens" not in usage:
            return BackendBatchResult(responses, None, "invalid")
        completion_tokens = usage.get("completion_tokens")
        if (
            not isinstance(completion_tokens, int)
            or isinstance(completion_tokens, bool)
            or completion_tokens < 0
        ):
            return BackendBatchResult(responses, None, "invalid")
        return BackendBatchResult(responses, completion_tokens, "observed")
