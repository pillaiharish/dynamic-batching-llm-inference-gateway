"""Pooled asynchronous client for a vLLM OpenAI-compatible server."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx2

from gateway.config import Settings
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


class VLLMBackend:
    """Forward validated non-streaming chat requests to one configured vLLM server."""

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        headers = {"Content-Type": "application/json"}
        if settings.vllm_api_key is not None:
            api_key = settings.vllm_api_key.get_secret_value()
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

        timeout = httpx2.Timeout(
            settings.vllm_request_timeout_seconds,
            connect=settings.vllm_connect_timeout_seconds,
        )
        self._client = httpx2.AsyncClient(
            base_url=settings.vllm_base_url,
            headers=headers,
            timeout=timeout,
            transport=transport,
        )

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
            extra={"backend": "vllm", "upstream_status": response.status_code},
        )
        self._raise_for_upstream_status(response.status_code)

        try:
            payload = response.json()
        except ValueError as exc:
            raise BackendProtocolError() from exc
        if not isinstance(payload, dict):
            raise BackendProtocolError()
        return payload

    def stream(self, request: Any) -> AsyncIterator[Any]:
        """Return an iterator that explicitly rejects unsupported streaming."""
        return self._unsupported_stream(request)

    async def _unsupported_stream(self, _request: Any) -> AsyncIterator[Any]:
        raise BackendCapabilityError("Streaming is not supported in v0.2")
        yield  # pragma: no cover

    async def generate_batch(self, requests: list[Any]) -> list[Any]:
        """Reject backend batching until the batching milestone is implemented."""
        raise BackendCapabilityError("Batch generation is not supported in v0.2")

    async def close(self) -> None:
        """Close the long-lived pooled HTTP client."""
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
