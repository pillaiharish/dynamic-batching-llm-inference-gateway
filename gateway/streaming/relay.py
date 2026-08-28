"""Incremental SSE relay with disconnect-safe resource cleanup."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Literal

from starlette.requests import ClientDisconnect
from starlette.responses import StreamingResponse
from starlette.types import Receive, Scope, Send

from gateway.admission.controller import AdmissionLease
from gateway.backends.base import BackendStream

logger = logging.getLogger(__name__)

StreamOutcome = Literal["completed", "cancelled", "upstream_error"]


class StreamingRelay:
    """Own an opened backend stream and its admission lease until relay completion."""

    def __init__(
        self,
        backend_stream: BackendStream,
        lease: AdmissionLease,
        *,
        tenant_id: str,
        request_id: str | None,
    ) -> None:
        self._backend_stream = backend_stream
        self._lease = lease
        self._tenant_id = tenant_id
        self._request_id = request_id
        self._close_lock = asyncio.Lock()
        self._closed = False

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[bytes]:
        try:
            async for chunk in self._backend_stream:
                yield chunk
        except asyncio.CancelledError:
            await self.aclose("cancelled")
            raise
        except Exception as exc:
            logger.warning(
                "upstream stream failed after downstream streaming began",
                extra=self._log_fields(
                    "upstream_error",
                    error_type=type(exc).__name__,
                ),
            )
            await self.aclose("upstream_error")
        else:
            await self.aclose("completed")
        finally:
            if not self._closed:
                await self.aclose("cancelled")

    async def aclose(self, outcome: StreamOutcome = "cancelled") -> None:
        """Close upstream and release admission exactly once."""
        async with self._close_lock:
            if self._closed:
                return
            self._closed = True
            final_outcome = outcome
            try:
                await self._backend_stream.aclose()
            except Exception as exc:
                final_outcome = "upstream_error"
                logger.warning(
                    "failed to close upstream stream",
                    extra=self._log_fields(
                        final_outcome,
                        error_type=type(exc).__name__,
                    ),
                )
            finally:
                await self._lease.release()

            logger.info(
                "stream relay finished",
                extra=self._log_fields(final_outcome),
            )

    def _log_fields(
        self,
        outcome: StreamOutcome,
        *,
        error_type: str | None = None,
    ) -> dict[str, object]:
        fields: dict[str, object] = {
            "request_id": self._request_id,
            "tenant_id": self._tenant_id,
            "streaming": True,
            "stream_outcome": outcome,
        }
        if error_type is not None:
            fields["stream_error_type"] = error_type
        return fields


class SSEStreamingResponse(StreamingResponse):
    """Streaming response that always finalizes its relay on transport teardown."""

    def __init__(self, relay: StreamingRelay) -> None:
        self._relay = relay
        super().__init__(
            relay,
            status_code=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        except ClientDisconnect:
            await self._relay.aclose("cancelled")
        except asyncio.CancelledError:
            await self._relay.aclose("cancelled")
            raise
        finally:
            await self._relay.aclose()
