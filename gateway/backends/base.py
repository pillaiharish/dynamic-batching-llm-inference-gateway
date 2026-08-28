"""Backend-neutral inference contracts."""

from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class BackendStream(Protocol):
    """An opened incremental backend response with explicit cleanup."""

    backend_id: str | None
    upstream_request_started_at: float | None

    def __aiter__(self) -> AsyncIterator[bytes]:
        """Yield response bytes without imposing event boundaries."""
        ...

    async def aclose(self) -> None:
        """Close the upstream response and release its connection."""
        ...


@runtime_checkable
class InferenceBackend(Protocol):
    """Operations an inference backend must provide to gateway services."""

    async def generate(self, request: Any) -> Any:
        """Generate one complete response."""
        ...

    async def stream(self, request: Any) -> BackendStream:
        """Open one streaming response before downstream headers are sent."""
        ...

    async def generate_batch(self, requests: list[Any]) -> list[Any]:
        """Generate responses for a batch while preserving input order."""
        ...

    async def close(self) -> None:
        """Release resources owned by the backend."""
        ...


@runtime_checkable
class HealthCheckBackend(InferenceBackend, Protocol):
    """A leaf inference backend whose routability can be probed."""

    async def check_health(self) -> bool:
        """Return whether the backend health endpoint currently succeeds."""
        ...
