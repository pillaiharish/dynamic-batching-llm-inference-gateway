"""Backend-neutral inference contract."""

from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class InferenceBackend(Protocol):
    """Operations an inference backend must provide to gateway services."""

    async def generate(self, request: Any) -> Any:
        """Generate one complete response."""
        ...

    def stream(self, request: Any) -> AsyncIterator[Any]:
        """Stream response chunks for one request."""
        ...

    async def generate_batch(self, requests: list[Any]) -> list[Any]:
        """Generate responses for a batch while preserving input order."""
        ...
