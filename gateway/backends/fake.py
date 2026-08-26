"""Deterministic backend used by tests and local development."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class FakeBackend:
    """A tiny deterministic implementation of the inference backend contract."""

    prefix: str = "fake"

    async def generate(self, request: Any) -> dict[str, Any]:
        return {"input": request, "output": f"{self.prefix}:{request}"}

    async def stream(self, request: Any) -> AsyncIterator[dict[str, Any]]:
        for index, chunk in enumerate((self.prefix, str(request))):
            yield {"index": index, "chunk": chunk}

    async def generate_batch(self, requests: list[Any]) -> list[dict[str, Any]]:
        return [await self.generate(request) for request in requests]
