"""Deterministic backend used by tests and local development."""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from gateway.backends.base import BackendStream
from gateway.schemas.chat import ChatCompletionRequest


@dataclass(slots=True)
class FakeBackendStream:
    """Small byte stream used for deterministic gateway tests."""

    chunks: tuple[bytes, ...]
    backend_id: str | None = "fake"
    upstream_request_started_at: float | None = None
    closed: bool = False

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[bytes]:
        try:
            for chunk in self.chunks:
                yield chunk
        finally:
            await self.aclose()

    async def aclose(self) -> None:
        self.closed = True


@dataclass(slots=True)
class FakeBackend:
    """A tiny deterministic implementation of the inference backend contract."""

    prefix: str = "fake"
    stream_chunks: tuple[bytes, ...] = (
        b'data: {"id":"chatcmpl-fake","choices":[{"delta":{"content":"fake"}}]}\n\n',
        b"data: [DONE]\n\n",
    )
    closed: bool = False
    last_stream: FakeBackendStream | None = field(default=None, init=False)

    async def generate(self, request: Any) -> dict[str, Any]:
        if isinstance(request, ChatCompletionRequest):
            return {
                "id": "chatcmpl-fake",
                "object": "chat.completion",
                "created": 0,
                "model": request.model,
                "choices": [
                    {
                        "index": index,
                        "message": {
                            "role": "assistant",
                            "content": f"{self.prefix} response",
                        },
                        "finish_reason": "stop",
                    }
                    for index in range(request.n)
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }
        return {"input": request, "output": f"{self.prefix}:{request}"}

    async def stream(self, _request: Any) -> BackendStream:
        self.last_stream = FakeBackendStream(
            self.stream_chunks,
            upstream_request_started_at=perf_counter(),
        )
        return self.last_stream

    async def generate_batch(self, requests: list[Any]) -> list[dict[str, Any]]:
        return [await self.generate(request) for request in requests]

    async def check_health(self) -> bool:
        return not self.closed

    async def close(self) -> None:
        if self.last_stream is not None:
            await self.last_stream.aclose()
        self.closed = True
