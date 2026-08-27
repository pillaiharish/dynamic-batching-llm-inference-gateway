"""Deterministic backend used by tests and local development."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from gateway.schemas.chat import ChatCompletionRequest


@dataclass(slots=True)
class FakeBackend:
    """A tiny deterministic implementation of the inference backend contract."""

    prefix: str = "fake"
    closed: bool = False

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

    async def stream(self, request: Any) -> AsyncIterator[dict[str, Any]]:
        for index, chunk in enumerate((self.prefix, str(request))):
            yield {"index": index, "chunk": chunk}

    async def generate_batch(self, requests: list[Any]) -> list[dict[str, Any]]:
        return [await self.generate(request) for request in requests]

    async def close(self) -> None:
        self.closed = True
