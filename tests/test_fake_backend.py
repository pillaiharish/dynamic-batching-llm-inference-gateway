import pytest

from gateway.backends.base import InferenceBackend
from gateway.backends.fake import FakeBackend


@pytest.mark.asyncio
async def test_generate_is_deterministic() -> None:
    backend = FakeBackend()

    assert isinstance(backend, InferenceBackend)
    assert await backend.generate("hello") == {"input": "hello", "output": "fake:hello"}


@pytest.mark.asyncio
async def test_stream_is_deterministic() -> None:
    backend = FakeBackend()

    chunks = [chunk async for chunk in backend.stream("hello")]

    assert chunks == [
        {"index": 0, "chunk": "fake"},
        {"index": 1, "chunk": "hello"},
    ]


@pytest.mark.asyncio
async def test_generate_batch_preserves_order() -> None:
    backend = FakeBackend()

    responses = await backend.generate_batch(["one", "two"])

    assert responses == [
        {"input": "one", "output": "fake:one"},
        {"input": "two", "output": "fake:two"},
    ]
