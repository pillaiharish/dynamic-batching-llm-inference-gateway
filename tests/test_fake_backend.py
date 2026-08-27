import pytest

from gateway.backends.base import InferenceBackend
from gateway.backends.fake import FakeBackend
from gateway.schemas.chat import ChatCompletionRequest


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


@pytest.mark.asyncio
async def test_generate_chat_completion_shape() -> None:
    backend = FakeBackend()
    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "Hello"}],
    )

    response = await backend.generate(request)

    assert response["object"] == "chat.completion"
    assert response["model"] == "test-model"
    assert response["choices"][0]["message"] == {
        "role": "assistant",
        "content": "fake response",
    }


@pytest.mark.asyncio
async def test_close_marks_backend_closed() -> None:
    backend = FakeBackend()

    await backend.close()

    assert backend.closed is True
