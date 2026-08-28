import asyncio
import json
from typing import Any

import httpx2
import pytest
from fastapi.testclient import TestClient

from gateway.app import create_app
from gateway.backends.vllm import VLLMBackend
from gateway.config import BackendConfig, Settings
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

UPSTREAM_RESPONSE = {
    "id": "chatcmpl-upstream",
    "object": "chat.completion",
    "created": 123,
    "model": "served-model",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    "vllm_extension": {"preserved": True},
}

SSE_CHUNKS = (
    b'data: {"cho',
    b'ices":[{"delta":{"content":"Hel"}}]}\n\ndata: {"choices":[{"delta":{"content":"lo"}}]}\n\n',
    b": keepalive\n\n",
    b"data: [DONE]\n\n",
)


class TrackingByteStream(httpx2.AsyncByteStream):
    def __init__(self, chunks: tuple[bytes, ...] = SSE_CHUNKS) -> None:
        self.chunks = chunks
        self.iterated = False
        self.closed = False

    async def __aiter__(self):
        self.iterated = True
        for chunk in self.chunks:
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


class BlockingByteStream(httpx2.AsyncByteStream):
    def __init__(self) -> None:
        self.first_produced = asyncio.Event()
        self.release_second = asyncio.Event()
        self.closed = False

    async def __aiter__(self):
        self.first_produced.set()
        yield SSE_CHUNKS[0]
        await self.release_second.wait()
        yield SSE_CHUNKS[1]

    async def aclose(self) -> None:
        self.closed = True
        self.release_second.set()


class FailingByteStream(httpx2.AsyncByteStream):
    def __init__(self, exception_type: type[httpx2.RequestError]) -> None:
        self.exception_type = exception_type
        self.closed = False

    async def __aiter__(self):
        yield SSE_CHUNKS[0]
        request = httpx2.Request("POST", "https://vllm.example.test/v1/chat/completions")
        raise self.exception_type("midstream transport failure", request=request)

    async def aclose(self) -> None:
        self.closed = True


def chat_request(**overrides: Any) -> ChatCompletionRequest:
    payload: dict[str, Any] = {
        "model": "served-model",
        "messages": [{"role": "user", "content": "Hello"}],
        **overrides,
    }
    return ChatCompletionRequest.model_validate(payload)


def make_backend(
    handler: Any,
    *,
    backend_id: str = "test-backend",
    base_url: str = "https://vllm.example.test/root/",
    api_key: str | None = None,
    vllm_connect_timeout_seconds: float = 5.0,
    vllm_request_timeout_seconds: float = 120.0,
    backend_health_timeout_seconds: float = 2.0,
) -> VLLMBackend:
    return VLLMBackend(
        backend_id,
        BackendConfig(base_url=base_url, api_key=api_key),
        connect_timeout_seconds=vllm_connect_timeout_seconds,
        request_timeout_seconds=vllm_request_timeout_seconds,
        health_timeout_seconds=backend_health_timeout_seconds,
        transport=httpx2.MockTransport(handler),
    )


@pytest.mark.asyncio
async def test_generate_forwards_url_headers_and_exact_payload() -> None:
    captured: dict[str, Any] = {}

    async def handler(request: httpx2.Request) -> httpx2.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["content_type"] = request.headers["Content-Type"]
        captured["authorization"] = request.headers["Authorization"]
        captured["payload"] = json.loads(request.content)
        captured["timeout"] = request.extensions["timeout"]
        return httpx2.Response(200, json=UPSTREAM_RESPONSE)

    backend = make_backend(
        handler,
        api_key="backend-secret",
        vllm_connect_timeout_seconds=2.5,
        vllm_request_timeout_seconds=9.0,
    )
    request = chat_request(
        max_tokens=32,
        temperature=0.4,
        top_p=0.8,
        stop=["done"],
        seed=7,
        n=2,
        stream=False,
    )

    try:
        response = await backend.generate(request)
    finally:
        await backend.close()

    assert captured["method"] == "POST"
    assert captured["url"] == "https://vllm.example.test/root/v1/chat/completions"
    assert captured["content_type"] == "application/json"
    assert captured["authorization"] == "Bearer backend-secret"
    assert captured["payload"] == {
        "model": "served-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.4,
        "top_p": 0.8,
        "max_tokens": 32,
        "stop": ["done"],
        "seed": 7,
        "n": 2,
        "stream": False,
    }
    assert captured["timeout"] == {
        "connect": 2.5,
        "read": 9.0,
        "write": 9.0,
        "pool": 9.0,
    }
    assert response == UPSTREAM_RESPONSE


@pytest.mark.asyncio
async def test_generate_omits_authorization_when_key_is_unconfigured() -> None:
    async def handler(request: httpx2.Request) -> httpx2.Response:
        assert "Authorization" not in request.headers
        return httpx2.Response(200, json=UPSTREAM_RESPONSE)

    backend = make_backend(handler)
    try:
        await backend.generate(chat_request())
    finally:
        await backend.close()


@pytest.mark.parametrize(
    "exception_type,expected_error",
    [
        (httpx2.ConnectError, BackendUnavailableError),
        (httpx2.ReadTimeout, BackendTimeoutError),
    ],
)
@pytest.mark.asyncio
async def test_transport_failures_are_mapped(
    exception_type: type[httpx2.RequestError],
    expected_error: type[Exception],
) -> None:
    async def handler(request: httpx2.Request) -> httpx2.Response:
        raise exception_type("transport failed", request=request)

    backend = make_backend(handler)
    try:
        with pytest.raises(expected_error):
            await backend.generate(chat_request())
    finally:
        await backend.close()


@pytest.mark.parametrize("status_code", [400, 404, 422])
@pytest.mark.asyncio
async def test_safe_upstream_client_errors_preserve_status(status_code: int) -> None:
    backend = make_backend(lambda _request: httpx2.Response(status_code, json={"error": "detail"}))
    try:
        with pytest.raises(BackendRequestRejectedError) as caught:
            await backend.generate(chat_request())
    finally:
        await backend.close()

    assert caught.value.status_code == status_code
    assert caught.value.message == "Inference backend rejected the request"


@pytest.mark.parametrize("status_code", [401, 403])
@pytest.mark.asyncio
async def test_upstream_authentication_errors_are_gateway_failures(status_code: int) -> None:
    backend = make_backend(lambda _request: httpx2.Response(status_code))
    try:
        with pytest.raises(BackendConfigurationError):
            await backend.generate(chat_request())
    finally:
        await backend.close()


@pytest.mark.parametrize("status_code", [429, 500, 503])
@pytest.mark.asyncio
async def test_other_upstream_errors_map_to_safe_bad_gateway(status_code: int) -> None:
    backend = make_backend(lambda _request: httpx2.Response(status_code))
    try:
        with pytest.raises(BackendHTTPError) as caught:
            await backend.generate(chat_request())
    finally:
        await backend.close()

    assert caught.value.status_code == 502


@pytest.mark.parametrize(
    "response",
    [
        httpx2.Response(200, content=b"not json", headers={"Content-Type": "text/plain"}),
        httpx2.Response(200, json=["not", "an", "object"]),
    ],
)
@pytest.mark.asyncio
async def test_invalid_success_response_maps_to_protocol_error(
    response: httpx2.Response,
) -> None:
    backend = make_backend(lambda _request: response)
    try:
        with pytest.raises(BackendProtocolError):
            await backend.generate(chat_request())
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_stream_opens_unbuffered_request_and_relays_exact_bytes() -> None:
    captured: dict[str, Any] = {}
    response_body = TrackingByteStream()

    async def handler(request: httpx2.Request) -> httpx2.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers["Authorization"]
        captured["accept"] = request.headers["Accept"]
        captured["payload"] = json.loads(request.content)
        return httpx2.Response(
            200,
            headers={"Content-Type": "text/event-stream; charset=utf-8"},
            stream=response_body,
        )

    backend = make_backend(handler, api_key="backend-secret")
    try:
        stream = await backend.stream(chat_request(stream=True))
        assert response_body.iterated is False
        chunks = [chunk async for chunk in stream]
    finally:
        await backend.close()

    assert captured == {
        "method": "POST",
        "url": "https://vllm.example.test/root/v1/chat/completions",
        "authorization": "Bearer backend-secret",
        "accept": "text/event-stream",
        "payload": {
            "model": "served-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        },
    }
    assert "tenant" not in captured["authorization"]
    assert chunks == list(SSE_CHUNKS)
    assert b"".join(chunks).count(b"data: [DONE]\n\n") == 1
    assert response_body.closed is True


@pytest.mark.asyncio
async def test_stream_open_does_not_consume_body_before_returning() -> None:
    response_body = BlockingByteStream()
    backend = make_backend(
        lambda _request: httpx2.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            stream=response_body,
        )
    )
    stream = await backend.stream(chat_request(stream=True))

    assert response_body.first_produced.is_set() is False
    iterator = stream.__aiter__()
    assert await anext(iterator) == SSE_CHUNKS[0]
    assert response_body.first_produced.is_set() is True
    response_body.release_second.set()
    assert await anext(iterator) == SSE_CHUNKS[1]
    with pytest.raises(StopAsyncIteration):
        await anext(iterator)
    await backend.close()


@pytest.mark.parametrize(
    "exception_type,expected_error",
    [
        (httpx2.ConnectError, BackendUnavailableError),
        (httpx2.ReadTimeout, BackendTimeoutError),
    ],
)
@pytest.mark.asyncio
async def test_stream_open_transport_failures_are_normalized(
    exception_type: type[httpx2.RequestError],
    expected_error: type[Exception],
) -> None:
    async def handler(request: httpx2.Request) -> httpx2.Response:
        raise exception_type("stream open failed", request=request)

    backend = make_backend(handler)
    try:
        with pytest.raises(expected_error):
            await backend.stream(chat_request(stream=True))
    finally:
        await backend.close()


@pytest.mark.parametrize(
    "status_code,expected_error",
    [
        (400, BackendRequestRejectedError),
        (404, BackendRequestRejectedError),
        (422, BackendRequestRejectedError),
        (401, BackendConfigurationError),
        (403, BackendConfigurationError),
        (429, BackendHTTPError),
        (500, BackendHTTPError),
        (503, BackendHTTPError),
    ],
)
@pytest.mark.asyncio
async def test_stream_upstream_http_errors_close_response_before_raising(
    status_code: int,
    expected_error: type[Exception],
) -> None:
    response_body = TrackingByteStream()
    backend = make_backend(lambda _request: httpx2.Response(status_code, stream=response_body))
    try:
        with pytest.raises(expected_error):
            await backend.stream(chat_request(stream=True))
    finally:
        await backend.close()

    assert response_body.iterated is False
    assert response_body.closed is True


@pytest.mark.parametrize("content_type", ["application/json", "text/plain", ""])
@pytest.mark.asyncio
async def test_stream_rejects_non_sse_success_and_closes_response(content_type: str) -> None:
    response_body = TrackingByteStream()
    headers = {"Content-Type": content_type} if content_type else {}
    backend = make_backend(
        lambda _request: httpx2.Response(200, headers=headers, stream=response_body)
    )
    try:
        with pytest.raises(BackendProtocolError):
            await backend.stream(chat_request(stream=True))
    finally:
        await backend.close()

    assert response_body.iterated is False
    assert response_body.closed is True


@pytest.mark.asyncio
async def test_backend_close_closes_open_stream_without_consuming_it() -> None:
    response_body = TrackingByteStream()
    backend = make_backend(
        lambda _request: httpx2.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            stream=response_body,
        )
    )
    await backend.stream(chat_request(stream=True))

    await backend.close()

    assert response_body.iterated is False
    assert response_body.closed is True


@pytest.mark.asyncio
async def test_explicit_stream_close_releases_upstream_response() -> None:
    response_body = TrackingByteStream()
    backend = make_backend(
        lambda _request: httpx2.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            stream=response_body,
        )
    )
    stream = await backend.stream(chat_request(stream=True))

    await stream.aclose()
    await stream.aclose()
    await backend.close()

    assert response_body.iterated is False
    assert response_body.closed is True


@pytest.mark.parametrize(
    "exception_type,expected_error",
    [
        (httpx2.ReadTimeout, BackendTimeoutError),
        (httpx2.ReadError, BackendUnavailableError),
    ],
)
@pytest.mark.asyncio
async def test_midstream_transport_failures_are_normalized(
    exception_type: type[httpx2.RequestError],
    expected_error: type[Exception],
) -> None:
    response_body = FailingByteStream(exception_type)
    backend = make_backend(
        lambda _request: httpx2.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            stream=response_body,
        )
    )
    stream = await backend.stream(chat_request(stream=True))
    iterator = stream.__aiter__()

    assert await anext(iterator) == SSE_CHUNKS[0]
    with pytest.raises(expected_error):
        await anext(iterator)

    assert response_body.closed is True
    await backend.close()


@pytest.mark.parametrize("status_code,expected", [(200, True), (204, True), (503, False)])
@pytest.mark.asyncio
async def test_health_probe_uses_health_endpoint_without_generation(
    status_code: int,
    expected: bool,
) -> None:
    captured: dict[str, Any] = {}

    async def handler(request: httpx2.Request) -> httpx2.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers["Authorization"]
        captured["timeout"] = request.extensions["timeout"]
        return httpx2.Response(status_code, content=b"ignored-health-body")

    backend = make_backend(
        handler,
        api_key="backend-secret",
        backend_health_timeout_seconds=0.75,
    )
    try:
        healthy = await backend.check_health()
    finally:
        await backend.close()

    assert healthy is expected
    assert captured == {
        "method": "GET",
        "url": "https://vllm.example.test/root/health",
        "authorization": "Bearer backend-secret",
        "timeout": {"connect": 0.75, "read": 0.75, "write": 0.75, "pool": 0.75},
    }


@pytest.mark.parametrize("exception_type", [httpx2.ConnectError, httpx2.ReadTimeout])
@pytest.mark.asyncio
async def test_health_probe_transport_failure_is_unhealthy(
    exception_type: type[httpx2.RequestError],
) -> None:
    async def handler(request: httpx2.Request) -> httpx2.Response:
        raise exception_type("health failed", request=request)

    backend = make_backend(handler)
    try:
        assert await backend.check_health() is False
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_unsupported_backend_capabilities_fail_explicitly() -> None:
    backend = make_backend(lambda _request: httpx2.Response(200, json=UPSTREAM_RESPONSE))
    try:
        with pytest.raises(BackendCapabilityError):
            await backend.generate_batch([chat_request()])
    finally:
        await backend.close()


def test_application_lifespan_closes_vllm_client() -> None:
    backend = make_backend(lambda _request: httpx2.Response(200, json=UPSTREAM_RESPONSE))
    app = create_app(Settings(_env_file=None), backend=backend)

    with TestClient(app):
        assert backend.is_closed is False

    assert backend.is_closed is True
