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
    clock: Any | None = None,
) -> VLLMBackend:
    clock_options = {} if clock is None else {"clock": clock}
    return VLLMBackend(
        backend_id,
        BackendConfig(base_url=base_url, api_key=api_key),
        connect_timeout_seconds=vllm_connect_timeout_seconds,
        request_timeout_seconds=vllm_request_timeout_seconds,
        health_timeout_seconds=backend_health_timeout_seconds,
        transport=httpx2.MockTransport(handler),
        **clock_options,
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
async def test_stream_forwards_usage_options_and_captures_upstream_start() -> None:
    captured: dict[str, Any] = {}

    async def handler(request: httpx2.Request) -> httpx2.Response:
        captured["payload"] = json.loads(request.content)
        return httpx2.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            stream=TrackingByteStream(),
        )

    backend = make_backend(handler, backend_id="gpu-a", clock=lambda: 42.5)
    stream = await backend.stream(chat_request(stream=True, stream_options={"include_usage": True}))
    try:
        assert stream.backend_id == "gpu-a"
        assert stream.upstream_request_started_at == 42.5
        assert captured["payload"]["stream_options"] == {"include_usage": True}
    finally:
        await stream.aclose()
        await backend.close()


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
async def test_generate_batch_sends_one_exact_request_with_backend_credentials() -> None:
    captured: list[dict[str, Any]] = []

    async def handler(request: httpx2.Request) -> httpx2.Response:
        captured.append(
            {
                "method": request.method,
                "url": str(request.url),
                "authorization": request.headers.get("Authorization"),
                "payload": json.loads(request.content),
            }
        )
        return httpx2.Response(
            200,
            json={
                "id": "chatcmpl-shared",
                "object": "chat.completion",
                "created": 123,
                "model": "served-model",
                "choices": [
                    {
                        "index": index,
                        "message": {"role": "assistant", "content": f"result-{index}"},
                        "finish_reason": "stop",
                    }
                    for index in range(3)
                ],
                "usage": {"prompt_tokens": 9, "completion_tokens": 27, "total_tokens": 36},
            },
        )

    backend = make_backend(handler, api_key="backend-secret")
    requests = [
        chat_request(
            messages=[{"role": "user", "content": f"private-{index}"}],
            temperature=0.7,
            max_tokens=64,
            n=1,
            stream=False,
        )
        for index in range(3)
    ]
    try:
        result = await backend.generate_batch(requests)
    finally:
        await backend.close()

    assert captured == [
        {
            "method": "POST",
            "url": "https://vllm.example.test/root/v1/chat/completions/batch",
            "authorization": "Bearer backend-secret",
            "payload": {
                "model": "served-model",
                "temperature": 0.7,
                "max_tokens": 64,
                "messages": [
                    [{"role": "user", "content": "private-0"}],
                    [{"role": "user", "content": "private-1"}],
                    [{"role": "user", "content": "private-2"}],
                ],
            },
        }
    ]
    assert result.aggregate_completion_tokens == 27
    assert result.usage_result == "observed"
    assert all("usage" not in response for response in result.responses)
    assert {response["id"] for response in result.responses} == {"chatcmpl-shared"}


@pytest.mark.asyncio
async def test_generate_batch_demultiplexes_out_of_order_without_response_leakage() -> None:
    choices = [
        {
            "index": index,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }
        for index, content in ((2, "gamma-result"), (0, "alpha-result"), (1, "beta-result"))
    ]
    backend = make_backend(
        lambda _request: httpx2.Response(
            200,
            json={**UPSTREAM_RESPONSE, "choices": choices},
        )
    )
    try:
        result = await backend.generate_batch(
            [chat_request(messages=[{"role": "user", "content": value}]) for value in "abc"]
        )
    finally:
        await backend.close()

    assert [response["choices"][0]["message"]["content"] for response in result.responses] == [
        "alpha-result",
        "beta-result",
        "gamma-result",
    ]
    assert all(response["choices"][0]["index"] == 0 for response in result.responses)
    assert all("usage" not in response for response in result.responses)


@pytest.mark.parametrize(
    "choices",
    [
        None,
        [{"index": 0}, {"index": 0}],
        [{"index": 0}, {}],
        [{"index": 0}, {"index": 2}],
        [{"index": 0}],
        [{"index": 0}, {"index": "1"}],
        [{"index": 0}, {"index": True}],
        [{"index": -1}, {"index": 1}],
    ],
)
@pytest.mark.asyncio
async def test_generate_batch_rejects_malformed_choice_association(choices: Any) -> None:
    payload = {"id": "chatcmpl-batch"}
    if choices is not None:
        payload["choices"] = choices
    backend = make_backend(lambda _request: httpx2.Response(200, json=payload))
    try:
        with pytest.raises(BackendProtocolError):
            await backend.generate_batch([chat_request(), chat_request()])
    finally:
        await backend.close()


@pytest.mark.parametrize(
    "requests",
    [
        [],
        [chat_request(stream=True)],
        [chat_request(n=2)],
        [chat_request(temperature=0.7), chat_request(temperature=0.8)],
        [chat_request(), chat_request(temperature=1.0)],
    ],
)
@pytest.mark.asyncio
async def test_generate_batch_defensively_rejects_incompatible_members(
    requests: list[ChatCompletionRequest],
) -> None:
    called = False

    async def handler(_request: httpx2.Request) -> httpx2.Response:
        nonlocal called
        called = True
        return httpx2.Response(200, json=UPSTREAM_RESPONSE)

    backend = make_backend(handler)
    try:
        with pytest.raises(BackendCapabilityError):
            await backend.generate_batch(requests)
    finally:
        await backend.close()

    assert called is False


@pytest.mark.parametrize(
    "usage,expected_result",
    [
        (None, "missing"),
        ({}, "invalid"),
        ({"completion_tokens": -1}, "invalid"),
        ({"completion_tokens": True}, "invalid"),
    ],
)
@pytest.mark.asyncio
async def test_generate_batch_classifies_aggregate_usage_without_member_copy(
    usage: Any,
    expected_result: str,
) -> None:
    payload = {
        "id": "chatcmpl-batch",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
    }
    if usage is not None:
        payload["usage"] = usage
    backend = make_backend(lambda _request: httpx2.Response(200, json=payload))
    try:
        result = await backend.generate_batch([chat_request()])
    finally:
        await backend.close()

    assert result.usage_result == expected_result
    assert result.aggregate_completion_tokens is None
    assert "usage" not in result.responses[0]


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
    ],
)
@pytest.mark.asyncio
async def test_generate_batch_normalizes_upstream_http_failures(
    status_code: int,
    expected_error: type[Exception],
) -> None:
    backend = make_backend(lambda _request: httpx2.Response(status_code))
    try:
        with pytest.raises(expected_error):
            await backend.generate_batch([chat_request()])
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
async def test_generate_batch_normalizes_transport_failures(
    exception_type: type[httpx2.RequestError],
    expected_error: type[Exception],
) -> None:
    async def handler(request: httpx2.Request) -> httpx2.Response:
        raise exception_type("batch transport failed", request=request)

    backend = make_backend(handler)
    try:
        with pytest.raises(expected_error):
            await backend.generate_batch([chat_request()])
    finally:
        await backend.close()


@pytest.mark.parametrize(
    "response",
    [
        httpx2.Response(200, content=b"not-json"),
        httpx2.Response(200, json=["not", "an", "object"]),
    ],
)
@pytest.mark.asyncio
async def test_generate_batch_rejects_invalid_json_success(response: httpx2.Response) -> None:
    backend = make_backend(lambda _request: response)
    try:
        with pytest.raises(BackendProtocolError):
            await backend.generate_batch([chat_request()])
    finally:
        await backend.close()


def test_application_lifespan_closes_vllm_client() -> None:
    backend = make_backend(lambda _request: httpx2.Response(200, json=UPSTREAM_RESPONSE))
    app = create_app(Settings(_env_file=None), backend=backend)

    with TestClient(app):
        assert backend.is_closed is False

    assert backend.is_closed is True
