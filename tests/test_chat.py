from collections.abc import AsyncIterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from gateway.app import create_app
from gateway.backends.base import InferenceBackend
from gateway.backends.fake import FakeBackend
from gateway.config import Settings
from gateway.core.errors import BackendTimeoutError

VALID_REQUEST: dict[str, Any] = {
    "model": "test-model",
    "messages": [{"role": "user", "content": "Hello"}],
}


def make_client(
    *,
    settings: Settings | None = None,
    backend: InferenceBackend | None = None,
) -> TestClient:
    return TestClient(
        create_app(
            settings or Settings(_env_file=None),
            backend=backend or FakeBackend(),
        )
    )


def assert_invalid_request(response: Any, *, message: str | None = None) -> None:
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"
    assert response.json()["error"]["request_id"] == response.headers["X-Request-ID"]
    if message is not None:
        assert response.json()["error"]["message"] == message


def test_valid_chat_completion_request() -> None:
    payload = {
        **VALID_REQUEST,
        "temperature": 0.5,
        "top_p": 0.9,
        "max_tokens": 32,
        "stop": ["done"],
        "seed": 7,
        "n": 2,
        "stream": False,
    }

    with make_client() as client:
        response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    assert response.json()["object"] == "chat.completion"
    assert response.json()["model"] == "test-model"
    assert len(response.json()["choices"]) == 2


class ResponseBackend:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response

    async def generate(self, _request: Any) -> dict[str, Any]:
        return self.response

    def stream(self, _request: Any) -> AsyncIterator[Any]:
        raise NotImplementedError

    async def generate_batch(self, _requests: list[Any]) -> list[Any]:
        raise NotImplementedError

    async def close(self) -> None:
        return None


def test_success_response_preserves_upstream_fields() -> None:
    upstream_response = {
        "id": "chatcmpl-upstream",
        "object": "chat.completion",
        "created": 123,
        "model": "test-model",
        "choices": [],
        "usage": {"total_tokens": 0},
        "vllm_extension": {"preserved": True},
    }

    with make_client(backend=ResponseBackend(upstream_response)) as client:
        response = client.post("/v1/chat/completions", json=VALID_REQUEST)

    assert response.status_code == 200
    assert response.json() == upstream_response


@pytest.mark.parametrize(
    "payload",
    [
        {"messages": VALID_REQUEST["messages"]},
        {**VALID_REQUEST, "model": "  "},
        {"model": "test-model"},
        {**VALID_REQUEST, "messages": []},
    ],
)
def test_required_field_validation(payload: dict[str, Any]) -> None:
    with make_client() as client:
        response = client.post("/v1/chat/completions", json=payload)

    assert_invalid_request(response)


@pytest.mark.parametrize(
    "message",
    [
        {"role": "tool", "content": "result"},
        {"role": "user", "content": ["not", "text"]},
        {"role": "user", "content": "  "},
        {"role": "user"},
        {"role": "user", "content": "hello", "name": "unsupported"},
    ],
)
def test_message_validation(message: dict[str, Any]) -> None:
    payload = {**VALID_REQUEST, "messages": [message]}

    with make_client() as client:
        response = client.post("/v1/chat/completions", json=payload)

    assert_invalid_request(response)


@pytest.mark.parametrize(
    "field,value",
    [
        ("temperature", -0.1),
        ("temperature", 2.1),
        ("top_p", -0.1),
        ("top_p", 1.1),
        ("max_tokens", 0),
        ("max_tokens", -1),
        ("n", 0),
        ("n", -1),
        ("n", "2"),
    ],
)
def test_generation_schema_validation(field: str, value: object) -> None:
    with make_client() as client:
        response = client.post(
            "/v1/chat/completions",
            json={**VALID_REQUEST, field: value},
        )

    assert_invalid_request(response)


@pytest.mark.parametrize(
    "field,value,expected_message",
    [
        ("max_tokens", 9, "max_tokens must not exceed 8"),
        ("n", 3, "n must not exceed 2"),
    ],
)
def test_configured_generation_limits(
    field: str,
    value: int,
    expected_message: str,
) -> None:
    settings = Settings(_env_file=None, max_completion_tokens=8, max_choices=2)

    with make_client(settings=settings) as client:
        response = client.post(
            "/v1/chat/completions",
            json={**VALID_REQUEST, field: value},
        )

    assert_invalid_request(response, message=expected_message)


def test_streaming_is_explicitly_rejected() -> None:
    with make_client() as client:
        response = client.post(
            "/v1/chat/completions",
            json={**VALID_REQUEST, "stream": True},
        )

    assert_invalid_request(
        response,
        message="Streaming chat completions are not supported",
    )


def test_unsupported_request_field_is_rejected() -> None:
    with make_client() as client:
        response = client.post(
            "/v1/chat/completions",
            json={**VALID_REQUEST, "tools": []},
        )

    assert_invalid_request(response)


def test_malformed_json_is_normalized() -> None:
    with make_client() as client:
        response = client.post(
            "/v1/chat/completions",
            content=b'{"model":',
            headers={"Content-Type": "application/json"},
        )

    assert_invalid_request(response, message="Invalid chat completion request")
    assert "json_invalid" not in response.text


def test_injected_backend_is_closed_on_shutdown() -> None:
    backend = FakeBackend()

    with make_client(backend=backend) as client:
        assert backend.closed is False
        assert client.get("/readyz").status_code == 200

    assert backend.closed is True


class TimeoutBackend:
    async def generate(self, _request: Any) -> Any:
        raise BackendTimeoutError()

    def stream(self, _request: Any) -> AsyncIterator[Any]:
        raise NotImplementedError

    async def generate_batch(self, _requests: list[Any]) -> list[Any]:
        raise NotImplementedError

    async def close(self) -> None:
        return None


def test_backend_error_is_normalized_by_route() -> None:
    app = create_app(Settings(_env_file=None), backend=TimeoutBackend())

    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=VALID_REQUEST)

    assert response.status_code == 504
    assert response.json()["error"]["code"] == "backend_timeout"
    assert response.json()["error"]["message"] == "Inference backend timed out"
