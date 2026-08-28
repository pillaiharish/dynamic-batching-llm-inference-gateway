import asyncio
import json
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.types import Message, Scope

from gateway.admission.controller import AdmissionController
from gateway.admission.models import AdmissionSnapshot
from gateway.app import create_app
from gateway.auth.tenants import TenantContext
from gateway.backends.base import BackendStream
from gateway.backends.fake import FakeBackend
from gateway.config import Settings
from gateway.core.errors import BackendTimeoutError, BackendUnavailableError
from gateway.routing.pool import BackendPool
from gateway.schemas.chat import ChatCompletionRequest
from gateway.streaming.relay import StreamingRelay

AUTHORIZATION = b"Bearer tenant-a-key"
DONE = b"data: [DONE]\n\n"


def make_settings(**overrides: Any) -> Settings:
    return Settings(
        _env_file=None,
        tenants_json={
            "tenant-a": {
                "api_key": "tenant-a-key",
                "max_inflight": 2,
                "max_queue": 4,
            }
        },
        global_max_inflight=1,
        global_max_queue=4,
        **overrides,
    )


class ControlledBackendStream:
    def __init__(
        self,
        request_id: str,
        *,
        fail_after_first: bool = False,
        failure: Exception | None = None,
    ) -> None:
        self.request_id = request_id
        self.fail_after_first = fail_after_first
        self.failure = failure
        self.first_produced = asyncio.Event()
        self.allow_finish = asyncio.Event()
        self.second_produced = asyncio.Event()
        self.closed = asyncio.Event()
        self.close_count = 0

    @property
    def first_chunk(self) -> bytes:
        return f'data: {{"request":"{self.request_id}","delta":"first"}}\n\n'.encode()

    @property
    def second_chunk(self) -> bytes:
        return f'data: {{"request":"{self.request_id}","delta":"second"}}\n\n'.encode()

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[bytes]:
        try:
            self.first_produced.set()
            yield self.first_chunk
            await self.allow_finish.wait()
            if self.failure is not None:
                raise self.failure
            if self.fail_after_first:
                raise RuntimeError("simulated upstream failure: private-stream-content")
            self.second_produced.set()
            yield self.second_chunk
            yield DONE
        finally:
            await self.aclose()

    async def aclose(self) -> None:
        if self.closed.is_set():
            return
        self.close_count += 1
        self.allow_finish.set()
        self.closed.set()


class ControlledStreamingBackend:
    def __init__(
        self,
        *,
        fail_after_first: set[str] | None = None,
        backend_fail_after_first: set[str] | None = None,
    ) -> None:
        self.fail_after_first = fail_after_first or set()
        self.backend_fail_after_first = backend_fail_after_first or set()
        self.streams: dict[str, ControlledBackendStream] = {}
        self.opened: asyncio.Queue[str] = asyncio.Queue()
        self.generate_calls = 0
        self.closed = False

    async def generate(self, request: ChatCompletionRequest) -> dict[str, Any]:
        self.generate_calls += 1
        return {"object": "chat.completion", "model": request.model, "choices": []}

    async def stream(self, request: ChatCompletionRequest) -> BackendStream:
        request_id = request.messages[-1].content
        stream = ControlledBackendStream(
            request_id,
            fail_after_first=request_id in self.fail_after_first,
            failure=(
                BackendUnavailableError() if request_id in self.backend_fail_after_first else None
            ),
        )
        self.streams[request_id] = stream
        self.opened.put_nowait(request_id)
        return stream

    async def generate_batch(self, _requests: list[Any]) -> list[Any]:
        raise NotImplementedError

    async def check_health(self) -> bool:
        return not self.closed

    async def close(self) -> None:
        for stream in self.streams.values():
            await stream.aclose()
        self.closed = True


class ASGIRequest:
    def __init__(self, app: FastAPI, request_id: str) -> None:
        self.app = app
        self.request_id = request_id
        self.incoming: asyncio.Queue[Message] = asyncio.Queue()
        self.outgoing: asyncio.Queue[Message] = asyncio.Queue()
        self.task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        body = json.dumps(
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": self.request_id}],
                "stream": True,
            }
        ).encode()
        self.incoming.put_nowait({"type": "http.request", "body": body, "more_body": False})
        self.task = asyncio.create_task(self.app(self._scope(), self._receive, self._send))

    async def next_message(self) -> Message:
        return await asyncio.wait_for(self.outgoing.get(), timeout=1)

    async def disconnect(self) -> None:
        self.incoming.put_nowait({"type": "http.disconnect"})

    async def wait_finished(self) -> None:
        assert self.task is not None
        await asyncio.wait_for(self.task, timeout=1)

    async def _receive(self) -> Message:
        return await self.incoming.get()

    async def _send(self, message: Message) -> None:
        self.outgoing.put_nowait(message)

    def _scope(self) -> Scope:
        return {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/chat/completions",
            "raw_path": b"/v1/chat/completions",
            "query_string": b"",
            "root_path": "",
            "headers": [
                (b"host", b"testserver"),
                (b"content-type", b"application/json"),
                (b"authorization", AUTHORIZATION),
                (b"x-request-id", self.request_id.encode()),
            ],
            "client": ("127.0.0.1", 12345),
            "server": ("testserver", 80),
            "state": {},
        }


async def wait_for_snapshot(
    controller: AdmissionController,
    predicate: Callable[[AdmissionSnapshot], bool],
) -> AdmissionSnapshot:
    for _ in range(100):
        snapshot = await controller.snapshot()
        if predicate(snapshot):
            return snapshot
        await asyncio.sleep(0)
    raise AssertionError(f"admission state did not reach expected condition: {snapshot}")


async def finish_stream(request: ASGIRequest, stream: ControlledBackendStream) -> list[bytes]:
    stream.allow_finish.set()
    bodies: list[bytes] = []
    while True:
        message = await request.next_message()
        assert message["type"] == "http.response.body"
        body = message.get("body", b"")
        if body:
            bodies.append(body)
        if not message.get("more_body", False):
            break
    await request.wait_finished()
    return bodies


@pytest.mark.asyncio
async def test_route_relays_incrementally_and_preserves_done() -> None:
    backend = ControlledStreamingBackend()
    app = create_app(make_settings(), backend=backend)

    async with app.router.lifespan_context(app):
        request = ASGIRequest(app, "S1")
        await request.start()
        start = await request.next_message()
        first = await request.next_message()
        stream = backend.streams["S1"]

        assert start["type"] == "http.response.start"
        assert start["status"] == 200
        headers = dict(start["headers"])
        assert headers[b"content-type"] == b"text/event-stream"
        assert headers[b"cache-control"] == b"no-cache"
        assert headers[b"x-accel-buffering"] == b"no"
        assert headers[b"x-request-id"] == b"S1"
        assert b"content-length" not in headers
        assert b"connection" not in headers
        assert b"transfer-encoding" not in headers
        assert first == {
            "type": "http.response.body",
            "body": stream.first_chunk,
            "more_body": True,
        }
        assert stream.first_produced.is_set() is True
        assert stream.second_produced.is_set() is False

        remaining = await finish_stream(request, stream)
        assert remaining == [stream.second_chunk, DONE]
        assert (stream.first_chunk + b"".join(remaining)).count(DONE) == 1
        assert stream.close_count == 1
        snapshot = await app.state.admission_controller.snapshot()
        assert snapshot.global_inflight == 0


@pytest.mark.asyncio
async def test_stream_holds_admission_until_full_response_finishes() -> None:
    backend = ControlledStreamingBackend()
    app = create_app(make_settings(), backend=backend)

    async with app.router.lifespan_context(app):
        first_request = ASGIRequest(app, "S1")
        await first_request.start()
        await first_request.next_message()
        await first_request.next_message()

        second_request = ASGIRequest(app, "S2")
        await second_request.start()
        snapshot = await wait_for_snapshot(
            app.state.admission_controller,
            lambda state: state.global_queued == 1,
        )
        assert snapshot.global_inflight == 1
        assert "S2" not in backend.streams

        first_stream = backend.streams["S1"]
        await finish_stream(first_request, first_stream)
        second_start = await second_request.next_message()
        second_first = await second_request.next_message()
        second_stream = backend.streams["S2"]
        assert second_start["type"] == "http.response.start"
        assert second_first["body"] == second_stream.first_chunk

        await finish_stream(second_request, second_stream)
        snapshot = await app.state.admission_controller.snapshot()
        assert snapshot.global_inflight == 0
        assert snapshot.global_queued == 0


@pytest.mark.asyncio
async def test_client_disconnect_closes_upstream_and_releases_admission() -> None:
    backend = ControlledStreamingBackend()
    app = create_app(make_settings(), backend=backend)

    async with app.router.lifespan_context(app):
        request = ASGIRequest(app, "disconnect")
        await request.start()
        await request.next_message()
        await request.next_message()
        stream = backend.streams["disconnect"]
        assert stream.second_produced.is_set() is False

        await request.disconnect()
        await request.wait_finished()
        await asyncio.wait_for(stream.closed.wait(), timeout=1)

        snapshot = await app.state.admission_controller.snapshot()
        assert stream.close_count == 1
        assert stream.second_produced.is_set() is False
        assert snapshot.global_inflight == 0
        assert snapshot.tenants["tenant-a"].inflight == 0


@pytest.mark.asyncio
async def test_client_disconnect_releases_router_and_admission_without_ejection() -> None:
    backend = ControlledStreamingBackend()
    pool = BackendPool(
        {"gpu-a": backend},
        health_interval_seconds=60,
        health_timeout_seconds=1,
    )
    app = create_app(make_settings(), backend=pool)

    async with app.router.lifespan_context(app):
        request = ASGIRequest(app, "routed-disconnect")
        await request.start()
        await request.next_message()
        await request.next_message()
        stream = backend.streams["routed-disconnect"]

        routing_active = await pool.snapshot()
        admission_active = await app.state.admission_controller.snapshot()
        assert routing_active.backends["gpu-a"].inflight == 1
        assert admission_active.global_inflight == 1

        await request.disconnect()
        await request.wait_finished()
        await asyncio.wait_for(stream.closed.wait(), timeout=1)

        routing_finished = await pool.snapshot()
        admission_finished = await app.state.admission_controller.snapshot()
        assert routing_finished.backends["gpu-a"].inflight == 0
        assert routing_finished.backends["gpu-a"].healthy is True
        assert admission_finished.global_inflight == 0
        assert admission_finished.tenants["tenant-a"].inflight == 0
        assert stream.close_count == 1


@pytest.mark.asyncio
async def test_midstream_backend_failure_isolated_through_http_relay() -> None:
    backend_a = ControlledStreamingBackend()
    backend_b = ControlledStreamingBackend(backend_fail_after_first={"failure-on-b"})
    pool = BackendPool(
        {"gpu-a": backend_a, "gpu-b": backend_b},
        health_interval_seconds=60,
        health_timeout_seconds=1,
    )
    app = create_app(make_settings(), backend=pool)
    warmup = ChatCompletionRequest.model_validate(
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "warmup"}],
        }
    )

    async with app.router.lifespan_context(app):
        await pool.generate(warmup)
        request = ASGIRequest(app, "failure-on-b")
        await request.start()
        start = await request.next_message()
        first = await request.next_message()
        stream = backend_b.streams["failure-on-b"]
        remaining = await finish_stream(request, stream)

        assert start["status"] == 200
        assert first["body"] == stream.first_chunk
        assert remaining == []
        assert "failure-on-b" not in backend_a.streams

        routing = await pool.snapshot()
        admission = await app.state.admission_controller.snapshot()
        assert routing.backends["gpu-b"].healthy is False
        assert routing.backends["gpu-b"].inflight == 0
        assert admission.global_inflight == 0

        await pool.generate(warmup)
        assert backend_a.generate_calls == 2
        assert backend_b.generate_calls == 0


@pytest.mark.asyncio
async def test_streaming_admission_rejection_remains_json_before_sse_starts() -> None:
    settings = Settings(
        _env_file=None,
        tenants_json={
            "tenant-a": {
                "api_key": "tenant-a-key",
                "max_inflight": 2,
                "max_queue": 0,
            }
        },
        global_max_inflight=1,
        global_max_queue=1,
    )
    backend = ControlledStreamingBackend()
    app = create_app(settings, backend=backend)

    async with app.router.lifespan_context(app):
        active_request = ASGIRequest(app, "active")
        await active_request.start()
        await active_request.next_message()
        await active_request.next_message()

        rejected_request = ASGIRequest(app, "rejected")
        await rejected_request.start()
        rejected_start = await rejected_request.next_message()
        rejected_body = await rejected_request.next_message()
        await rejected_request.wait_finished()

        assert rejected_start["status"] == 429
        assert dict(rejected_start["headers"])[b"content-type"].startswith(b"application/json")
        assert json.loads(rejected_body["body"])["error"]["code"] == "tenant_queue_full"
        assert "rejected" not in backend.streams

        await active_request.disconnect()
        await active_request.wait_finished()


@pytest.mark.asyncio
async def test_direct_relay_cancellation_closes_stream_and_releases_lease() -> None:
    tenant = TenantContext("tenant-a", max_inflight=1, max_queue=1)
    controller = AdmissionController(
        [tenant],
        global_max_inflight=1,
        global_max_queue=1,
        queue_timeout_seconds=1,
    )
    lease = await controller.acquire(tenant)
    stream = ControlledBackendStream("cancelled")
    relay = StreamingRelay(
        stream,
        lease,
        tenant_id=tenant.tenant_id,
        request_id="cancelled-request-id",
    )

    async def consume() -> None:
        async for _chunk in relay:
            pass

    task = asyncio.create_task(consume())
    await asyncio.wait_for(stream.first_produced.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    snapshot = await controller.snapshot()
    assert stream.closed.is_set() is True
    assert stream.close_count == 1
    assert snapshot.global_inflight == 0


@pytest.mark.asyncio
async def test_midstream_failure_ends_without_json_error_and_cleans_up(
    capsys: pytest.CaptureFixture[str],
) -> None:
    backend = ControlledStreamingBackend(fail_after_first={"failure"})
    app = create_app(make_settings(), backend=backend)

    async with app.router.lifespan_context(app):
        request = ASGIRequest(app, "failure")
        await request.start()
        start = await request.next_message()
        first = await request.next_message()
        stream = backend.streams["failure"]
        remaining = await finish_stream(request, stream)

        assert dict(start["headers"])[b"content-type"] == b"text/event-stream"
        assert first["body"] == stream.first_chunk
        assert remaining == []
        assert stream.close_count == 1
        snapshot = await app.state.admission_controller.snapshot()
        assert snapshot.global_inflight == 0

    captured = capsys.readouterr()
    assert "private-stream-content" not in captured.out
    assert '"stream_outcome":"upstream_error"' in captured.out
    assert '"stream_error_type":"RuntimeError"' in captured.out


class FailFirstStreamOpenBackend(FakeBackend):
    stream_calls: int = 0

    async def stream(self, request: Any) -> BackendStream:
        self.stream_calls += 1
        if self.stream_calls == 1:
            raise BackendTimeoutError()
        return await super().stream(request)


def test_pre_stream_failure_is_json_and_releases_admission_slot() -> None:
    backend = FailFirstStreamOpenBackend()
    app = create_app(make_settings(), backend=backend)
    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
    }

    with TestClient(app, headers={"Authorization": AUTHORIZATION.decode()}) as client:
        failed = client.post("/v1/chat/completions", json=payload)
        succeeded = client.post("/v1/chat/completions", json=payload)

    assert failed.status_code == 504
    assert failed.headers["Content-Type"].startswith("application/json")
    assert failed.json()["error"]["code"] == "backend_timeout"
    assert succeeded.status_code == 200
    assert succeeded.headers["Content-Type"] == "text/event-stream"
    assert backend.stream_calls == 2


def test_streaming_still_requires_tenant_authentication() -> None:
    app = create_app(make_settings(), backend=FakeBackend())

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"
    assert response.json()["error"]["code"] == "unauthorized"


def test_streaming_logs_lifecycle_metadata_without_content(
    capsys: pytest.CaptureFixture[str],
) -> None:
    backend = FakeBackend(
        stream_chunks=(b"data: private-generated-content\n\n", DONE),
    )
    app = create_app(make_settings(), backend=backend)

    with TestClient(app, headers={"Authorization": AUTHORIZATION.decode()}) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "private-prompt-content"}],
                "stream": True,
            },
        )

    assert response.status_code == 200
    captured = capsys.readouterr()
    assert "private-prompt-content" not in captured.out
    assert "private-generated-content" not in captured.out
    assert "tenant-a-key" not in captured.out
    assert '"stream_outcome":"completed"' in captured.out
