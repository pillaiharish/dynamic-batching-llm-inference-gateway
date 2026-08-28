import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from gateway.backends.base import BackendStream
from gateway.core.errors import (
    BackendRequestRejectedError,
    BackendTimeoutError,
    BackendUnavailableError,
    NoHealthyBackendError,
)
from gateway.routing.pool import BackendPool

DONE = b"data: [DONE]\n\n"


class ControlledStream:
    def __init__(
        self,
        backend_id: str,
        request: str,
        *,
        failure: Exception | None = None,
        auto_finish: bool = False,
    ) -> None:
        self.first_chunk = f"data: {backend_id}:{request}\n\n".encode()
        self.failure = failure
        self.auto_finish = auto_finish
        self.first_produced = asyncio.Event()
        self.allow_finish = asyncio.Event()
        self.closed = asyncio.Event()
        self.close_count = 0

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[bytes]:
        try:
            self.first_produced.set()
            yield self.first_chunk
            if not self.auto_finish:
                await self.allow_finish.wait()
            if self.failure is not None:
                raise self.failure
            yield DONE
        finally:
            await self.aclose()

    async def aclose(self) -> None:
        if self.closed.is_set():
            return
        self.close_count += 1
        self.allow_finish.set()
        self.closed.set()


class ControlledBackend:
    def __init__(
        self,
        backend_id: str,
        *,
        healthy: bool = True,
        block_generate: bool = False,
        auto_finish_stream: bool = False,
    ) -> None:
        self.backend_id = backend_id
        self.healthy = healthy
        self.block_generate = block_generate
        self.auto_finish_stream = auto_finish_stream
        self.generate_error: Exception | None = None
        self.stream_open_error: Exception | None = None
        self.stream_failure: Exception | None = None
        self.generate_calls: list[str] = []
        self.stream_calls: list[str] = []
        self.health_calls = 0
        self.generate_started: asyncio.Queue[str] = asyncio.Queue()
        self.release_generate = asyncio.Event()
        self.probe_started = asyncio.Event()
        self.release_probe = asyncio.Event()
        self.block_probe = False
        self.probe_target = 0
        self.probe_target_reached = asyncio.Event()
        self.streams: list[ControlledStream] = []
        self.closed = False

    async def generate(self, request: str) -> dict[str, str]:
        self.generate_calls.append(request)
        self.generate_started.put_nowait(request)
        if self.block_generate:
            await self.release_generate.wait()
        else:
            await asyncio.sleep(0)
        if self.generate_error is not None:
            raise self.generate_error
        return {"backend_id": self.backend_id, "request": request}

    async def stream(self, request: str) -> BackendStream:
        self.stream_calls.append(request)
        if self.stream_open_error is not None:
            raise self.stream_open_error
        stream = ControlledStream(
            self.backend_id,
            request,
            failure=self.stream_failure,
            auto_finish=self.auto_finish_stream,
        )
        self.streams.append(stream)
        return stream

    async def generate_batch(self, _requests: list[Any]) -> list[Any]:
        raise NotImplementedError

    async def check_health(self) -> bool:
        self.health_calls += 1
        if self.probe_target and self.health_calls >= self.probe_target:
            self.probe_target_reached.set()
        if self.block_probe:
            self.probe_started.set()
            await self.release_probe.wait()
        return self.healthy

    async def close(self) -> None:
        for stream in self.streams:
            await stream.aclose()
        self.release_generate.set()
        self.release_probe.set()
        self.closed = True


def make_pool(
    *backends: ControlledBackend,
    health_interval_seconds: float = 60,
    health_timeout_seconds: float = 1,
) -> BackendPool:
    return BackendPool(
        {backend.backend_id: backend for backend in backends},
        health_interval_seconds=health_interval_seconds,
        health_timeout_seconds=health_timeout_seconds,
    )


@pytest.mark.asyncio
async def test_initial_probe_tracks_health_independently() -> None:
    backend_a = ControlledBackend("A", healthy=True)
    backend_b = ControlledBackend("B", healthy=False)
    pool = make_pool(backend_a, backend_b)

    await pool.probe_once()
    snapshot = await pool.snapshot()

    assert snapshot.backends["A"].healthy is True
    assert snapshot.backends["B"].healthy is False
    assert backend_a.health_calls == backend_b.health_calls == 1
    await pool.close()


@pytest.mark.asyncio
async def test_health_probes_run_concurrently_and_never_consume_inflight() -> None:
    backend_a = ControlledBackend("A")
    backend_b = ControlledBackend("B")
    backend_a.block_probe = backend_b.block_probe = True
    pool = make_pool(backend_a, backend_b)

    probe = asyncio.create_task(pool.probe_once())
    await asyncio.wait_for(backend_a.probe_started.wait(), timeout=1)
    await asyncio.wait_for(backend_b.probe_started.wait(), timeout=1)
    snapshot = await pool.snapshot()

    assert all(slot.inflight == 0 for slot in snapshot.backends.values())
    backend_a.release_probe.set()
    backend_b.release_probe.set()
    await probe
    await pool.close()


@pytest.mark.asyncio
async def test_equal_inflight_tie_break_rotates_in_configured_order() -> None:
    backends = [ControlledBackend(backend_id) for backend_id in ("A", "B", "C")]
    pool = make_pool(*backends)
    await pool.probe_once()

    results = [await pool.generate(f"request-{index}") for index in range(4)]

    assert [result["backend_id"] for result in results] == ["A", "B", "C", "A"]
    await pool.close()


@pytest.mark.asyncio
async def test_least_inflight_prefers_idle_backend() -> None:
    backend_a = ControlledBackend("A", block_generate=True)
    backend_b = ControlledBackend("B")
    pool = make_pool(backend_a, backend_b)
    await pool.probe_once()

    active = asyncio.create_task(pool.generate("held-on-a"))
    assert await asyncio.wait_for(backend_a.generate_started.get(), timeout=1) == "held-on-a"
    snapshot = await pool.snapshot()
    assert snapshot.backends["A"].inflight == 1

    result = await pool.generate("routes-to-idle")
    assert result["backend_id"] == "B"
    assert backend_b.generate_calls == ["routes-to-idle"]

    backend_a.release_generate.set()
    await active
    assert all(slot.inflight == 0 for slot in (await pool.snapshot()).backends.values())
    await pool.close()


@pytest.mark.asyncio
async def test_unhealthy_filter_precedes_inflight_selection() -> None:
    backend_a = ControlledBackend("A", healthy=True)
    backend_b = ControlledBackend("B", healthy=False)
    pool = make_pool(backend_a, backend_b)
    await pool.probe_once()

    results = [await pool.generate(str(index)) for index in range(3)]

    assert [result["backend_id"] for result in results] == ["A", "A", "A"]
    assert backend_b.generate_calls == []
    await pool.close()


@pytest.mark.asyncio
async def test_no_healthy_backend_rejects_generate_and_stream() -> None:
    backend_a = ControlledBackend("A", healthy=False)
    backend_b = ControlledBackend("B", healthy=False)
    pool = make_pool(backend_a, backend_b)
    await pool.probe_once()

    with pytest.raises(NoHealthyBackendError):
        await pool.generate("json")
    with pytest.raises(NoHealthyBackendError):
        await pool.stream("sse")

    assert backend_a.generate_calls == backend_b.generate_calls == []
    assert backend_a.stream_calls == backend_b.stream_calls == []
    await pool.close()


@pytest.mark.asyncio
async def test_nonstreaming_failure_is_isolated_without_retry() -> None:
    backend_a = ControlledBackend("A")
    backend_b = ControlledBackend("B")
    backend_a.generate_error = BackendUnavailableError()
    pool = make_pool(backend_a, backend_b)
    await pool.probe_once()

    with pytest.raises(BackendUnavailableError):
        await pool.generate("fails-on-a")

    failed_snapshot = await pool.snapshot()
    assert backend_a.generate_calls == ["fails-on-a"]
    assert backend_b.generate_calls == []
    assert failed_snapshot.backends["A"].healthy is False
    assert failed_snapshot.backends["A"].inflight == 0

    result = await pool.generate("future-request")
    assert result["backend_id"] == "B"
    await pool.close()


@pytest.mark.asyncio
async def test_client_request_rejection_does_not_eject_backend() -> None:
    backend = ControlledBackend("A")
    backend.generate_error = BackendRequestRejectedError(status_code=400)
    pool = make_pool(backend)
    await pool.probe_once()

    with pytest.raises(BackendRequestRejectedError):
        await pool.generate("invalid-client-request")

    snapshot = await pool.snapshot()
    assert snapshot.backends["A"].healthy is True
    assert snapshot.backends["A"].inflight == 0
    await pool.close()


@pytest.mark.asyncio
async def test_successful_probe_recovers_backend_and_restores_traffic() -> None:
    backend_a = ControlledBackend("A")
    backend_b = ControlledBackend("B")
    backend_a.generate_error = BackendTimeoutError()
    pool = make_pool(backend_a, backend_b)
    await pool.probe_once()

    with pytest.raises(BackendTimeoutError):
        await pool.generate("failure")
    assert (await pool.snapshot()).backends["A"].healthy is False
    assert (await pool.generate("survivor"))["backend_id"] == "B"

    backend_a.generate_error = None
    backend_a.healthy = True
    await pool.probe_once()
    recovered = await pool.generate("recovered")

    assert recovered["backend_id"] == "A"
    assert backend_a.generate_calls == ["failure", "recovered"]
    await pool.close()


@pytest.mark.asyncio
async def test_stream_holds_backend_inflight_until_eof() -> None:
    backend_a = ControlledBackend("A")
    backend_b = ControlledBackend("B")
    pool = make_pool(backend_a, backend_b)
    await pool.probe_once()

    stream = await pool.stream("streaming")
    iterator = stream.__aiter__()
    assert (await pool.snapshot()).backends["A"].inflight == 1
    assert await anext(iterator) == backend_a.streams[0].first_chunk
    assert (await pool.snapshot()).backends["A"].inflight == 1

    backend_a.streams[0].allow_finish.set()
    assert await anext(iterator) == DONE
    with pytest.raises(StopAsyncIteration):
        await anext(iterator)

    assert (await pool.snapshot()).backends["A"].inflight == 0
    assert backend_a.streams[0].close_count == 1
    await pool.close()


@pytest.mark.asyncio
async def test_stream_cancellation_releases_inflight_without_ejecting_backend() -> None:
    backend = ControlledBackend("A")
    pool = make_pool(backend)
    await pool.probe_once()
    stream = await pool.stream("cancelled")
    first_received = asyncio.Event()

    async def consume() -> None:
        async for _chunk in stream:
            first_received.set()

    task = asyncio.create_task(consume())
    await asyncio.wait_for(first_received.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    snapshot = await pool.snapshot()
    assert snapshot.backends["A"].inflight == 0
    assert snapshot.backends["A"].healthy is True
    assert backend.streams[0].closed.is_set()
    await pool.close()


@pytest.mark.asyncio
async def test_midstream_backend_failure_ejects_backend_without_retry() -> None:
    backend_a = ControlledBackend("A")
    backend_b = ControlledBackend("B")
    backend_b.stream_failure = BackendUnavailableError()
    pool = make_pool(backend_a, backend_b)
    await pool.probe_once()

    assert (await pool.generate("rotate-to-b"))["backend_id"] == "A"
    stream = await pool.stream("fails-on-b")
    iterator = stream.__aiter__()
    assert await anext(iterator) == backend_b.streams[0].first_chunk
    backend_b.streams[0].allow_finish.set()
    with pytest.raises(BackendUnavailableError):
        await anext(iterator)

    snapshot = await pool.snapshot()
    assert snapshot.backends["B"].healthy is False
    assert snapshot.backends["B"].inflight == 0
    assert backend_b.stream_calls == ["fails-on-b"]
    assert (await pool.generate("future-request"))["backend_id"] == "A"
    await pool.close()


@pytest.mark.asyncio
async def test_stream_open_failure_releases_inflight_and_is_not_retried() -> None:
    backend_a = ControlledBackend("A")
    backend_b = ControlledBackend("B")
    backend_a.stream_open_error = BackendTimeoutError()
    pool = make_pool(backend_a, backend_b)
    await pool.probe_once()

    with pytest.raises(BackendTimeoutError):
        await pool.stream("open-failure")

    snapshot = await pool.snapshot()
    assert snapshot.backends["A"].inflight == 0
    assert snapshot.backends["A"].healthy is False
    assert backend_b.stream_calls == []
    await pool.close()


@pytest.mark.asyncio
async def test_probe_does_not_change_inference_inflight() -> None:
    backend = ControlledBackend("A")
    backend.block_probe = True
    pool = make_pool(backend)

    probe = asyncio.create_task(pool.probe_once())
    await asyncio.wait_for(backend.probe_started.wait(), timeout=1)
    snapshot = await pool.snapshot()

    assert snapshot.backends["A"].inflight == 0
    backend.release_probe.set()
    await probe
    assert (await pool.snapshot()).backends["A"].inflight == 0
    await pool.close()


@pytest.mark.asyncio
async def test_periodic_probe_recovers_and_probe_task_stops_cleanly() -> None:
    backend = ControlledBackend("A", healthy=False)
    backend.probe_target = 2
    pool = make_pool(backend, health_interval_seconds=0.01)

    await pool.start()
    assert (await pool.snapshot()).backends["A"].healthy is False
    backend.healthy = True
    await asyncio.wait_for(backend.probe_target_reached.wait(), timeout=1)
    for _ in range(100):
        if (await pool.snapshot()).backends["A"].healthy:
            break
        await asyncio.sleep(0)
    assert (await pool.snapshot()).backends["A"].healthy is True
    probe_task = pool._probe_task

    await pool.close()

    assert probe_task is not None and probe_task.done()
    assert backend.closed is True
    assert (await pool.snapshot()).closed is True


@pytest.mark.asyncio
async def test_pool_shutdown_closes_active_stream_and_releases_assignment() -> None:
    backend_a = ControlledBackend("A")
    backend_b = ControlledBackend("B")
    pool = make_pool(backend_a, backend_b)
    await pool.probe_once()
    await pool.stream("active-at-shutdown")

    assert (await pool.snapshot()).backends["A"].inflight == 1
    await pool.close()
    snapshot = await pool.snapshot()

    assert snapshot.closed is True
    assert all(slot.inflight == 0 for slot in snapshot.backends.values())
    assert backend_a.streams[0].closed.is_set()
    assert backend_a.closed is backend_b.closed is True
    with pytest.raises(NoHealthyBackendError):
        await pool.generate("after-close")


@pytest.mark.asyncio
async def test_pool_shutdown_clears_active_nonstreaming_assignment() -> None:
    backend = ControlledBackend("A", block_generate=True)
    pool = make_pool(backend)
    await pool.probe_once()
    active = asyncio.create_task(pool.generate("active-json-at-shutdown"))
    await asyncio.wait_for(backend.generate_started.get(), timeout=1)

    assert (await pool.snapshot()).backends["A"].inflight == 1
    await pool.close()
    result = await active

    assert result["backend_id"] == "A"
    assert (await pool.snapshot()).backends["A"].inflight == 0


@pytest.mark.asyncio
async def test_concurrent_mixed_routing_stress_leaves_balanced_clean_state() -> None:
    backends = [
        ControlledBackend(backend_id, auto_finish_stream=True) for backend_id in ("A", "B", "C")
    ]
    pool = make_pool(*backends)
    await pool.probe_once()

    async def operation(index: int) -> None:
        if index % 3:
            await pool.generate(f"json-{index}")
            return
        stream = await pool.stream(f"stream-{index}")
        if index % 9 == 0:
            await stream.aclose()
        else:
            _ = [chunk async for chunk in stream]

    await asyncio.gather(*(operation(index) for index in range(150)))
    snapshot = await pool.snapshot()
    assignments = [len(backend.generate_calls) + len(backend.stream_calls) for backend in backends]

    assert sum(assignments) == 150
    assert all(assignments)
    assert all(slot.healthy and slot.inflight == 0 for slot in snapshot.backends.values())
    await pool.close()
