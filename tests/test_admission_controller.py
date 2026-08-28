import asyncio
from collections.abc import Callable

import pytest

from gateway.admission.controller import AdmissionController
from gateway.admission.models import AdmissionSnapshot
from gateway.auth.tenants import TenantContext
from gateway.core.errors import (
    AdmissionTimeoutError,
    AdmissionUnavailableError,
    BackendHTTPError,
    BackendTimeoutError,
    BackendUnavailableError,
    GatewayQueueFullError,
    TenantQueueFullError,
)

TENANT_A = TenantContext("tenant-a", max_inflight=1, max_queue=8)
TENANT_B = TenantContext("tenant-b", max_inflight=1, max_queue=8)


def make_controller(
    *tenants: TenantContext,
    global_max_inflight: int = 1,
    global_max_queue: int = 16,
    queue_timeout_seconds: float = 1.0,
) -> AdmissionController:
    return AdmissionController(
        tenants,
        global_max_inflight=global_max_inflight,
        global_max_queue=global_max_queue,
        queue_timeout_seconds=queue_timeout_seconds,
    )


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


class ControlledBackend:
    def __init__(self) -> None:
        self.started: asyncio.Queue[str] = asyncio.Queue()
        self._release_events: dict[str, asyncio.Event] = {}
        self.current = 0
        self.max_current = 0

    async def generate(self, request_id: str) -> str:
        release_event = self._release_events.setdefault(request_id, asyncio.Event())
        self.current += 1
        self.max_current = max(self.max_current, self.current)
        self.started.put_nowait(request_id)
        try:
            await release_event.wait()
            return request_id
        finally:
            self.current -= 1

    def release(self, request_id: str) -> None:
        self._release_events[request_id].set()


async def run_request(
    controller: AdmissionController,
    backend: ControlledBackend,
    tenant: TenantContext,
    request_id: str,
) -> str:
    async with controller.admit(tenant):
        return await backend.generate(request_id)


@pytest.mark.asyncio
async def test_per_tenant_inflight_limit_queues_then_runs() -> None:
    controller = make_controller(TENANT_A, global_max_inflight=2)
    backend = ControlledBackend()
    first = asyncio.create_task(run_request(controller, backend, TENANT_A, "A1"))
    assert await backend.started.get() == "A1"

    second = asyncio.create_task(run_request(controller, backend, TENANT_A, "A2"))
    snapshot = await wait_for_snapshot(
        controller,
        lambda state: state.tenants["tenant-a"].queued == 1,
    )
    assert snapshot.tenants["tenant-a"].inflight == 1
    assert backend.max_current == 1

    backend.release("A1")
    assert await backend.started.get() == "A2"
    backend.release("A2")
    assert await first == "A1"
    assert await second == "A2"
    assert backend.max_current == 1


@pytest.mark.asyncio
async def test_tenant_queue_bound_rejects_without_backend_execution() -> None:
    tenant = TenantContext("tenant-a", max_inflight=1, max_queue=1)
    controller = make_controller(tenant)
    first = await controller.acquire(tenant)
    second = asyncio.create_task(controller.acquire(tenant))
    await wait_for_snapshot(controller, lambda state: state.global_queued == 1)

    with pytest.raises(TenantQueueFullError):
        await controller.acquire(tenant)

    await first.release()
    second_lease = await second
    await second_lease.release()


@pytest.mark.asyncio
async def test_global_inflight_limit_blocks_other_tenant() -> None:
    controller = make_controller(TENANT_A, TENANT_B, global_max_inflight=1)
    backend = ControlledBackend()
    first = asyncio.create_task(run_request(controller, backend, TENANT_A, "A1"))
    assert await backend.started.get() == "A1"

    second = asyncio.create_task(run_request(controller, backend, TENANT_B, "B1"))
    await wait_for_snapshot(controller, lambda state: state.global_queued == 1)
    assert backend.max_current == 1

    backend.release("A1")
    assert await backend.started.get() == "B1"
    backend.release("B1")
    await first
    await second
    assert backend.max_current == 1


@pytest.mark.asyncio
async def test_global_queue_bound_rejects_additional_tenant() -> None:
    controller = make_controller(
        TENANT_A,
        TENANT_B,
        global_max_inflight=1,
        global_max_queue=1,
    )
    first = await controller.acquire(TENANT_A)
    queued = asyncio.create_task(controller.acquire(TENANT_A))
    await wait_for_snapshot(controller, lambda state: state.global_queued == 1)

    with pytest.raises(GatewayQueueFullError):
        await controller.acquire(TENANT_B)

    await first.release()
    queued_lease = await queued
    await queued_lease.release()


@pytest.mark.asyncio
async def test_round_robin_prevents_noisy_tenant_starvation() -> None:
    controller = make_controller(TENANT_A, TENANT_B, global_max_inflight=1)
    backend = ControlledBackend()
    tasks: dict[str, asyncio.Task[str]] = {}

    tasks["A1"] = asyncio.create_task(run_request(controller, backend, TENANT_A, "A1"))
    assert await backend.started.get() == "A1"
    for index in (2, 3, 4):
        request_id = f"A{index}"
        tasks[request_id] = asyncio.create_task(
            run_request(controller, backend, TENANT_A, request_id)
        )
        await wait_for_snapshot(
            controller,
            lambda state, expected=index - 1: state.tenants["tenant-a"].queued == expected,
        )
    tasks["B1"] = asyncio.create_task(run_request(controller, backend, TENANT_B, "B1"))
    await wait_for_snapshot(controller, lambda state: state.global_queued == 4)

    backend.release("A1")
    assert await backend.started.get() == "A2"
    backend.release("A2")
    assert await backend.started.get() == "B1"
    assert not tasks["A3"].done()

    backend.release("B1")
    assert await backend.started.get() == "A3"
    backend.release("A3")
    assert await backend.started.get() == "A4"
    backend.release("A4")
    assert await asyncio.gather(*tasks.values()) == ["A1", "A2", "A3", "A4", "B1"]


@pytest.mark.asyncio
async def test_queue_timeout_removes_waiter_without_late_dispatch() -> None:
    controller = make_controller(
        TENANT_A,
        queue_timeout_seconds=0.01,
    )
    first = await controller.acquire(TENANT_A)

    with pytest.raises(AdmissionTimeoutError):
        await controller.acquire(TENANT_A)

    snapshot = await controller.snapshot()
    assert snapshot.global_queued == 0
    assert snapshot.tenants["tenant-a"].queued == 0
    await first.release()
    replacement = await controller.acquire(TENANT_A)
    assert replacement.was_queued is False
    await replacement.release()


@pytest.mark.asyncio
async def test_cancelled_queued_request_is_removed_and_never_dispatched() -> None:
    controller = make_controller(TENANT_A)
    first = await controller.acquire(TENANT_A)
    queued = asyncio.create_task(controller.acquire(TENANT_A))
    await wait_for_snapshot(controller, lambda state: state.global_queued == 1)

    queued.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued

    snapshot = await controller.snapshot()
    assert snapshot.global_queued == 0
    assert snapshot.tenants["tenant-a"].queued == 0
    await first.release()
    replacement = await controller.acquire(TENANT_A)
    await replacement.release()


@pytest.mark.asyncio
async def test_cancellation_after_admission_releases_slot() -> None:
    controller = make_controller(TENANT_A)
    backend = ControlledBackend()
    task = asyncio.create_task(run_request(controller, backend, TENANT_A, "A1"))
    assert await backend.started.get() == "A1"

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    snapshot = await controller.snapshot()
    assert snapshot.global_inflight == 0
    assert snapshot.tenants["tenant-a"].inflight == 0
    replacement = await controller.acquire(TENANT_A)
    await replacement.release()


@pytest.mark.parametrize(
    "backend_error",
    [BackendTimeoutError(), BackendUnavailableError(), BackendHTTPError()],
)
@pytest.mark.asyncio
async def test_backend_failure_releases_slot_for_next_request(
    backend_error: Exception,
) -> None:
    controller = make_controller(TENANT_A)

    async def fail_after_admission() -> None:
        async with controller.admit(TENANT_A):
            raise backend_error

    with pytest.raises(type(backend_error)):
        await fail_after_admission()

    snapshot = await controller.snapshot()
    assert snapshot.global_inflight == 0
    replacement = await controller.acquire(TENANT_A)
    await replacement.release()


@pytest.mark.asyncio
async def test_shutdown_fails_waiters_and_stops_new_admission() -> None:
    controller = make_controller(TENANT_A)
    first = await controller.acquire(TENANT_A)
    queued = asyncio.create_task(controller.acquire(TENANT_A))
    await wait_for_snapshot(controller, lambda state: state.global_queued == 1)

    await controller.shutdown()

    with pytest.raises(AdmissionUnavailableError):
        await queued
    snapshot = await controller.snapshot()
    assert snapshot.closed is True
    assert snapshot.global_queued == 0
    await first.release()
    with pytest.raises(AdmissionUnavailableError):
        await controller.acquire(TENANT_A)
