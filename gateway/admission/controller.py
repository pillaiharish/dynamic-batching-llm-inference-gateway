"""Concurrency-safe bounded admission with round-robin tenant fairness."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from gateway.admission.models import AdmissionSnapshot, TenantAdmissionSnapshot
from gateway.auth.tenants import TenantContext
from gateway.core.errors import (
    AdmissionTimeoutError,
    AdmissionUnavailableError,
    GatewayQueueFullError,
    TenantQueueFullError,
)


@dataclass(slots=True)
class _Waiter:
    future: asyncio.Future[AdmissionLease]


@dataclass(slots=True)
class _TenantState:
    tenant: TenantContext
    inflight: int = 0
    queue: deque[_Waiter] = field(default_factory=deque)


class AdmissionLease:
    """An idempotently releasable tenant/global inflight slot."""

    def __init__(
        self,
        controller: AdmissionController,
        tenant_id: str,
        *,
        was_queued: bool,
    ) -> None:
        self._controller = controller
        self.tenant_id = tenant_id
        self.was_queued = was_queued
        self._released = False

    async def release(self) -> None:
        await self._controller._release(self)

    async def __aenter__(self) -> AdmissionLease:
        return self

    async def __aexit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        await self.release()


class AdmissionController:
    """Enforce process-local tenant/global inflight and bounded queue limits."""

    def __init__(
        self,
        tenants: Iterable[TenantContext],
        *,
        global_max_inflight: int,
        global_max_queue: int,
        queue_timeout_seconds: float,
    ) -> None:
        configured_tenants = tuple(tenants)
        if global_max_inflight <= 0 or global_max_queue < 0 or queue_timeout_seconds <= 0:
            raise ValueError("global admission limits are invalid")
        self._tenants = {
            tenant.tenant_id: _TenantState(tenant=tenant) for tenant in configured_tenants
        }
        if len(self._tenants) != len(configured_tenants):
            raise ValueError("tenant IDs must be unique")
        self._global_max_inflight = global_max_inflight
        self._global_max_queue = global_max_queue
        self._queue_timeout_seconds = queue_timeout_seconds
        self._global_inflight = 0
        self._global_queued = 0
        self._active_tenants: deque[str] = deque()
        self._active_tenant_ids: set[str] = set()
        self._lock = asyncio.Lock()
        self._closed = False

    async def acquire(self, tenant: TenantContext) -> AdmissionLease:
        """Acquire immediately or join a bounded queue until admitted or timed out."""
        async with self._lock:
            if self._closed:
                raise AdmissionUnavailableError()
            state = self._state_for(tenant)
            self._dispatch_locked()
            if not state.queue and self._can_admit(state):
                return self._grant_locked(state, was_queued=False)
            if len(state.queue) >= state.tenant.max_queue:
                raise TenantQueueFullError()
            if self._global_queued >= self._global_max_queue:
                raise GatewayQueueFullError()

            future = asyncio.get_running_loop().create_future()
            waiter = _Waiter(future=future)
            state.queue.append(waiter)
            self._global_queued += 1
            self._activate_tenant_locked(state.tenant.tenant_id)
            self._dispatch_locked()

        try:
            return await asyncio.wait_for(future, timeout=self._queue_timeout_seconds)
        except TimeoutError as exc:
            async with self._lock:
                self._remove_waiter_locked(state, waiter)
                self._dispatch_locked()
            raise AdmissionTimeoutError() from exc
        except asyncio.CancelledError:
            async with self._lock:
                removed = self._remove_waiter_locked(state, waiter)
                if not removed and future.done() and not future.cancelled():
                    try:
                        result = future.result()
                    except AdmissionUnavailableError:
                        pass
                    else:
                        self._release_locked(result)
                self._dispatch_locked()
            raise

    @asynccontextmanager
    async def admit(self, tenant: TenantContext) -> AsyncIterator[AdmissionLease]:
        """Hold an admission slot for the duration of one backend operation."""
        lease: AdmissionLease | None = None
        try:
            lease = await self.acquire(tenant)
            yield lease
        finally:
            if lease is not None:
                await lease.release()

    async def snapshot(self) -> AdmissionSnapshot:
        """Return a credential-free consistent state snapshot."""
        async with self._lock:
            return AdmissionSnapshot(
                global_inflight=self._global_inflight,
                global_queued=self._global_queued,
                tenants={
                    tenant_id: TenantAdmissionSnapshot(
                        inflight=state.inflight,
                        queued=len(state.queue),
                    )
                    for tenant_id, state in self._tenants.items()
                },
                closed=self._closed,
            )

    async def shutdown(self) -> None:
        """Stop admission and safely fail every unresolved queued waiter."""
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            for state in self._tenants.values():
                while state.queue:
                    waiter = state.queue.popleft()
                    self._global_queued -= 1
                    if not waiter.future.done():
                        waiter.future.set_exception(AdmissionUnavailableError())
            self._active_tenants.clear()
            self._active_tenant_ids.clear()
            if self._global_queued != 0:
                raise RuntimeError("admission queue accounting became inconsistent")

    async def _release(self, lease: AdmissionLease) -> None:
        async with self._lock:
            self._release_locked(lease)
            self._dispatch_locked()

    def _release_locked(self, lease: AdmissionLease) -> None:
        if lease._released:
            return
        state = self._tenants[lease.tenant_id]
        if state.inflight <= 0 or self._global_inflight <= 0:
            raise RuntimeError("admission inflight accounting became inconsistent")
        lease._released = True
        state.inflight -= 1
        self._global_inflight -= 1

    def _dispatch_locked(self) -> None:
        if self._closed:
            return
        while self._global_inflight < self._global_max_inflight and self._active_tenants:
            candidate_count = len(self._active_tenants)
            granted = False
            for _ in range(candidate_count):
                tenant_id = self._active_tenants.popleft()
                self._active_tenant_ids.remove(tenant_id)
                state = self._tenants[tenant_id]
                self._discard_completed_waiters_locked(state)
                if not state.queue:
                    continue
                if state.inflight >= state.tenant.max_inflight:
                    self._activate_tenant_locked(tenant_id)
                    continue

                waiter = state.queue.popleft()
                self._global_queued -= 1
                if state.queue:
                    self._activate_tenant_locked(tenant_id)
                lease = self._grant_locked(state, was_queued=True)
                waiter.future.set_result(lease)
                granted = True
                break
            if not granted:
                break

    def _discard_completed_waiters_locked(self, state: _TenantState) -> None:
        while state.queue and state.queue[0].future.done():
            state.queue.popleft()
            self._global_queued -= 1

    def _remove_waiter_locked(self, state: _TenantState, waiter: _Waiter) -> bool:
        try:
            state.queue.remove(waiter)
        except ValueError:
            return False
        self._global_queued -= 1
        if not state.queue:
            self._deactivate_tenant_locked(state.tenant.tenant_id)
        return True

    def _grant_locked(self, state: _TenantState, *, was_queued: bool) -> AdmissionLease:
        if not self._can_admit(state):
            raise RuntimeError("attempted to grant admission without capacity")
        state.inflight += 1
        self._global_inflight += 1
        return AdmissionLease(
            self,
            state.tenant.tenant_id,
            was_queued=was_queued,
        )

    def _can_admit(self, state: _TenantState) -> bool:
        return (
            state.inflight < state.tenant.max_inflight
            and self._global_inflight < self._global_max_inflight
        )

    def _activate_tenant_locked(self, tenant_id: str) -> None:
        if tenant_id not in self._active_tenant_ids:
            self._active_tenants.append(tenant_id)
            self._active_tenant_ids.add(tenant_id)

    def _deactivate_tenant_locked(self, tenant_id: str) -> None:
        if tenant_id not in self._active_tenant_ids:
            return
        self._active_tenant_ids.remove(tenant_id)
        self._active_tenants.remove(tenant_id)

    def _state_for(self, tenant: TenantContext) -> _TenantState:
        try:
            return self._tenants[tenant.tenant_id]
        except KeyError as exc:
            raise AdmissionUnavailableError() from exc
