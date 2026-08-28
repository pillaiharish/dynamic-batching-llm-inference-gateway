"""Health-filtered least-inflight backend routing."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from gateway.backends.base import BackendBatchResult, BackendStream, HealthCheckBackend
from gateway.core.errors import (
    BackendCapabilityError,
    BackendConfigurationError,
    BackendHTTPError,
    BackendProtocolError,
    BackendTimeoutError,
    BackendUnavailableError,
    NoHealthyBackendError,
)
from gateway.routing.models import BackendPoolSnapshot, BackendSlotSnapshot

logger = logging.getLogger(__name__)

_HEALTH_IMPACTING_ERRORS = (
    BackendUnavailableError,
    BackendTimeoutError,
    BackendProtocolError,
    BackendConfigurationError,
    BackendHTTPError,
)

BackendOperation = Literal["generate", "stream", "batch"]
BackendOutcome = Literal["success", "error", "cancelled"]


class BackendPoolObserver(Protocol):
    """Synchronous, non-authoritative observation of routing transitions."""

    def backend_state_changed(
        self,
        backends: Mapping[str, tuple[bool, int]],
    ) -> None: ...

    def backend_operation_observed(
        self,
        backend_id: str,
        operation: BackendOperation,
        outcome: BackendOutcome,
    ) -> None: ...


@dataclass(slots=True)
class _BackendSlot:
    backend_id: str
    backend: HealthCheckBackend
    healthy: bool | None = None
    inflight: int = 0


class BackendLease:
    """An idempotently releasable assignment to one backend slot."""

    def __init__(self, pool: BackendPool, slot: _BackendSlot, *, weight: int) -> None:
        self._pool = pool
        self._slot = slot
        self.backend_id = slot.backend_id
        self.backend = slot.backend
        self.weight = weight
        self._released = False

    async def release(self) -> None:
        await self._pool._release(self)


class RoutedBackendStream:
    """Own a leaf stream and its backend assignment for the full SSE lifetime."""

    def __init__(
        self,
        pool: BackendPool,
        stream: BackendStream,
        lease: BackendLease,
    ) -> None:
        self._pool = pool
        self._stream = stream
        self._lease = lease
        self._close_lock = asyncio.Lock()
        self._closed = False

    @property
    def backend_id(self) -> str:
        """Expose the bounded selected backend ID to passive stream observers."""
        return self._lease.backend_id

    @property
    def upstream_request_started_at(self) -> float | None:
        """Delegate the leaf's timestamp captured immediately before HTTP begins."""
        return getattr(self._stream, "upstream_request_started_at", None)

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[bytes]:
        try:
            async for chunk in self._stream:
                yield chunk
        except asyncio.CancelledError:
            await self._finish("cancelled")
            raise
        except _HEALTH_IMPACTING_ERRORS:
            await self._pool._mark_unhealthy(self._lease.backend_id)
            await self._finish("error")
            raise
        except Exception:
            await self._finish("error")
            raise
        else:
            await self._finish("success")
        finally:
            if not self._closed:
                await self._finish("cancelled")

    async def aclose(self) -> None:
        await self._finish("cancelled")

    async def _finish(self, outcome: BackendOutcome) -> None:
        async with self._close_lock:
            if self._closed:
                return
            self._closed = True
            final_outcome = outcome
            try:
                await self._stream.aclose()
            except asyncio.CancelledError:
                final_outcome = "cancelled"
                raise
            except Exception:
                final_outcome = "error"
                raise
            finally:
                try:
                    await self._lease.release()
                finally:
                    try:
                        await self._pool._unregister_stream(self)
                    finally:
                        self._pool._observe_operation(
                            self._lease.backend_id,
                            "stream",
                            final_outcome,
                        )


class BackendPool:
    """Route operations across healthy backends using least-inflight selection."""

    def __init__(
        self,
        backends: Mapping[str, HealthCheckBackend],
        *,
        health_interval_seconds: float,
        health_timeout_seconds: float,
        observer: BackendPoolObserver | None = None,
    ) -> None:
        if not backends:
            raise ValueError("at least one backend is required")
        if health_interval_seconds <= 0 or health_timeout_seconds <= 0:
            raise ValueError("backend health timing must be positive")
        self._slots = [
            _BackendSlot(backend_id=backend_id, backend=backend)
            for backend_id, backend in backends.items()
        ]
        if any(not slot.backend_id.strip() for slot in self._slots):
            raise ValueError("backend IDs must not be blank")
        self._slots_by_id = {slot.backend_id: slot for slot in self._slots}
        if len(self._slots_by_id) != len(self._slots):
            raise ValueError("backend IDs must be unique")

        self._health_interval_seconds = health_interval_seconds
        self._health_timeout_seconds = health_timeout_seconds
        self._cursor = 0
        self._lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._stop_event = asyncio.Event()
        self._probe_task: asyncio.Task[None] | None = None
        self._active_leases: set[BackendLease] = set()
        self._active_streams: set[RoutedBackendStream] = set()
        self._started = False
        self._closed = False
        self._observer = observer
        self._notify_observer()

    def set_observer(self, observer: BackendPoolObserver | None) -> None:
        """Attach an optional observer and immediately publish current state."""
        self._observer = observer
        self._notify_observer()

    async def start(self) -> None:
        """Discover initial health and start the single periodic probe loop."""
        async with self._lock:
            if self._closed:
                raise NoHealthyBackendError()
            if self._started:
                return
            self._started = True

        await self.probe_once()

        async with self._lock:
            if not self._closed and self._probe_task is None:
                self._probe_task = asyncio.create_task(
                    self._probe_loop(),
                    name="backend-health-probe-loop",
                )

    async def probe_once(self) -> None:
        """Probe every backend concurrently without changing inference inflight."""
        async with self._lock:
            if self._closed:
                return
            slots = tuple(self._slots)

        async def probe(slot: _BackendSlot) -> None:
            healthy = False
            try:
                async with asyncio.timeout(self._health_timeout_seconds):
                    healthy = await slot.backend.check_health()
            except Exception:
                healthy = False
            await self._set_health(slot.backend_id, healthy)

        await asyncio.gather(*(probe(slot) for slot in slots))

    async def generate(self, request: Any) -> Any:
        """Route one non-streaming operation and release assignment on every exit."""
        lease = await self._select()
        try:
            result = await lease.backend.generate(request)
        except asyncio.CancelledError:
            self._observe_operation(lease.backend_id, "generate", "cancelled")
            raise
        except _HEALTH_IMPACTING_ERRORS:
            await self._mark_unhealthy(lease.backend_id)
            self._observe_operation(lease.backend_id, "generate", "error")
            raise
        except Exception:
            self._observe_operation(lease.backend_id, "generate", "error")
            raise
        else:
            self._observe_operation(lease.backend_id, "generate", "success")
            return result
        finally:
            await lease.release()

    async def stream(self, request: Any) -> BackendStream:
        """Open a routed stream whose wrapper retains backend inflight ownership."""
        lease = await self._select()
        try:
            stream = await lease.backend.stream(request)
        except asyncio.CancelledError:
            self._observe_operation(lease.backend_id, "stream", "cancelled")
            await lease.release()
            raise
        except _HEALTH_IMPACTING_ERRORS:
            await self._mark_unhealthy(lease.backend_id)
            self._observe_operation(lease.backend_id, "stream", "error")
            await lease.release()
            raise
        except BaseException:
            self._observe_operation(lease.backend_id, "stream", "error")
            await lease.release()
            raise

        routed_stream = RoutedBackendStream(self, stream, lease)
        if not await self._register_stream(routed_stream):
            await routed_stream._finish("error")
            raise NoHealthyBackendError()
        return routed_stream

    async def generate_batch(self, requests: list[Any]) -> BackendBatchResult:
        """Route one weighted batch operation to one healthy backend."""
        if not requests:
            raise BackendCapabilityError("Batch generation requires at least one request")
        lease = await self._select(weight=len(requests))
        try:
            result = await lease.backend.generate_batch(requests)
        except asyncio.CancelledError:
            self._observe_operation(lease.backend_id, "batch", "cancelled")
            raise
        except _HEALTH_IMPACTING_ERRORS:
            await self._mark_unhealthy(lease.backend_id)
            self._observe_operation(lease.backend_id, "batch", "error")
            raise
        except Exception:
            self._observe_operation(lease.backend_id, "batch", "error")
            raise
        else:
            self._observe_operation(lease.backend_id, "batch", "success")
            return result
        finally:
            await lease.release()

    async def is_routable(self) -> bool:
        """Return whether a new request can select at least one healthy backend."""
        async with self._lock:
            return not self._closed and any(slot.healthy is True for slot in self._slots)

    async def snapshot(self) -> BackendPoolSnapshot:
        """Return health and gateway-observed inflight without URLs or credentials."""
        async with self._lock:
            return BackendPoolSnapshot(
                closed=self._closed,
                backends={
                    slot.backend_id: BackendSlotSnapshot(
                        healthy=slot.healthy is True,
                        inflight=slot.inflight,
                    )
                    for slot in self._slots
                },
            )

    async def close(self) -> None:
        """Stop probes, close routed streams, and close every leaf client once."""
        async with self._close_lock:
            async with self._lock:
                if self._closed:
                    return
                self._closed = True
                self._stop_event.set()
                probe_task = self._probe_task
                leases = tuple(self._active_leases)
                streams = tuple(self._active_streams)

            if probe_task is not None:
                probe_task.cancel()
                try:
                    await probe_task
                except asyncio.CancelledError:
                    pass

            await asyncio.gather(
                *(stream.aclose() for stream in streams),
                return_exceptions=True,
            )
            await asyncio.gather(
                *(slot.backend.close() for slot in self._slots),
                return_exceptions=True,
            )
            await asyncio.gather(*(lease.release() for lease in leases))
            async with self._lock:
                for slot in self._slots:
                    slot.healthy = False
                self._notify_observer()

    async def _probe_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=self._health_interval_seconds,
                    )
                except TimeoutError:
                    await self.probe_once()
        except asyncio.CancelledError:
            raise

    async def _select(self, *, weight: int = 1) -> BackendLease:
        if weight <= 0:
            raise ValueError("backend assignment weight must be positive")
        async with self._lock:
            if self._closed:
                raise NoHealthyBackendError()

            healthy_indices = [
                index for index, slot in enumerate(self._slots) if slot.healthy is True
            ]
            if not healthy_indices:
                raise NoHealthyBackendError()

            minimum_inflight = min(self._slots[index].inflight for index in healthy_indices)
            candidates = {
                index
                for index in healthy_indices
                if self._slots[index].inflight == minimum_inflight
            }
            selected_index = next(
                index
                for offset in range(len(self._slots))
                if (index := (self._cursor + offset) % len(self._slots)) in candidates
            )
            slot = self._slots[selected_index]
            slot.inflight += weight
            self._cursor = (selected_index + 1) % len(self._slots)
            lease = BackendLease(self, slot, weight=weight)
            self._active_leases.add(lease)
            self._notify_observer()

        logger.info(
            "backend selected",
            extra={
                "backend_id": slot.backend_id,
                "backend_healthy": True,
                "routing_result": "selected",
            },
        )
        return lease

    async def _release(self, lease: BackendLease) -> None:
        async with self._lock:
            if lease._released:
                return
            if lease._slot.inflight < lease.weight:
                raise RuntimeError("backend inflight accounting became inconsistent")
            lease._released = True
            lease._slot.inflight -= lease.weight
            self._active_leases.discard(lease)
            self._notify_observer()

    async def _register_stream(self, stream: RoutedBackendStream) -> bool:
        async with self._lock:
            if self._closed:
                return False
            self._active_streams.add(stream)
            return True

    async def _unregister_stream(self, stream: RoutedBackendStream) -> None:
        async with self._lock:
            self._active_streams.discard(stream)

    async def _mark_unhealthy(self, backend_id: str) -> None:
        await self._set_health(backend_id, False)

    async def _set_health(self, backend_id: str, healthy: bool) -> None:
        async with self._lock:
            if self._closed:
                return
            slot = self._slots_by_id[backend_id]
            previous = slot.healthy
            slot.healthy = healthy
            self._notify_observer()

        if previous is True and not healthy:
            logger.warning(
                "backend became unhealthy",
                extra={"backend_id": backend_id, "backend_healthy": False},
            )
        elif previous is False and healthy:
            logger.info(
                "backend recovered",
                extra={"backend_id": backend_id, "backend_healthy": True},
            )
        elif previous is None and not healthy:
            logger.warning(
                "backend failed initial health probe",
                extra={"backend_id": backend_id, "backend_healthy": False},
            )

    def _observe_operation(
        self,
        backend_id: str,
        operation: BackendOperation,
        outcome: BackendOutcome,
    ) -> None:
        if self._observer is None:
            return
        try:
            self._observer.backend_operation_observed(backend_id, operation, outcome)
        except Exception:
            logger.warning("backend_pool_observer_error", exc_info=True)

    def _notify_observer(self) -> None:
        if self._observer is None:
            return
        try:
            self._observer.backend_state_changed(
                {
                    slot.backend_id: (slot.healthy is True and not self._closed, slot.inflight)
                    for slot in self._slots
                }
            )
        except Exception:
            logger.warning("backend_pool_observer_error", exc_info=True)
