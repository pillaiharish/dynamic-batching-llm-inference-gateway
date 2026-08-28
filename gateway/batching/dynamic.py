"""Concurrency-safe size-or-time dynamic request aggregation."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from time import perf_counter
from typing import Literal, Protocol

from gateway.backends.base import BackendBatchResult, InferenceBackend
from gateway.batching.models import BatchItemResult
from gateway.core.errors import BackendProtocolError, BatchingUnavailableError
from gateway.schemas.chat import ChatCompletionRequest

logger = logging.getLogger(__name__)

FlushReason = Literal["size", "timeout"]
BatchOutcome = Literal["success", "error", "cancelled"]
ItemState = Literal["pending", "dispatched", "completed", "cancelled"]


class DynamicBatchObserver(Protocol):
    """Synchronous, fail-isolated observations of batch transitions."""

    def batch_state_changed(self, pending: int, inflight: int) -> None: ...

    def batch_dispatched(
        self,
        flush_reason: FlushReason,
        size: int,
        wait_seconds: tuple[float, ...],
    ) -> None: ...

    def batch_completed(self, flush_reason: FlushReason, outcome: BatchOutcome) -> None: ...

    def observe_batch_usage(
        self,
        usage_result: Literal["observed", "missing", "invalid"],
        completion_tokens: int | None,
        member_count: int,
    ) -> None: ...


@dataclass(slots=True)
class _BatchItem:
    request: ChatCompletionRequest
    future: asyncio.Future[BatchItemResult]
    submitted_at: float
    state: ItemState = "pending"


@dataclass(slots=True)
class _PendingBatch:
    key: str
    items: list[_BatchItem] = field(default_factory=list)
    timer_task: asyncio.Task[None] | None = None


@dataclass(slots=True)
class _FlushOperation:
    items: tuple[_BatchItem, ...]
    flush_reason: FlushReason
    started: bool = False


def compatibility_key(tenant_id: str, request: ChatCompletionRequest) -> str:
    """Canonicalize tenant-local exact shared upstream field semantics."""
    canonical_shared = json.dumps(
        request.to_batch_shared_payload(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return json.dumps(
        [tenant_id, canonical_shared],
        separators=(",", ":"),
        ensure_ascii=False,
    )


class DynamicBatcher:
    """Aggregate compatible admitted requests into bounded upstream batches."""

    def __init__(
        self,
        backend: InferenceBackend,
        *,
        max_size: int,
        max_wait_seconds: float,
        observer: DynamicBatchObserver | None = None,
        clock: Callable[[], float] = perf_counter,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if not 2 <= max_size <= 64:
            raise ValueError("dynamic batch max size must be between 2 and 64")
        if not 0 < max_wait_seconds <= 1:
            raise ValueError("dynamic batch max wait must be in (0, 1]")
        self._backend = backend
        self._max_size = max_size
        self._max_wait_seconds = max_wait_seconds
        self._observer = observer
        self._clock = clock
        self._sleep = sleep
        self._groups: dict[str, _PendingBatch] = {}
        self._timer_tasks: set[asyncio.Task[None]] = set()
        self._flush_tasks: set[asyncio.Task[None]] = set()
        self._flush_operations: dict[asyncio.Task[None], _FlushOperation] = {}
        self._pending = 0
        self._inflight = 0
        self._lock = asyncio.Lock()
        self._shutdown_lock = asyncio.Lock()
        self._closed = False
        self._notify_state()

    async def submit(
        self,
        tenant_id: str,
        request: ChatCompletionRequest,
    ) -> BatchItemResult:
        """Queue one already-admitted eligible request and await its own result."""
        loop = asyncio.get_running_loop()
        item = _BatchItem(
            request=request,
            future=loop.create_future(),
            submitted_at=self._clock(),
        )
        key = compatibility_key(tenant_id, request)

        async with self._lock:
            if self._closed:
                raise BatchingUnavailableError()
            batch = self._groups.get(key)
            if batch is None:
                batch = _PendingBatch(key=key)
                self._groups[key] = batch
                timer = asyncio.create_task(
                    self._timer_expired(batch),
                    name="dynamic-batch-timer",
                )
                batch.timer_task = timer
                self._track(timer, self._timer_tasks)
            batch.items.append(item)
            self._pending += 1
            self._notify_state()
            if len(batch.items) >= self._max_size:
                self._detach_locked(batch, "size")

        try:
            return await asyncio.shield(item.future)
        except asyncio.CancelledError:
            await self._cancel_item(item, key)
            raise

    async def shutdown(self) -> None:
        """Stop submissions, resolve pending members, and cancel all owned work."""
        async with self._shutdown_lock:
            async with self._lock:
                if self._closed:
                    return
                self._closed = True
                pending_batches = tuple(self._groups.values())
                self._groups.clear()
                for batch in pending_batches:
                    if batch.timer_task is not None:
                        batch.timer_task.cancel()
                    for item in batch.items:
                        if item.state == "pending":
                            item.state = "completed"
                            self._pending -= 1
                            if not item.future.done():
                                item.future.set_exception(BatchingUnavailableError())
                flush_tasks = tuple(self._flush_tasks)
                never_started_reasons: list[FlushReason] = []
                for task in flush_tasks:
                    task.cancel()
                    operation = self._flush_operations.get(task)
                    if operation is None or operation.started:
                        continue
                    for item in operation.items:
                        if item.state == "dispatched":
                            item.state = "completed"
                            if not item.future.done():
                                item.future.set_exception(BatchingUnavailableError())
                    if self._inflight <= 0:
                        raise RuntimeError("dynamic batch inflight accounting became inconsistent")
                    self._inflight -= 1
                    never_started_reasons.append(operation.flush_reason)
                timer_tasks = tuple(self._timer_tasks)
                self._notify_state()

            for flush_reason in never_started_reasons:
                self._observe_completed(flush_reason, "cancelled")
            if timer_tasks:
                await asyncio.gather(*timer_tasks, return_exceptions=True)
            if flush_tasks:
                await asyncio.gather(*flush_tasks, return_exceptions=True)

    async def snapshot(self) -> tuple[int, int]:
        """Return process-local pending logical members and inflight HTTP batches."""
        async with self._lock:
            return self._pending, self._inflight

    async def _timer_expired(self, batch: _PendingBatch) -> None:
        try:
            await self._sleep(self._max_wait_seconds)
            async with self._lock:
                self._detach_locked(batch, "timeout")
        except asyncio.CancelledError:
            raise

    def _detach_locked(self, batch: _PendingBatch, flush_reason: FlushReason) -> None:
        if self._groups.get(batch.key) is not batch or not batch.items:
            return
        del self._groups[batch.key]
        current_task = asyncio.current_task()
        if batch.timer_task is not None and batch.timer_task is not current_task:
            batch.timer_task.cancel()

        dispatched_at = self._clock()
        for item in batch.items:
            if item.state != "pending":
                raise RuntimeError("dynamic batch item state became inconsistent")
            item.state = "dispatched"
        size = len(batch.items)
        self._pending -= size
        self._inflight += 1
        self._notify_state()
        self._observe_dispatched(
            flush_reason,
            size,
            tuple(max(0.0, dispatched_at - item.submitted_at) for item in batch.items),
        )
        operation = _FlushOperation(tuple(batch.items), flush_reason)
        task = asyncio.create_task(
            self._execute_batch(operation),
            name="dynamic-batch-flush",
        )
        self._track(task, self._flush_tasks)
        self._flush_operations[task] = operation
        task.add_done_callback(self._flush_operations.pop)

    async def _execute_batch(
        self,
        operation: _FlushOperation,
    ) -> None:
        operation.started = True
        items = operation.items
        outcome: BatchOutcome = "error"
        try:
            result = await self._backend.generate_batch([item.request for item in items])
            if not isinstance(result, BackendBatchResult) or len(result.responses) != len(items):
                raise BackendProtocolError()
            delivered = await self._complete_success(items, result)
            self._observe_usage(result, delivered)
            outcome = "success"
        except asyncio.CancelledError:
            outcome = "cancelled"
            await self._fail_items(items, BatchingUnavailableError())
            raise
        except Exception as exc:
            await self._fail_items(items, exc)
        finally:
            async with self._lock:
                if self._inflight <= 0:
                    raise RuntimeError("dynamic batch inflight accounting became inconsistent")
                self._inflight -= 1
                self._notify_state()
            self._observe_completed(operation.flush_reason, outcome)

    async def _complete_success(
        self,
        items: tuple[_BatchItem, ...],
        result: BackendBatchResult,
    ) -> int:
        delivered = 0
        async with self._lock:
            for item, response in zip(items, result.responses, strict=True):
                if item.state == "cancelled":
                    continue
                if item.state != "dispatched":
                    raise RuntimeError("dynamic batch item state became inconsistent")
                item.state = "completed"
                if not item.future.done():
                    item.future.set_result(BatchItemResult(response=response))
                    delivered += 1
        return delivered

    async def _fail_items(
        self,
        items: tuple[_BatchItem, ...],
        exc: Exception,
    ) -> None:
        async with self._lock:
            for item in items:
                if item.state == "cancelled":
                    continue
                if item.state != "dispatched":
                    raise RuntimeError("dynamic batch item state became inconsistent")
                item.state = "completed"
                if not item.future.done():
                    item.future.set_exception(exc)

    async def _cancel_item(self, item: _BatchItem, key: str) -> None:
        async with self._lock:
            if item.state == "pending":
                batch = self._groups.get(key)
                if batch is not None and item in batch.items:
                    batch.items.remove(item)
                    self._pending -= 1
                    if not batch.items:
                        del self._groups[key]
                        if batch.timer_task is not None:
                            batch.timer_task.cancel()
                    self._notify_state()
                item.state = "cancelled"
            elif item.state == "dispatched":
                item.state = "cancelled"
            if not item.future.done():
                item.future.cancel()
            elif not item.future.cancelled():
                item.future.exception()

    @staticmethod
    def _track(task: asyncio.Task[None], tasks: set[asyncio.Task[None]]) -> None:
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    def _notify_state(self) -> None:
        if self._observer is None:
            return
        try:
            self._observer.batch_state_changed(self._pending, self._inflight)
        except Exception:
            logger.warning("dynamic_batch_observer_error", exc_info=True)

    def _observe_dispatched(
        self,
        flush_reason: FlushReason,
        size: int,
        wait_seconds: tuple[float, ...],
    ) -> None:
        if self._observer is None:
            return
        try:
            self._observer.batch_dispatched(flush_reason, size, wait_seconds)
        except Exception:
            logger.warning("dynamic_batch_observer_error", exc_info=True)

    def _observe_completed(self, flush_reason: FlushReason, outcome: BatchOutcome) -> None:
        if self._observer is None:
            return
        try:
            self._observer.batch_completed(flush_reason, outcome)
        except Exception:
            logger.warning("dynamic_batch_observer_error", exc_info=True)

    def _observe_usage(self, result: BackendBatchResult, member_count: int) -> None:
        if self._observer is None:
            return
        try:
            self._observer.observe_batch_usage(
                result.usage_result,
                result.aggregate_completion_tokens,
                member_count,
            )
        except Exception:
            logger.warning("dynamic_batch_observer_error", exc_info=True)
