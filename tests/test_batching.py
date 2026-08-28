import asyncio
from collections.abc import Awaitable
from typing import Any

import pytest

from gateway.admission.controller import AdmissionController
from gateway.auth.tenants import TenantContext
from gateway.backends.base import BackendBatchResult, BackendStream
from gateway.batching.dynamic import DynamicBatcher, compatibility_key
from gateway.batching.eligibility import batching_eligibility
from gateway.core.errors import (
    BackendUnavailableError,
    BatchingUnavailableError,
)
from gateway.observability.metrics import GatewayMetrics
from gateway.schemas.chat import ChatCompletionRequest


def chat_request(message: str, **overrides: Any) -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate(
        {
            "model": "model-a",
            "messages": [{"role": "user", "content": message}],
            **overrides,
        }
    )


class ControlledSleeper:
    def __init__(self) -> None:
        self.waiters: asyncio.Queue[asyncio.Future[None]] = asyncio.Queue()
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)
        future = asyncio.get_running_loop().create_future()
        self.waiters.put_nowait(future)
        await future

    async def trigger_next(self) -> None:
        while True:
            future = await self.waiters.get()
            if not future.done():
                future.set_result(None)
                return


class StateObserver:
    def __init__(self) -> None:
        self.latest = (0, 0)
        self.transitions: asyncio.Queue[tuple[int, int]] = asyncio.Queue()

    def batch_state_changed(self, pending: int, inflight: int) -> None:
        self.latest = (pending, inflight)
        self.transitions.put_nowait(self.latest)

    def batch_dispatched(self, *_args: Any) -> None:
        return None

    def batch_completed(self, *_args: Any) -> None:
        return None

    def observe_batch_usage(self, *_args: Any) -> None:
        return None

    async def wait_for(self, expected: tuple[int, int]) -> None:
        while self.latest != expected:
            await self.transitions.get()


class ControlledBatchBackend:
    def __init__(
        self,
        *,
        block: bool = False,
        error: Exception | None = None,
        usage_result: str = "observed",
        completion_tokens: int | None = 27,
    ) -> None:
        self.calls: list[list[ChatCompletionRequest]] = []
        self.started: asyncio.Queue[list[ChatCompletionRequest]] = asyncio.Queue()
        self.release = asyncio.Event()
        if not block:
            self.release.set()
        self.error = error
        self.usage_result = usage_result
        self.completion_tokens = completion_tokens
        self.closed = False

    async def generate(self, _request: Any) -> Any:
        raise AssertionError("ordinary generation must not be used")

    async def stream(self, _request: Any) -> BackendStream:
        raise AssertionError("streaming must not be used")

    async def generate_batch(self, requests: list[Any]) -> BackendBatchResult:
        typed_requests = list(requests)
        self.calls.append(typed_requests)
        self.started.put_nowait(typed_requests)
        await self.release.wait()
        if self.error is not None:
            raise self.error
        responses = []
        for request in typed_requests:
            content = request.messages[-1].content
            responses.append(
                {
                    "id": "chatcmpl-batch",
                    "object": "chat.completion",
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": f"result:{content}"},
                            "finish_reason": "stop",
                        }
                    ],
                }
            )
        return BackendBatchResult(
            responses=responses,
            aggregate_completion_tokens=self.completion_tokens,
            usage_result=self.usage_result,  # type: ignore[arg-type]
        )

    async def close(self) -> None:
        self.release.set()
        self.closed = True


@pytest.mark.parametrize(
    "enabled,overrides,decision,reason",
    [
        (False, {}, "bypass", "disabled"),
        (True, {"stream": True}, "bypass", "streaming"),
        (True, {"n": 2}, "bypass", "n_gt_1"),
        (True, {}, "eligible", "eligible"),
    ],
)
def test_batching_eligibility_is_explicit_and_bounded(
    enabled: bool,
    overrides: dict[str, Any],
    decision: str,
    reason: str,
) -> None:
    eligibility = batching_eligibility(chat_request("hello", **overrides), enabled=enabled)

    assert eligibility.decision == decision
    assert eligibility.reason == reason
    assert eligibility.eligible is (decision == "eligible")


def test_compatibility_key_groups_messages_but_preserves_shared_semantics() -> None:
    baseline = chat_request("one", temperature=0.7, top_p=0.8, max_tokens=64, seed=7)
    different_message = chat_request("two", temperature=0.7, top_p=0.8, max_tokens=64, seed=7)

    assert compatibility_key("tenant-a", baseline) == compatibility_key(
        "tenant-a", different_message
    )
    assert compatibility_key("tenant-a", baseline) != compatibility_key("tenant-b", baseline)

    for field, value in (
        ("model", "model-b"),
        ("temperature", 0.8),
        ("top_p", 0.9),
        ("max_tokens", 32),
        ("stop", ["done"]),
        ("seed", 8),
    ):
        candidate = chat_request("other", **{**baseline.model_dump(), field: value})
        assert compatibility_key("tenant-a", baseline) != compatibility_key("tenant-a", candidate)


def test_compatibility_key_preserves_explicit_versus_omitted_defaults() -> None:
    omitted = chat_request("one")
    explicit = chat_request("two", temperature=1.0)

    assert omitted.temperature == explicit.temperature == 1.0
    assert compatibility_key("tenant-a", omitted) != compatibility_key("tenant-a", explicit)


@pytest.mark.asyncio
async def test_size_flush_dispatches_exactly_one_batch() -> None:
    sleeper = ControlledSleeper()
    backend = ControlledBatchBackend()
    state = StateObserver()
    batcher = DynamicBatcher(
        backend, max_size=3, max_wait_seconds=0.5, sleep=sleeper, observer=state
    )

    tasks = [
        asyncio.create_task(batcher.submit("tenant-a", chat_request(str(index))))
        for index in range(2)
    ]
    await asyncio.wait_for(state.wait_for((2, 0)), timeout=1)
    assert backend.calls == []

    tasks.append(asyncio.create_task(batcher.submit("tenant-a", chat_request("2"))))
    dispatched = await asyncio.wait_for(backend.started.get(), timeout=1)
    results = await asyncio.gather(*tasks)

    assert len(backend.calls) == 1
    assert len(dispatched) == 3
    assert [result.response["choices"][0]["message"]["content"] for result in results] == [
        "result:0",
        "result:1",
        "result:2",
    ]
    await asyncio.wait_for(state.wait_for((0, 0)), timeout=1)
    await batcher.shutdown()


@pytest.mark.asyncio
async def test_timeout_flush_uses_controlled_first_member_timer() -> None:
    sleeper = ControlledSleeper()
    backend = ControlledBatchBackend()
    metrics = GatewayMetrics()
    batcher = DynamicBatcher(
        backend,
        max_size=8,
        max_wait_seconds=0.25,
        sleep=sleeper,
        observer=metrics,
    )
    task = asyncio.create_task(batcher.submit("tenant-a", chat_request("one")))
    timer = await asyncio.wait_for(sleeper.waiters.get(), timeout=1)
    assert await batcher.snapshot() == (1, 0)
    timer.set_result(None)
    result = await task

    assert len(backend.calls) == 1
    assert len(backend.calls[0]) == 1
    assert result.response["choices"][0]["message"]["content"] == "result:one"
    assert (
        metrics.registry.get_sample_value(
            "gateway_batches_total", {"flush_reason": "timeout", "outcome": "success"}
        )
        == 1
    )
    await batcher.shutdown()


@pytest.mark.asyncio
async def test_compatible_arrivals_do_not_restart_first_member_timer() -> None:
    sleeper = ControlledSleeper()
    backend = ControlledBatchBackend()
    state = StateObserver()
    batcher = DynamicBatcher(
        backend, max_size=4, max_wait_seconds=0.25, sleep=sleeper, observer=state
    )
    task_a = asyncio.create_task(batcher.submit("tenant-a", chat_request("a")))
    await asyncio.wait_for(state.wait_for((1, 0)), timeout=1)
    timer = await asyncio.wait_for(sleeper.waiters.get(), timeout=1)

    task_b = asyncio.create_task(batcher.submit("tenant-a", chat_request("b")))
    await asyncio.wait_for(state.wait_for((2, 0)), timeout=1)
    assert sleeper.delays == [0.25]

    timer.set_result(None)
    await asyncio.gather(task_a, task_b)
    assert [len(call) for call in backend.calls] == [2]
    await batcher.shutdown()


@pytest.mark.asyncio
async def test_max_size_rollover_forms_two_full_and_one_timeout_batch() -> None:
    sleeper = ControlledSleeper()
    backend = ControlledBatchBackend()
    state = StateObserver()
    batcher = DynamicBatcher(
        backend, max_size=4, max_wait_seconds=0.5, sleep=sleeper, observer=state
    )
    tasks = [
        asyncio.create_task(batcher.submit("tenant-a", chat_request(str(index))))
        for index in range(9)
    ]

    await asyncio.wait_for(backend.started.get(), timeout=1)
    await asyncio.wait_for(backend.started.get(), timeout=1)
    await asyncio.wait_for(state.wait_for((1, 0)), timeout=1)
    await sleeper.trigger_next()
    await asyncio.gather(*tasks)

    assert [len(call) for call in backend.calls] == [4, 4, 1]
    assert max(map(len, backend.calls)) == 4
    await batcher.shutdown()


@pytest.mark.asyncio
async def test_incompatible_and_cross_tenant_requests_never_share_a_batch() -> None:
    sleeper = ControlledSleeper()
    backend = ControlledBatchBackend()
    state = StateObserver()
    batcher = DynamicBatcher(
        backend, max_size=2, max_wait_seconds=0.5, sleep=sleeper, observer=state
    )
    tasks = [
        asyncio.create_task(batcher.submit("tenant-a", chat_request("a", temperature=0.7))),
        asyncio.create_task(batcher.submit("tenant-a", chat_request("b", temperature=0.8))),
        asyncio.create_task(batcher.submit("tenant-b", chat_request("c", temperature=0.7))),
    ]
    await asyncio.wait_for(state.wait_for((3, 0)), timeout=1)

    for _ in range(3):
        await sleeper.trigger_next()
    await asyncio.gather(*tasks)

    assert len(backend.calls) == 3
    assert all(len(call) == 1 for call in backend.calls)
    await batcher.shutdown()


@pytest.mark.asyncio
async def test_cancel_before_dispatch_removes_member_from_upstream_work() -> None:
    sleeper = ControlledSleeper()
    backend = ControlledBatchBackend()
    state = StateObserver()
    batcher = DynamicBatcher(
        backend, max_size=3, max_wait_seconds=0.5, sleep=sleeper, observer=state
    )
    task_a = asyncio.create_task(batcher.submit("tenant-a", chat_request("a")))
    task_b = asyncio.create_task(batcher.submit("tenant-a", chat_request("b")))
    await asyncio.wait_for(state.wait_for((2, 0)), timeout=1)

    task_b.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task_b
    await asyncio.wait_for(state.wait_for((1, 0)), timeout=1)
    await sleeper.trigger_next()
    result = await task_a

    assert [[request.messages[-1].content for request in call] for call in backend.calls] == [["a"]]
    assert result.response["choices"][0]["message"]["content"] == "result:a"
    await batcher.shutdown()


@pytest.mark.asyncio
async def test_cancel_last_pending_member_removes_group_and_timer() -> None:
    sleeper = ControlledSleeper()
    backend = ControlledBatchBackend()
    state = StateObserver()
    batcher = DynamicBatcher(
        backend, max_size=3, max_wait_seconds=0.5, sleep=sleeper, observer=state
    )
    task = asyncio.create_task(batcher.submit("tenant-a", chat_request("only")))
    await asyncio.wait_for(state.wait_for((1, 0)), timeout=1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.wait_for(state.wait_for((0, 0)), timeout=1)
    await asyncio.sleep(0)

    assert backend.calls == []
    assert not batcher._timer_tasks
    await batcher.shutdown()


@pytest.mark.asyncio
async def test_cancel_after_dispatch_does_not_cancel_shared_batch() -> None:
    backend = ControlledBatchBackend(block=True)
    state = StateObserver()
    batcher = DynamicBatcher(backend, max_size=3, max_wait_seconds=0.5, observer=state)
    tasks = [
        asyncio.create_task(batcher.submit("tenant-a", chat_request(value)))
        for value in ("a", "b", "c")
    ]
    dispatched = await asyncio.wait_for(backend.started.get(), timeout=1)
    await asyncio.wait_for(state.wait_for((0, 1)), timeout=1)

    tasks[1].cancel()
    with pytest.raises(asyncio.CancelledError):
        await tasks[1]
    assert not tasks[0].done()
    assert not tasks[2].done()
    assert len(dispatched) == 3

    backend.release.set()
    results = await asyncio.gather(tasks[0], tasks[2])
    assert [result.response["choices"][0]["message"]["content"] for result in results] == [
        "result:a",
        "result:c",
    ]
    await asyncio.wait_for(state.wait_for((0, 0)), timeout=1)
    await batcher.shutdown()


@pytest.mark.asyncio
async def test_one_backend_failure_reaches_every_surviving_member_without_retry() -> None:
    backend = ControlledBatchBackend(error=BackendUnavailableError())
    state = StateObserver()
    batcher = DynamicBatcher(backend, max_size=3, max_wait_seconds=0.5, observer=state)
    tasks = [
        asyncio.create_task(batcher.submit("tenant-a", chat_request(str(index))))
        for index in range(3)
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    assert len(backend.calls) == 1
    assert all(isinstance(result, BackendUnavailableError) for result in results)
    await asyncio.wait_for(state.wait_for((0, 0)), timeout=1)
    await batcher.shutdown()


@pytest.mark.asyncio
async def test_batch_failure_releases_every_individual_admission_lease() -> None:
    tenant = TenantContext("tenant-a", max_inflight=3, max_queue=3)
    admission = AdmissionController(
        [tenant],
        global_max_inflight=3,
        global_max_queue=3,
        queue_timeout_seconds=1,
    )
    backend = ControlledBatchBackend(error=BackendUnavailableError())
    batcher = DynamicBatcher(backend, max_size=3, max_wait_seconds=0.5)

    async def admitted_request(index: int) -> None:
        async with admission.admit(tenant):
            await batcher.submit(tenant.tenant_id, chat_request(str(index)))

    results = await asyncio.gather(
        *(admitted_request(index) for index in range(3)),
        return_exceptions=True,
    )
    snapshot = await admission.snapshot()

    assert all(isinstance(result, BackendUnavailableError) for result in results)
    assert snapshot.global_inflight == 0
    assert snapshot.global_queued == 0
    assert snapshot.tenants[tenant.tenant_id].inflight == 0
    await admission.shutdown()
    await batcher.shutdown()


@pytest.mark.asyncio
async def test_aggregate_usage_is_counted_once_and_member_usage_is_aggregate_only() -> None:
    backend = ControlledBatchBackend(completion_tokens=27)
    metrics = GatewayMetrics()
    batcher = DynamicBatcher(
        backend,
        max_size=3,
        max_wait_seconds=0.5,
        observer=metrics,
    )
    results = await asyncio.gather(
        *(batcher.submit("tenant-a", chat_request(str(index))) for index in range(3))
    )

    assert all("usage" not in result.response for result in results)
    assert (
        metrics.registry.get_sample_value(
            "gateway_observed_output_tokens_total", {"mode": "non_streaming"}
        )
        == 27
    )
    assert (
        metrics.registry.get_sample_value(
            "gateway_token_accounting_requests_total",
            {"mode": "non_streaming", "result": "aggregate_only"},
        )
        == 3
    )
    await batcher.shutdown()


@pytest.mark.asyncio
async def test_batch_histograms_and_state_gauges_follow_real_transitions() -> None:
    backend = ControlledBatchBackend(block=True)
    metrics = GatewayMetrics()
    batcher = DynamicBatcher(
        backend,
        max_size=2,
        max_wait_seconds=0.5,
        observer=metrics,
    )
    task_a = asyncio.create_task(batcher.submit("tenant-a", chat_request("a")))
    while metrics.registry.get_sample_value("gateway_batch_pending") != 1:  # noqa: ASYNC110 - passive metrics are the state under test
        await asyncio.sleep(0)
    assert metrics.registry.get_sample_value("gateway_batch_inflight") == 0

    task_b = asyncio.create_task(batcher.submit("tenant-a", chat_request("b")))
    await asyncio.wait_for(backend.started.get(), timeout=1)
    assert metrics.registry.get_sample_value("gateway_batch_pending") == 0
    assert metrics.registry.get_sample_value("gateway_batch_inflight") == 1
    assert metrics.registry.get_sample_value("gateway_batch_size_count") == 1
    assert metrics.registry.get_sample_value("gateway_batch_size_sum") == 2
    assert metrics.registry.get_sample_value("gateway_batch_wait_seconds_count") == 2

    backend.release.set()
    await asyncio.gather(task_a, task_b)
    assert metrics.registry.get_sample_value("gateway_batch_inflight") == 0
    await batcher.shutdown()


@pytest.mark.parametrize("usage_result", ["missing", "invalid"])
@pytest.mark.asyncio
async def test_missing_or_invalid_batch_usage_records_member_coverage(
    usage_result: str,
) -> None:
    backend = ControlledBatchBackend(usage_result=usage_result, completion_tokens=None)
    metrics = GatewayMetrics()
    batcher = DynamicBatcher(
        backend,
        max_size=2,
        max_wait_seconds=0.5,
        observer=metrics,
    )

    await asyncio.gather(
        batcher.submit("tenant-a", chat_request("a")),
        batcher.submit("tenant-a", chat_request("b")),
    )

    assert (
        metrics.registry.get_sample_value(
            "gateway_token_accounting_requests_total",
            {"mode": "non_streaming", "result": usage_result},
        )
        == 2
    )
    await batcher.shutdown()


@pytest.mark.asyncio
async def test_shutdown_fails_pending_members_and_leaves_no_owned_work() -> None:
    sleeper = ControlledSleeper()
    backend = ControlledBatchBackend()
    state = StateObserver()
    batcher = DynamicBatcher(
        backend, max_size=4, max_wait_seconds=0.5, sleep=sleeper, observer=state
    )
    task = asyncio.create_task(batcher.submit("tenant-a", chat_request("pending")))
    await asyncio.wait_for(state.wait_for((1, 0)), timeout=1)

    await batcher.shutdown()

    with pytest.raises(BatchingUnavailableError):
        await task
    assert await batcher.snapshot() == (0, 0)
    assert not batcher._timer_tasks
    assert not batcher._flush_tasks
    with pytest.raises(BatchingUnavailableError):
        await batcher.submit("tenant-a", chat_request("late"))


@pytest.mark.asyncio
async def test_shutdown_cancels_active_flush_and_resolves_all_members() -> None:
    backend = ControlledBatchBackend(block=True)
    metrics = GatewayMetrics()
    batcher = DynamicBatcher(
        backend,
        max_size=2,
        max_wait_seconds=0.5,
        observer=metrics,
    )
    tasks = [
        asyncio.create_task(batcher.submit("tenant-a", chat_request(value))) for value in ("a", "b")
    ]
    await asyncio.wait_for(backend.started.get(), timeout=1)

    await batcher.shutdown()
    results = await asyncio.gather(*tasks, return_exceptions=True)

    assert all(isinstance(result, BatchingUnavailableError) for result in results)
    assert await batcher.snapshot() == (0, 0)
    assert not batcher._timer_tasks
    assert not batcher._flush_tasks
    assert (
        metrics.registry.get_sample_value(
            "gateway_batches_total", {"flush_reason": "size", "outcome": "cancelled"}
        )
        == 1
    )


@pytest.mark.asyncio
async def test_shutdown_resolves_a_detached_flush_cancelled_before_it_starts() -> None:
    backend = ControlledBatchBackend(block=True)
    state = StateObserver()
    batcher = DynamicBatcher(
        backend,
        max_size=2,
        max_wait_seconds=0.5,
        observer=state,
    )
    tasks = [
        asyncio.create_task(batcher.submit("tenant-a", chat_request(value))) for value in ("a", "b")
    ]
    await asyncio.wait_for(state.wait_for((0, 1)), timeout=1)

    await batcher.shutdown()
    results = await asyncio.gather(*tasks, return_exceptions=True)

    assert all(isinstance(result, BatchingUnavailableError) for result in results)
    assert backend.calls == []
    assert await batcher.snapshot() == (0, 0)
    assert not batcher._flush_tasks
    assert not batcher._flush_operations


@pytest.mark.asyncio
async def test_concurrent_multi_tenant_grouping_stress_has_no_loss_or_leakage() -> None:
    backend = ControlledBatchBackend()
    batcher = DynamicBatcher(backend, max_size=4, max_wait_seconds=0.5)
    submitted: list[tuple[str, ChatCompletionRequest]] = []
    for tenant_id in ("tenant-a", "tenant-b", "tenant-c"):
        for temperature in (0.4, 0.8):
            for index in range(20):
                submitted.append(
                    (
                        tenant_id,
                        chat_request(
                            f"{tenant_id}:{temperature}:{index}",
                            temperature=temperature,
                        ),
                    )
                )

    results = await asyncio.gather(
        *(batcher.submit(tenant_id, request) for tenant_id, request in submitted)
    )

    assert len(results) == 120
    assert len(backend.calls) == 30
    assert all(len(call) == 4 for call in backend.calls)
    for call in backend.calls:
        tenant_ids = {request.messages[-1].content.split(":", 1)[0] for request in call}
        temperatures = {request.temperature for request in call}
        assert len(tenant_ids) == 1
        assert len(temperatures) == 1
    assert [result.response["choices"][0]["message"]["content"] for result in results] == [
        f"result:{request.messages[-1].content}" for _tenant_id, request in submitted
    ]
    assert await batcher.snapshot() == (0, 0)
    await batcher.shutdown()


@pytest.mark.asyncio
async def test_observer_failure_cannot_break_batch_execution() -> None:
    class BrokenObserver:
        def __getattr__(self, _name: str) -> Awaitable[None]:
            raise RuntimeError("observer failed")

    backend = ControlledBatchBackend()
    batcher = DynamicBatcher(
        backend,
        max_size=2,
        max_wait_seconds=0.5,
        observer=BrokenObserver(),  # type: ignore[arg-type]
    )

    results = await asyncio.gather(
        batcher.submit("tenant-a", chat_request("a")),
        batcher.submit("tenant-a", chat_request("b")),
    )

    assert len(results) == 2
    await batcher.shutdown()
