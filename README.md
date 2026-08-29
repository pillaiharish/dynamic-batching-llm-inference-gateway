# Dynamic Batching LLM Inference Gateway

This repository is the v0.7 milestone of a multi-tenant LLM inference gateway:

> Compatibility-aware dynamic batching of eligible non-streaming Chat Completions requests, using
> bounded size/time flushes, one upstream vLLM batch request, strict response demultiplexing,
> explicit streaming bypass, and batch observability.

The gateway currently provides typed environment configuration, JSON request logging, request-ID
propagation, stable error responses, health/readiness endpoints, and this inference path:

```text
request received T0
        │
        ▼
auth / validation
        │
        ▼
admission begins TQ ──────────────┐
        │                         │ queue wait
        ▼                         │
admission granted T1 ◀────────────┘
        │
        ▼
health-aware backend routing
        │
        ▼
leaf vLLM HTTP request begins T2
        │
        ▼
first generated-content SSE event T3
        │
        ▼
JSON response / transparent SSE byte relay
        │
        ▼
completion, error, or disconnect T4
```

The Prometheus timing contract is `T1−TQ` admission queue wait, `T3−T0` client-visible TTFT,
`T3−T2` gateway-observed backend TTFT, and `T4−T0` end-to-end request duration. All duration
timestamps use the monotonic `time.perf_counter()` clock.

## Supported Chat Completions subset

`POST /v1/chat/completions` accepts simple text messages with the `system`, `user`, or `assistant`
role. Unknown request and message fields are rejected.

Supported request fields:

- `model` — required, non-empty string
- `messages` — required, non-empty list of text messages
- `temperature` — `0.0` through `2.0`; defaults to `1.0`
- `top_p` — `0.0` through `1.0`; defaults to `1.0`
- `max_tokens` — positive integer, capped by `MAX_COMPLETION_TOKENS`
- `stop` — a non-empty string or list of non-empty strings
- `seed` — integer
- `n` — positive integer, capped by `MAX_CHOICES`; defaults to `1`
- `stream` — boolean selecting a JSON response (`false`) or SSE relay (`true`); defaults to `false`
- `stream_options.include_usage` — optional boolean requesting vLLM's final authoritative streaming
  usage event; `stream_options` is accepted only with `stream=true`

Successful direct JSON responses from vLLM are returned without discarding OpenAI-compatible
extension fields. Batched responses follow the stricter demultiplexing contract below. Safe upstream
`400`, `404`, and `422` statuses retain their status with a normalized error body. Backend
connection failures return `503`, timeouts return `504`, and invalid responses or other upstream
failures return `502`. Upstream `401` and `403` responses are treated as backend configuration
failures, not gateway-client authentication failures.

## Dynamic batching

> Dynamic batching in v0.7 means gateway-side aggregation of compatible admitted non-streaming
> requests into one vLLM `/v1/chat/completions/batch` HTTP request.

Clients still call only `POST /v1/chat/completions` and receive ordinary single-request Chat
Completion responses. The batch endpoint is an internal gateway → vLLM implementation detail; the
gateway does not expose it as a client API. vLLM still owns GPU scheduling, continuous batching,
KV-cache management, prefill/decode scheduling, and execution. Gateway aggregation does not replace
any of those vLLM responsibilities.

```text
N authenticated and admitted requests
                 ↓
tenant-local compatibility groups
                 ↓
          size OR timeout flush
                 ↓
one POST /v1/chat/completions/batch
                 ↓
 strict indexed demultiplexing
                 ↓
N ordinary client responses
```

The default remains `DYNAMIC_BATCHING_ENABLED=false`. Configure batching with:

```text
DYNAMIC_BATCHING_ENABLED=false
DYNAMIC_BATCH_MAX_SIZE=8
DYNAMIC_BATCH_MAX_WAIT_SECONDS=0.005
```

The maximum size validates from 2 through 64 and maximum wait validates in the interval `(0, 1]`
seconds, even while batching is disabled. v0.7 makes gateway dynamic batching functionally real but
does not claim that it improves throughput or latency. vLLM already performs continuous batching
internally, so gateway-side request aggregation may improve, hurt, or have negligible performance
impact depending on workload and deployment. PR8 exists specifically to measure that impact.

### Eligibility and compatibility

| Request | Gateway batch? |
| --- | --- |
| `stream=true` | No |
| `stream=false, n=1`, batching enabled | Eligible |
| `stream=false, n>1` | No; ordinary generation path |
| batching disabled | No; ordinary generation path |
| same tenant + compatible parameters | Same group possible |
| different tenant | Never the same group |
| incompatible generation parameters | Separate groups |

Eligibility never reduces the public API subset: streaming retains the existing SSE path and
`n>1` retains ordinary `BackendPool.generate()`. Authentication, request validation, and individual
tenant/global admission all happen before an eligible request waits in the dynamic batcher. Each
logical member keeps its admission lease through its batch wait and result delivery.

The compatibility key contains `tenant_id` plus a deterministic canonical JSON representation of
`request.to_upstream_payload()` after removing `messages` and normalizing only the guaranteed batch
invariants `stream=false`, `n=1`, and absent `stream_options`. Messages intentionally do not affect
compatibility. All current and future supported shared generation fields therefore participate
automatically. Field presence is preserved: omitting `temperature` is not assumed equivalent to
explicitly sending `temperature=1.0`. Compatibility keys may contain user-controlled generation
fields, so they are neither logged nor exposed as metric labels.

v0.7 deliberately batches only within one tenant. Cross-tenant batching is excluded to avoid
creating a shared batch-level failure domain across tenants. This tradeoff may reduce aggregation
opportunities and is intentional.

### Flush, response, and cancellation semantics

The timer starts when the first member enters an empty compatibility group and is not restarted by
later arrivals. A pending group flushes when it reaches `DYNAMIC_BATCH_MAX_SIZE` or that first-member
timer reaches `DYNAMIC_BATCH_MAX_WAIT_SECONDS`, whichever occurs first. Detachment is atomic; new
arrivals can immediately start the next group while the detached batch executes without holding the
batcher lock. Timer and flush tasks are batcher-owned and are awaited or cancelled during shutdown.
Dynamic batches are process-local and never span gateway processes or replicas.

One detached group selects one healthy backend and makes exactly one upstream HTTP operation. The
gateway requires one indexed choice per conversation, unique integer indices covering exactly
`0..batch_size-1`, and maps by index even if choices arrive out of order. Any missing, duplicate,
non-integer, negative, out-of-range, or incomplete association fails the whole surviving batch with
`502 backend_protocol_error`; the gateway never guesses. Each client receives only its choice,
reindexed to zero, together with safe batch fields such as `id`, `object`, `created`, `model`, and
`system_fingerprint` when present. Batched members share the upstream vLLM batch response ID because
the gateway does not fabricate per-member upstream IDs.

Current vLLM batch usage is aggregate across the entire batch. v0.7 therefore does not copy
aggregate `usage` into each demultiplexed client response and does not divide it across members.
Aggregate completion tokens remain valid for gateway-level observed output TPS and are counted
exactly once. Per-member accounting coverage is recorded as `aggregate_only` when that aggregate is
valid, or `missing`/`invalid` as appropriate.

Cancelling a member before dispatch removes it from pending state and from the eventual upstream
payload. Cancelling one member after dispatch does not cancel the shared vLLM request, allowing
other members to finish; the cancelled member's result is discarded. Individual member
cancellation after dispatch cannot selectively cancel only that conversation in the shared vLLM
batch request. Its gateway admission lease may release, while backend logical inflight continues to
count the original dispatched batch weight until the upstream operation ends.

## Backend routing and health

Production backends are configured through one typed `BACKENDS_JSON` mapping. The mapping key is a
non-blank, unique backend ID; each value contains an HTTP(S) `base_url` and an optional secret
`api_key`:

```json
{
  "gpu-a": {
    "base_url": "http://vllm-a:8000",
    "api_key": "backend-a-secret"
  },
  "gpu-b": {
    "base_url": "http://vllm-b:8000"
  }
}
```

Each member owns its own long-lived pooled HTTP client. Startup probes every member concurrently,
then one pool-owned task repeats `GET /health` at `BACKEND_HEALTH_INTERVAL_SECONDS`; each call is
bounded by `BACKEND_HEALTH_TIMEOUT_SECONDS`. Health traffic does not use inference admission or
backend inflight. Connection failures, timeouts, and non-2xx health responses mark only that member
unhealthy. A later successful probe restores it to the routing candidates. The process can start
with only a subset healthy, but at least one backend must be configured.

For every admitted operation the pool first filters healthy backends, finds the minimum
gateway-observed inflight count, then uses a deterministic rotating cursor among only those tied
candidates. In configured order, sequential equal-load assignments rotate `A → B → C → A`. The
selection and inflight increment occur under one lock. Non-streaming operations release their
backend assignment in `finally`; streaming operations retain it until the routed stream reaches
EOF, fails, is closed, or is cancelled.

A batch of `N` is one selected backend HTTP operation but adds `N` to that backend's logical
inflight count until the batch operation ends:

```text
batch of N → one backend selected → backend logical inflight += N → one batch HTTP operation
```

This keeps least-inflight routing aware of logical work already assigned. Accordingly,
`gateway_backend_requests_total{operation="batch"}` increments once per upstream batch, not once per
member; `gateway_backend_inflight` is a gateway logical-request count, not an HTTP-operation count.

Backend-origin connection, timeout, HTTP, authentication/configuration, or protocol failures mark
the selected member unhealthy so future requests avoid it. Client-controlled `400`, `404`, and
`422` request rejections do not eject a backend. No failed generation is automatically replayed on
another backend, even before output, because the gateway cannot prove that upstream work never
started. This is failure isolation and future-request routing, not transparent generation failover.

`/healthz` is process liveness and remains healthy while the gateway is alive. `/readyz` is
routability: it returns `200 {"status":"ready"}` only after initialization and while at least one
production-pool backend is healthy; otherwise it returns `503 {"status":"not_ready"}`. When no
backend is healthy, admitted JSON and streaming requests fail before generation with the stable
JSON error `503 no_healthy_backend` and preserve their request ID.

## Streaming behavior

With `stream=false`, the gateway waits for vLLM's complete response and returns one JSON Chat
Completion. With `stream=true`, it first acquires normal tenant/global admission, opens and validates
the upstream vLLM response, and then returns `Content-Type: text/event-stream`. Upstream SSE bytes,
including comments, fragmented events, multiple events per network chunk, and `data: [DONE]`, are
relayed incrementally without reconstruction. A passive metrics observer receives a copy of those
bytes; it never changes framing, content, or `[DONE]` and never synthesizes downstream events.

The observer keeps at most 1 MiB for one incomplete event. It understands HTTP-chunk fragmentation
and coalescing, ignores comments and role-only deltas, and recognizes first generated text only from
a complete JSON `data:` event with non-empty `choices[*].delta.content`. Malformed or oversized
events disable parsing for that stream while byte relay continues unchanged. Event content is not
logged on parser failure.

The streaming admission lease and selected backend's routing inflight assignment both remain held
until the downstream response ends or is cancelled. A client disconnect stops consumption, closes
the upstream HTTP stream so vLLM can observe that disconnect, and releases both ownership records.
Client cancellation does not mark the selected backend unhealthy. This does not claim immediate
GPU-kernel cancellation.

Failures discovered while opening the upstream response remain normal JSON errors: connection
failures return `503`, timeouts return `504`, safe request rejections retain their mapped client
status, and incompatible streaming responses return `502`. Once SSE bytes have begun, the gateway
cannot change the HTTP status; a mid-stream upstream failure closes the stream, logs safe lifecycle
metadata, and releases resources. Generation is never retried automatically because replaying after
tokens were delivered could duplicate or diverge output.

Streaming requests always use the direct backend path and never wait in dynamic batching or call
the internal `/v1/chat/completions/batch` endpoint.

## Tenant authentication

Chat completions require `Authorization: Bearer <tenant-api-key>`. Health and readiness endpoints
remain unauthenticated. Tenants are configured through `TENANTS_JSON`, where each tenant has a
unique secret API key, `max_inflight`, and `max_queue` value. Missing, malformed, or unknown
credentials return the same safe `401 unauthorized` response with `WWW-Authenticate: Bearer`.

Tenant keys authenticate client → gateway traffic. Each optional `BACKENDS_JSON` `api_key`
separately authenticates gateway → vLLM traffic for only that member. Client authorization is never
forwarded upstream, and neither credential type is logged.

## Admission semantics

For each authenticated request, both the tenant and process-wide inflight limit must have capacity
before backend execution begins. Otherwise the request enters that tenant's bounded FIFO queue.
The gateway enforces:

- Per-tenant active concurrency: tenant `max_inflight`
- Per-tenant waiting bound: tenant `max_queue`
- Process-wide active concurrency: `GLOBAL_MAX_INFLIGHT`
- Process-wide waiting bound: `GLOBAL_MAX_QUEUE`
- Bounded queue wait: `ADMISSION_QUEUE_TIMEOUT_SECONDS`
- Deterministic round-robin admission opportunities across tenants with queued work

Queue-full and wait-timeout failures return normalized `429` errors. Queued cancellation removes
the waiter, and backend errors or cancellation after admission release the slot. The fairness
guarantee concerns queue admission opportunity: a noisy tenant cannot force another tenant to wait
for its whole backlog to drain. It does not promise equal throughput when tenant concurrency limits
differ.

These per-tenant limits mean **active concurrency plus bounded waiting queue**. They are not
requests-per-second rate limiting, billing quotas, token quotas, or production IAM.

Admission state is single-process and in-memory. Multiple gateway replicas each enforce their own
independent limits; no counters or queues are shared. Admission protects backend requests at the
gateway layer and is not GPU-aware, KV-cache-aware, token-budget-aware, or connected to the vLLM
scheduler.

## Prometheus observability

`GET /metrics` exposes the application's own `prometheus-client` `CollectorRegistry` using the
Prometheus text content type. The endpoint is unauthenticated for ordinary scraping and is excluded
from inference request metrics. Production deployments should network-restrict this observability
endpoint as appropriate. `/healthz` and `/readyz` are also excluded from the focused inference
request families.

Every application instance owns an isolated registry, so tests and multiple in-process app objects
cannot duplicate or leak collector state. Metrics updates are synchronous, in-memory observations;
they perform no remote calls or disk writes. The passive SSE parser fails open so an observability
failure cannot become an inference failure.

### Timing contract

| Timestamp | Definition |
| --- | --- |
| `T0` | Gateway receives the HTTP request, before validation or authentication |
| `TQ` | Request begins waiting for `AdmissionController` |
| `T1` | Admission lease is granted |
| `T2` | Selected leaf vLLM backend begins the upstream HTTP request |
| `T3` | Gateway observes the first SSE event containing non-empty generated assistant content |
| `T4` | Downstream request/stream lifecycle ends, including disconnect or error |

| Metric | Meaning |
| --- | --- |
| Request duration | `T4−T0` |
| Admission queue wait | `T1−TQ`, or wait until a failed acquire |
| Dynamic batch wait | Batch dispatch minus dynamic-batcher submission |
| Client-visible TTFT | `T3−T0` |
| Backend-observed TTFT | `T3−T2` |
| Observed output TPS | Prometheus `rate()` of authoritative `completion_tokens` |

Request duration covers the full streaming body lifecycle, not just response-header creation.
Backend TTFT includes gateway-observed backend HTTP, vLLM waiting, and prefill until content becomes
observable; it is not described as GPU-execution-only time.

Gateway TTFT is stream-observed TTFT. Non-streaming requests do not produce a gateway TTFT
observation because the gateway receives the completed response only after generation finishes.
The first HTTP/TCP chunk, SSE comment, role-only delta, and `[DONE]` do not end TTFT. Histograms use
explicit inference-oriented seconds buckets; the TTFT buckets include `0.8` seconds for later
800-ms SLA experiments. Percentiles are calculated with PromQL rather than inside the gateway.

Admission queue wait and dynamic batch wait are distinct. An eligible request can experience both
after gateway receipt: admission wait occurs before its lease is granted, while batch wait begins
only after admission and ends at batch dispatch. End-to-end request duration naturally includes
both. Dynamic batching does not create client or backend TTFT observations.

### Token accounting and TPS

`gateway_observed_output_tokens_total` is a cumulative counter. The gateway increments it only from
a valid, non-negative integer `usage.completion_tokens` in a successful non-streaming JSON response
or passively observed streaming usage event. For a successful dynamic batch, aggregate
`completion_tokens` increments this counter once for the whole upstream batch while each delivered
member records `result="aggregate_only"`. SSE event count, delta count, characters, whitespace,
bytes, and HTTP chunks are never treated as tokens.

For streaming requests, clients that need complete accounting should request authoritative usage:

```json
{
  "stream": true,
  "stream_options": {
    "include_usage": true
  }
}
```

The gateway does not silently add this option. Without a final usage event, the stream remains
transparent and token coverage is recorded as `missing`; malformed usage is `invalid`. The
`gateway_token_accounting_requests_total{mode,result}` counter makes this partial coverage visible.

TPS means **observed generated output tokens per second**. It does not mean HTTP requests, SSE
chunks, characters, or bytes per second. No application-side instantaneous TPS gauge exists; use:

```promql
sum(rate(gateway_observed_output_tokens_total[1m]))
```

The unit is generated output tokens/second, and the result is only as complete as the accompanying
authoritative token-accounting coverage.

### Metric families and bounded labels

| Metric family | Labels |
| --- | --- |
| `gateway_requests_total` | `mode`, `status_code`, `outcome` |
| `gateway_request_duration_seconds` | `mode`, `outcome` |
| `gateway_admission_queue_wait_seconds` | `outcome` |
| `gateway_client_ttft_seconds` | `backend_id` |
| `gateway_backend_ttft_seconds` | `backend_id` |
| `gateway_ttft_observations_total` | `result` |
| `gateway_observed_output_tokens_total` | `mode` |
| `gateway_token_accounting_requests_total` | `mode`, `result` |
| `gateway_errors_total` | `code` |
| `gateway_admission_inflight`, `gateway_admission_queued` | none |
| `gateway_tenant_admission_inflight`, `gateway_tenant_admission_queued` | `tenant_id` |
| `gateway_backend_healthy`, `gateway_backend_inflight` | `backend_id` |
| `gateway_backend_requests_total` | `backend_id`, `operation`, `outcome` |
| `gateway_batch_eligibility_total` | `decision`, `reason` |
| `gateway_batches_total` | `flush_reason`, `outcome` |
| `gateway_batch_size` | none |
| `gateway_batch_wait_seconds` | none |
| `gateway_batch_pending`, `gateway_batch_inflight` | none |

`mode`, lifecycle outcomes, operation, accounting result, HTTP status (or `unknown` when no response
status was sent), and stable error code come from small controlled sets. Backend and tenant IDs are
bounded by static configuration, with a singleton `unknown` backend fallback for injected streams
that provide no metadata; metric cardinality therefore scales with configured backends and tenants.
Request IDs, API keys,
authorization headers, prompts, generated content, raw URLs, arbitrary client model names, and
exception messages are never labels.

`gateway_errors_total` owns client-visible gateway lifecycle failures. A mid-stream upstream error,
which occurs after HTTP 200 can no longer change, is counted once as `stream_upstream_error`.
`gateway_backend_requests_total` separately owns the selected leaf operation's full outcome;
`operation="stream", outcome="success"` means normal stream completion, not merely successful
response opening. For batching, `gateway_batch_pending` counts admitted logical requests awaiting
dispatch, while `gateway_batch_inflight` counts detached upstream batch HTTP operations. A
dispatched eight-member batch therefore has batch pending `0`, batch inflight `1`, and backend
logical inflight increased by `8`.

Admission gauges are updated on controller acquire, queue, dispatch, timeout, cancellation, release,
and shutdown transitions. Backend gauges are updated on initial probe, health loss/recovery,
selection, release, stream completion/disconnect, and pool close. They observe the real scheduling
and routing state rather than polling snapshots during a scrape. `gateway_backend_healthy` uses
`1` for healthy and `0` for unhealthy.

### PromQL examples

Request rate:

```promql
sum(rate(gateway_requests_total[1m]))
```

Request p95:

```promql
histogram_quantile(
  0.95,
  sum by (le) (rate(gateway_request_duration_seconds_bucket[5m]))
)
```

Queue-wait p95:

```promql
histogram_quantile(
  0.95,
  sum by (le) (rate(gateway_admission_queue_wait_seconds_bucket[5m]))
)
```

Client TTFT p95:

```promql
histogram_quantile(
  0.95,
  sum by (le) (rate(gateway_client_ttft_seconds_bucket[5m]))
)
```

Backend TTFT p95 by backend:

```promql
histogram_quantile(
  0.95,
  sum by (backend_id, le) (rate(gateway_backend_ttft_seconds_bucket[5m]))
)
```

Errors by stable code:

```promql
sum by (code) (rate(gateway_errors_total[5m]))
```

Token-accounting coverage:

```promql
sum by (mode, result) (
  rate(gateway_token_accounting_requests_total[5m])
)
```

Batch rate by flush reason and outcome:

```promql
sum by (flush_reason, outcome) (
  rate(gateway_batches_total[1m])
)
```

Mean batch size:

```promql
rate(gateway_batch_size_sum[5m])
/
rate(gateway_batch_size_count[5m])
```

Batch-wait p95:

```promql
histogram_quantile(
  0.95,
  sum by (le) (rate(gateway_batch_wait_seconds_bucket[5m]))
)
```

Batch bypass rate:

```promql
sum by (reason) (
  rate(gateway_batch_eligibility_total{decision="bypass"}[1m])
)
```

The scrape example is in `observability/prometheus/prometheus.yml`; the static Grafana dashboard is
in `observability/grafana/dashboards/gateway-overview.json`. Histograms are used instead of
client-side quantile summaries so replicas can be aggregated later and p50/p95/p99 can be selected
at query time.

v0.7 does not configure `prometheus-client` multiprocess mode. Metrics represent one gateway
process, matching the current process-local admission, batching, and routing architecture. If the
gateway later runs with multiple worker processes, aggregation semantics—especially gauges—must be
revisited. The gateway does not expose synthetic GPU, KV-cache, or vLLM scheduler metrics and does
not scrape
backend `/metrics`; Prometheus can scrape those authoritative sources separately in a later
deployment.

## Local development

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
cp .env.example .env
python -m uvicorn gateway.app:app --host 0.0.0.0 --port 8080
```

Configure one or more trusted, model-compatible OpenAI-compatible vLLM servers through
`BACKENDS_JSON`. A member's optional `api_key` is sent only to that configured backend and is never
accepted from a client request. Configure safe local tenant credentials through `TENANTS_JSON` and
adjust global admission and health timing only as needed. All settings are documented in
`.env.example`.

Check gateway health:

```bash
curl http://localhost:8080/healthz
curl http://localhost:8080/readyz
curl http://localhost:8080/metrics
```

Forward a non-streaming completion to vLLM:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <tenant-api-key>' \
  -d '{
    "model": "<served-model>",
    "messages": [
      {"role": "user", "content": "Say hello in one sentence."}
    ],
    "max_tokens": 32
  }'
```

Relay a streaming completion incrementally (`-N` disables curl output buffering):

```bash
curl -N http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <tenant-api-key>' \
  -d '{
    "model": "<served-model>",
    "messages": [
      {"role": "user", "content": "Count from one to five slowly."}
    ],
    "stream": true,
    "stream_options": {"include_usage": true},
    "max_tokens": 64
  }'
```

Expected output arrives as vLLM produces it and ends with the upstream sentinel:

```text
data: {...}

data: {...}

data: [DONE]
```

An official OpenAI-style client can use its `api_key` value as the tenant bearer credential:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="<tenant-api-key>")
response = client.chat.completions.create(
    model="<served-model>",
    messages=[{"role": "user", "content": "Hello"}],
)
```

Pass `stream=True` to the same SDK call to consume the proxied streaming response.

The OpenAI SDK is not a project dependency; CI exercises the wire contract directly.

Run local checks with:

```bash
ruff check .
ruff format --check .
pytest -q
python -m pip check
```

## Docker

```bash
docker build -t inference-gateway:v0.7 .
docker run --rm -p 8080:8080 \
  -e 'BACKENDS_JSON={"local":{"base_url":"http://host.docker.internal:8000"}}' \
  -e 'TENANTS_JSON={"local":{"api_key":"local-example-key","max_inflight":2,"max_queue":4}}' \
  inference-gateway:v0.7
```

The image runs as a non-root user. No vLLM server or GPU stack is bundled into the image.

## Routing limitations

Routing health and inflight state are process-local and are not shared between gateway replicas.
Configured backends are assumed interchangeable for the models exposed through this gateway; v0.7
does not route by model or LoRA adapter. Backend inflight means requests assigned by this gateway
whose backend operation has not finished. It is not vLLM scheduler state, GPU utilization, KV-cache
occupancy, prefix-cache affinity, token load, latency, or a prediction of backend capacity. Health is
the result of HTTP `/health` probes plus simple immediate ejection on typed backend-origin failures.
No generation request is automatically retried on another member.

## Not implemented

The v0.7 API does not support WebSockets, stream reconnection/resume, automatic generation retries,
cross-tenant or streaming batching, token-budget or prompt-length batching, per-member batch retry,
tools/function calls, multimodal content, structured outputs, RPS limiting, token/billing quotas,
JWT/OAuth, persistent tenants, model/GPU/KV/token/cache/prefix-aware or predictive routing and
batching, distributed batching, benchmark traffic generation, automatic performance conclusions,
GPU/NVML/DCGM metrics, vLLM `/metrics` federation, OpenTelemetry, tracing/exemplars, push metrics,
alert rules, Redis/distributed admission, batching, or routing state, databases, Docker Compose,
Kubernetes, Terraform, or autoscaling. The included Prometheus scrape configuration and Grafana
dashboard are static examples, not runtime dependencies.
