# Dynamic Batching LLM Inference Gateway

This repository is the v0.5 milestone of a multi-tenant LLM inference gateway:

> Health-aware least-inflight routing across a pool of vLLM backends, with authenticated JSON and
> SSE Chat Completions, failure isolation, probe-based recovery, and disconnect-safe cleanup.

The gateway currently provides typed environment configuration, JSON request logging, request-ID
propagation, stable error responses, health/readiness endpoints, and this inference path:

```text
                         authenticated request
                                  │
                                  ▼
                            tenant admission
                                  │
                                  ▼
                             BackendPool
                                  │
                         filter healthy members
                                  │
                  minimum gateway-observed inflight
                                  │
                   round-robin among tied candidates
                                  │
                ┌─────────────────┼─────────────────┐
                ▼                 ▼                 ▼
             vLLM A            vLLM B            vLLM C
             healthy          unhealthy           healthy
                ▲                                   ▲
                └──────── periodic GET /health ─────┘
                                  │
                     JSON response or SSE byte relay
                                  │
              end/error/disconnect → release both leases
```

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

Successful JSON responses from vLLM are returned without discarding OpenAI-compatible extension
fields. Safe upstream `400`, `404`, and `422` statuses retain their status with a normalized error
body. Backend connection failures return `503`, timeouts return `504`, and invalid responses or
other upstream failures return `502`. Upstream `401` and `403` responses are treated as backend
configuration failures, not gateway-client authentication failures.

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
relayed incrementally without parsing or reconstructing events.

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

Streaming requests use the direct backend path and will bypass future gateway dynamic batching.

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
docker build -t inference-gateway:v0.5 .
docker run --rm -p 8080:8080 \
  -e 'BACKENDS_JSON={"local":{"base_url":"http://host.docker.internal:8000"}}' \
  -e 'TENANTS_JSON={"local":{"api_key":"local-example-key","max_inflight":2,"max_queue":4}}' \
  inference-gateway:v0.5
```

The image runs as a non-root user. No vLLM server or GPU stack is bundled into the image.

## Routing limitations

Routing health and inflight state are process-local and are not shared between gateway replicas.
Configured backends are assumed interchangeable for the models exposed through this gateway; v0.5
does not route by model or LoRA adapter. Backend inflight means requests assigned by this gateway
whose backend operation has not finished. It is not vLLM scheduler state, GPU utilization, KV-cache
occupancy, prefix-cache affinity, token load, latency, or a prediction of backend capacity. Health is
the result of HTTP `/health` probes plus simple immediate ejection on typed backend-origin failures.
No generation request is automatically retried on another member.

## Not implemented

The v0.5 API does not support WebSockets, stream reconnection/resume, automatic generation retries,
gateway dynamic batching, tools/function calls, multimodal content, structured outputs, RPS
limiting, token/billing quotas, JWT/OAuth, persistent tenants, model/GPU/KV/token/cache-aware or
predictive routing, Prometheus, TTFT/TPS metrics, Grafana, a benchmark harness, Redis/distributed
admission or routing state, databases, Docker Compose, Kubernetes, Terraform, or autoscaling.
