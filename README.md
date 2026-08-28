# Dynamic Batching LLM Inference Gateway

This repository is the v0.3 milestone of a multi-tenant LLM inference gateway:

> FastAPI OpenAI Chat Completions-compatible inference gateway with authenticated tenants and
> bounded per-tenant/global concurrency and queue limits.

The gateway currently provides typed environment configuration, JSON request logging, request-ID
propagation, stable error responses, health/readiness endpoints, and this inference path:

```text
                         Tenant A queue ─┐
client → bearer auth →  Tenant B queue ─┼→ fair admission → pooled async HTTP → vLLM
                         Tenant C queue ─┘
                                  │
                            global limits
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
- `stream` — must be `false`; defaults to `false`

Successful JSON responses from vLLM are returned without discarding OpenAI-compatible extension
fields. Safe upstream `400`, `404`, and `422` statuses retain their status with a normalized error
body. Backend connection failures return `503`, timeouts return `504`, and invalid responses or
other upstream failures return `502`. Upstream `401` and `403` responses are treated as backend
configuration failures, not gateway-client authentication failures.

## Tenant authentication

Chat completions require `Authorization: Bearer <tenant-api-key>`. Health and readiness endpoints
remain unauthenticated. Tenants are configured through `TENANTS_JSON`, where each tenant has a
unique secret API key, `max_inflight`, and `max_queue` value. Missing, malformed, or unknown
credentials return the same safe `401 unauthorized` response with `WWW-Authenticate: Bearer`.

Tenant keys authenticate client → gateway traffic. `VLLM_API_KEY` separately authenticates gateway
→ vLLM traffic. Client authorization is never forwarded upstream, and neither credential is logged.

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

Configure `VLLM_BASE_URL` for the trusted OpenAI-compatible vLLM server. If that server requires
authentication, set `VLLM_API_KEY`; the key is sent only to the configured backend and is never
accepted from a client request. Configure safe local tenant credentials through `TENANTS_JSON` and
adjust global admission bounds only as needed. All settings are documented in `.env.example`.

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

An official OpenAI-style client can use its `api_key` value as the tenant bearer credential:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="<tenant-api-key>")
response = client.chat.completions.create(
    model="<served-model>",
    messages=[{"role": "user", "content": "Hello"}],
)
```

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
docker build -t inference-gateway:v0.3 .
docker run --rm -p 8080:8080 \
  -e VLLM_BASE_URL=http://host.docker.internal:8000 \
  -e 'TENANTS_JSON={"local":{"api_key":"local-example-key","max_inflight":2,"max_queue":4}}' \
  inference-gateway:v0.3
```

The image runs as a non-root user. No vLLM server or GPU stack is bundled into the image.

## Not implemented

The v0.3 API does not support `stream=true`, tools/function calls, multimodal content, structured
outputs, RPS limiting, token/billing quotas, JWT/OAuth, persistent tenants, dynamic batching,
multi-backend pools/routing/failover, backend health probing, Prometheus, Grafana, a benchmark
harness, Redis/distributed admission, databases, Docker Compose, Kubernetes, or Terraform.
