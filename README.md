# Dynamic Batching LLM Inference Gateway

This repository is the v0.2 milestone of a multi-tenant LLM inference gateway:

> FastAPI OpenAI Chat Completions-compatible gateway with explicit request validation and a real
> vLLM HTTP backend for non-streaming generation.

The gateway currently provides typed environment configuration, JSON request logging, request-ID
propagation, stable error responses, health/readiness endpoints, and one real inference path:

```text
client → POST /v1/chat/completions → FastAPI gateway → pooled async HTTP → vLLM
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
accepted from a client request. Connection/request timeouts and non-tenant request limits are
documented in `.env.example`.

Check gateway health:

```bash
curl http://localhost:8080/healthz
curl http://localhost:8080/readyz
```

Forward a non-streaming completion to vLLM:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "<served-model>",
    "messages": [
      {"role": "user", "content": "Say hello in one sentence."}
    ],
    "max_tokens": 32
  }'
```

An official OpenAI-style client can target the same endpoint without gateway authentication in
v0.2:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused-for-pr2")
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
```

## Docker

```bash
docker build -t inference-gateway:v0.2 .
docker run --rm -p 8080:8080 \
  -e VLLM_BASE_URL=http://host.docker.internal:8000 \
  inference-gateway:v0.2
```

The image runs as a non-root user. No vLLM server or GPU stack is bundled into the image.

## Not implemented

The v0.2 API does not support `stream=true`, tools/function calls, multimodal content, structured
outputs, gateway client authentication, tenant admission or limits, queues, dynamic batching,
multi-backend pools/routing/failover, backend health probing, Prometheus, Grafana, a benchmark
harness, Redis, databases, Docker Compose, Kubernetes, or Terraform.
