# Dynamic Batching LLM Inference Gateway

This repository contains the **v0.1 foundation for a multi-tenant LLM inference gateway**.
It currently provides a FastAPI application factory, typed environment configuration, JSON
logging, request ID propagation, consistent error responses, health endpoints, and an
in-process fake implementation of the backend interface.

## Local development

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
cp .env.example .env
python -m uvicorn gateway.app:app --host 0.0.0.0 --port 8080
```

Verify the running process:

```bash
curl http://localhost:8080/healthz
curl http://localhost:8080/readyz
```

The endpoints return `{"status":"ok"}` and `{"status":"ready"}`, respectively. Incoming
`X-Request-ID` values are returned in the response; when absent, the gateway creates a UUID.
The header name and other settings can be changed using the variables in `.env.example`.

Run the local checks with:

```bash
ruff check .
ruff format --check .
pytest
```

## Docker

```bash
docker build -t inference-gateway .
docker run --rm -p 8080:8080 inference-gateway
curl http://localhost:8080/healthz
```

The container runs as a non-root user. `HOST` and `PORT` environment variables can override its
default bind address and port.

## Roadmap (not yet implemented)

Future versions may add OpenAI-compatible chat completions, tenant admission controls, dynamic
batching, backend routing and vLLM integration, SSE streaming, Prometheus metrics, and benchmarking.
None of those features are implemented in v0.1.
