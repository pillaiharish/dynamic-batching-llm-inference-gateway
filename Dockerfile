FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN groupadd --system gateway \
    && useradd --system --gid gateway --home-dir /app gateway

COPY pyproject.toml README.md ./
COPY gateway ./gateway

RUN pip install --no-cache-dir .

USER gateway

EXPOSE 8080

CMD ["sh", "-c", "exec uvicorn gateway.app:app --host \"${HOST:-0.0.0.0}\" --port \"${PORT:-8080}\""]
