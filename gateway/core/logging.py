"""Structured logging configuration."""

from __future__ import annotations

import json
import logging
import logging.config
from datetime import UTC, datetime
from typing import Any

from gateway.core.request_id import get_request_id


class JSONFormatter(logging.Formatter):
    """Render application log records as single-line JSON objects."""

    _structured_fields = (
        "method",
        "path",
        "status_code",
        "duration_ms",
        "error_code",
        "backend",
        "backend_id",
        "backend_healthy",
        "routing_result",
        "upstream_status",
        "tenant_id",
        "admission_result",
        "streaming",
        "stream_outcome",
        "stream_error_type",
    )

    def format(self, record: logging.LogRecord) -> str:
        event: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        request_id = getattr(record, "request_id", None) or get_request_id()
        if request_id is not None:
            event["request_id"] = request_id

        for field in self._structured_fields:
            value = getattr(record, field, None)
            if value is not None:
                event[field] = value

        if record.exc_info:
            event["exception"] = self.formatException(record.exc_info)

        return json.dumps(event, default=str, separators=(",", ":"))


def configure_logging(log_level: str) -> None:
    """Configure a JSON formatter for application and dependency logs."""
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {"json": {"()": "gateway.core.logging.JSONFormatter"}},
            "handlers": {
                "default": {
                    "class": "logging.StreamHandler",
                    "formatter": "json",
                    "stream": "ext://sys.stdout",
                }
            },
            "root": {"handlers": ["default"], "level": log_level},
        }
    )
