"""Request ID context and middleware."""

from __future__ import annotations

import logging
from contextvars import ContextVar, Token
from time import perf_counter
from typing import TYPE_CHECKING
from uuid import uuid4

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

if TYPE_CHECKING:
    from fastapi import FastAPI


_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
logger = logging.getLogger("gateway.request")


def get_request_id() -> str | None:
    """Return the request ID associated with the current async context, if any."""
    return _request_id.get()


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Propagate or create a request ID and emit one structured access log."""

    def __init__(self, app: ASGIApp, *, header_name: str) -> None:
        super().__init__(app)
        self.header_name = header_name

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get(self.header_name) or str(uuid4())
        request.state.request_id = request_id
        token: Token[str | None] = _request_id.set(request_id)
        started_at = perf_counter()

        try:
            response = await call_next(request)
            response.headers[self.header_name] = request_id
            duration_ms = round((perf_counter() - started_at) * 1000, 3)
            logger.info(
                "request completed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": duration_ms,
                },
            )
            return response
        finally:
            _request_id.reset(token)


def install_request_id_middleware(app: FastAPI, *, header_name: str) -> None:
    """Install request ID handling with a configured response header name."""
    app.add_middleware(RequestIDMiddleware, header_name=header_name)
