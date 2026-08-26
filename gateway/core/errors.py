"""Gateway exception types and HTTP error handlers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from fastapi import Request
from fastapi.responses import JSONResponse

from gateway.core.request_id import get_request_id

if TYPE_CHECKING:
    from fastapi import FastAPI


logger = logging.getLogger(__name__)


class GatewayError(Exception):
    """Base class for expected gateway failures safe to expose to clients."""

    code = "gateway_error"
    default_message = "Gateway request failed"
    status_code = 500

    def __init__(
        self,
        message: str | None = None,
        *,
        code: str | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message or self.default_message)
        self.message = message or self.default_message
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code


class ConfigurationError(GatewayError):
    """Raised when gateway configuration cannot be used."""

    code = "configuration_error"
    default_message = "Gateway configuration is invalid"


class BackendError(GatewayError):
    """Raised when an inference backend operation fails."""

    code = "backend_error"
    default_message = "Inference backend failed"
    status_code = 502


def _request_id_from(request: Request) -> str | None:
    return get_request_id() or getattr(request.state, "request_id", None)


def _error_payload(*, code: str, message: str, request_id: str | None) -> dict[str, dict[str, Any]]:
    return {
        "error": {
            "code": code,
            "message": message,
            "request_id": request_id,
        }
    }


def _error_headers(request: Request, request_id: str | None) -> dict[str, str]:
    if request_id is None:
        return {}
    return {request.app.state.settings.request_id_header: request_id}


async def gateway_error_handler(request: Request, exc: GatewayError) -> JSONResponse:
    """Return the stable client representation for an expected gateway error."""
    request_id = _request_id_from(request)
    logger.warning(
        "gateway request failed",
        extra={"error_code": exc.code, "request_id": request_id},
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_payload(code=exc.code, message=exc.message, request_id=request_id),
        headers=_error_headers(request, request_id),
    )


async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Log unexpected exceptions while returning a non-sensitive response."""
    request_id = _request_id_from(request)
    logger.exception(
        "unexpected gateway error",
        exc_info=exc,
        extra={"request_id": request_id},
    )
    return JSONResponse(
        status_code=500,
        content=_error_payload(
            code="internal_error",
            message="Internal gateway error",
            request_id=request_id,
        ),
        headers=_error_headers(request, request_id),
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register gateway-wide exception handlers."""
    app.add_exception_handler(GatewayError, gateway_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, unexpected_error_handler)
