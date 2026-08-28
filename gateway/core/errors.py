"""Gateway exception types and HTTP error handlers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError
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
    response_headers: dict[str, str] = {}

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


class InvalidRequestError(GatewayError):
    """Raised when a request violates a gateway-level API constraint."""

    code = "invalid_request"
    default_message = "Invalid chat completion request"
    status_code = 400


class BackendUnavailableError(BackendError):
    """Raised when the configured inference backend cannot be reached."""

    code = "backend_unavailable"
    default_message = "Inference backend is unavailable"
    status_code = 503


class BackendTimeoutError(BackendError):
    """Raised when an inference backend operation exceeds its timeout."""

    code = "backend_timeout"
    default_message = "Inference backend timed out"
    status_code = 504


class BackendProtocolError(BackendError):
    """Raised when the backend returns an invalid success response."""

    code = "backend_protocol_error"
    default_message = "Inference backend returned an invalid response"


class BackendHTTPError(BackendError):
    """Raised for a safe-to-normalize upstream HTTP failure."""

    code = "backend_http_error"
    default_message = "Inference backend request failed"


class BackendConfigurationError(BackendError):
    """Raised when backend authentication or configuration is rejected."""

    code = "backend_configuration_error"
    default_message = "Inference backend configuration was rejected"


class BackendRequestRejectedError(BackendError):
    """Raised when vLLM safely rejects client-controlled request parameters."""

    code = "backend_request_rejected"
    default_message = "Inference backend rejected the request"
    status_code = 400


class BackendCapabilityError(BackendError):
    """Raised when a backend operation is not implemented in this version."""

    code = "backend_capability_unsupported"
    default_message = "Inference backend capability is not supported"
    status_code = 501


class UnauthorizedError(GatewayError):
    """Raised when tenant bearer credentials are absent or invalid."""

    code = "unauthorized"
    default_message = "Invalid or missing tenant credentials"
    status_code = 401
    response_headers = {"WWW-Authenticate": "Bearer"}


class TenantQueueFullError(GatewayError):
    """Raised when one tenant has filled its bounded waiting queue."""

    code = "tenant_queue_full"
    default_message = "Tenant admission queue is full"
    status_code = 429


class GatewayQueueFullError(GatewayError):
    """Raised when the process-wide waiting queue bound is reached."""

    code = "gateway_queue_full"
    default_message = "Gateway admission queue is full"
    status_code = 429


class AdmissionTimeoutError(GatewayError):
    """Raised when a queued request exceeds its bounded wait time."""

    code = "admission_timeout"
    default_message = "Timed out waiting for inference admission"
    status_code = 429


class AdmissionUnavailableError(GatewayError):
    """Raised when admission has stopped during application shutdown."""

    code = "admission_unavailable"
    default_message = "Inference admission is unavailable"
    status_code = 503


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


def _error_response(
    request: Request,
    *,
    status_code: int,
    code: str,
    message: str,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    request_id = _request_id_from(request)
    response_headers = _error_headers(request, request_id)
    response_headers.update(headers or {})
    return JSONResponse(
        status_code=status_code,
        content=_error_payload(code=code, message=message, request_id=request_id),
        headers=response_headers,
    )


async def gateway_error_handler(request: Request, exc: GatewayError) -> JSONResponse:
    """Return the stable client representation for an expected gateway error."""
    request_id = _request_id_from(request)
    logger.warning(
        "gateway request failed",
        extra={"error_code": exc.code, "request_id": request_id},
    )
    return _error_response(
        request,
        status_code=exc.status_code,
        code=exc.code,
        message=exc.message,
        headers=exc.response_headers,
    )


async def request_validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Normalize malformed JSON and schema failures without exposing internals."""
    message = "Invalid chat completion request"
    if any(
        error.get("loc") == ("body", "stream") and error.get("input") is True
        for error in exc.errors()
    ):
        message = "Streaming chat completions are not supported"

    logger.info(
        "request validation failed",
        extra={"error_code": "invalid_request", "request_id": _request_id_from(request)},
    )
    return _error_response(
        request,
        status_code=400,
        code="invalid_request",
        message=message,
    )


async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Log unexpected exceptions while returning a non-sensitive response."""
    request_id = _request_id_from(request)
    logger.exception(
        "unexpected gateway error",
        exc_info=exc,
        extra={"request_id": request_id},
    )
    return _error_response(
        request,
        status_code=500,
        code="internal_error",
        message="Internal gateway error",
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register gateway-wide exception handlers."""
    app.add_exception_handler(  # type: ignore[arg-type]
        RequestValidationError,
        request_validation_error_handler,
    )
    app.add_exception_handler(GatewayError, gateway_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, unexpected_error_handler)
