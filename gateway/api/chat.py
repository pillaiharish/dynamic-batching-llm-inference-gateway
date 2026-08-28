"""OpenAI-compatible Chat Completions endpoint."""

from typing import Annotated, Any, cast

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from gateway.admission.controller import AdmissionController
from gateway.auth.tenants import TenantContext, authenticate_tenant
from gateway.backends.base import InferenceBackend
from gateway.config import Settings
from gateway.core.errors import InvalidRequestError
from gateway.schemas.chat import ChatCompletionRequest

router = APIRouter(prefix="/v1", tags=["chat-completions"])


def _enforce_configured_limits(payload: ChatCompletionRequest, settings: Settings) -> None:
    if payload.max_tokens is not None and payload.max_tokens > settings.max_completion_tokens:
        raise InvalidRequestError(f"max_tokens must not exceed {settings.max_completion_tokens}")
    if payload.n > settings.max_choices:
        raise InvalidRequestError(f"n must not exceed {settings.max_choices}")


@router.post("/chat/completions")
async def create_chat_completion(
    payload: ChatCompletionRequest,
    request: Request,
    tenant: Annotated[TenantContext, Depends(authenticate_tenant)],
) -> JSONResponse:
    """Validate and forward one non-streaming chat completion request."""
    settings = cast(Settings, request.app.state.settings)
    backend = cast(InferenceBackend, request.app.state.backend)
    admission = cast(AdmissionController, request.app.state.admission_controller)
    _enforce_configured_limits(payload, settings)
    async with admission.admit(tenant) as lease:
        request.state.admission_result = "queued" if lease.was_queued else "admitted"
        response = cast(dict[str, Any], await backend.generate(payload))
    return JSONResponse(content=response)
