"""OpenAI-compatible Chat Completions endpoint."""

from typing import Annotated, Any, cast

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response

from gateway.admission.controller import AdmissionController, AdmissionLease
from gateway.auth.tenants import TenantContext, authenticate_tenant
from gateway.backends.base import BackendStream, InferenceBackend
from gateway.config import Settings
from gateway.core.errors import InvalidRequestError
from gateway.schemas.chat import ChatCompletionRequest
from gateway.streaming.relay import SSEStreamingResponse, StreamingRelay

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
) -> Response:
    """Validate and forward one JSON or streaming chat completion request."""
    settings = cast(Settings, request.app.state.settings)
    backend = cast(InferenceBackend, request.app.state.backend)
    admission = cast(AdmissionController, request.app.state.admission_controller)
    _enforce_configured_limits(payload, settings)

    if payload.stream:
        return await _create_streaming_completion(
            payload=payload,
            request=request,
            tenant=tenant,
            backend=backend,
            admission=admission,
        )

    async with admission.admit(tenant) as lease:
        request.state.admission_result = "queued" if lease.was_queued else "admitted"
        response = cast(dict[str, Any], await backend.generate(payload))
    return JSONResponse(content=response)


async def _create_streaming_completion(
    *,
    payload: ChatCompletionRequest,
    request: Request,
    tenant: TenantContext,
    backend: InferenceBackend,
    admission: AdmissionController,
) -> SSEStreamingResponse:
    lease: AdmissionLease = await admission.acquire(tenant)
    request.state.admission_result = "queued" if lease.was_queued else "admitted"
    request.state.streaming = True
    backend_stream: BackendStream | None = None
    try:
        backend_stream = await backend.stream(payload)
        relay = StreamingRelay(
            backend_stream,
            lease,
            tenant_id=tenant.tenant_id,
            request_id=getattr(request.state, "request_id", None),
        )
        return SSEStreamingResponse(relay)
    except BaseException:
        if backend_stream is not None:
            await backend_stream.aclose()
        await lease.release()
        raise
