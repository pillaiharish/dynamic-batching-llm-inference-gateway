"""Liveness and readiness endpoints."""

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str


@router.get("/healthz", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Report that the gateway process is alive."""
    return HealthResponse(status="ok")


@router.get(
    "/readyz",
    response_model=HealthResponse,
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": HealthResponse}},
)
async def readiness(request: Request) -> HealthResponse | JSONResponse:
    """Report whether initialization completed and inference is routable."""
    ready = getattr(request.app.state, "ready", False)
    backend_pool = getattr(request.app.state, "backend_pool", None)
    if ready and backend_pool is not None:
        ready = await backend_pool.is_routable()
    if not ready:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not_ready"},
        )
    return HealthResponse(status="ready")
