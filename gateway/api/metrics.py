"""Prometheus exposition endpoint."""

from fastapi import APIRouter, Request
from prometheus_client import CONTENT_TYPE_LATEST
from starlette.responses import Response

router = APIRouter(tags=["observability"])


@router.get("/metrics", include_in_schema=False)
async def metrics(request: Request) -> Response:
    """Expose this application's isolated Prometheus registry."""
    return Response(
        request.app.state.metrics.render(),
        headers={"Content-Type": CONTENT_TYPE_LATEST},
    )
