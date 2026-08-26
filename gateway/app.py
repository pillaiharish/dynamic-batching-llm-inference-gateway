"""FastAPI application construction."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from gateway.api.health import router as health_router
from gateway.config import Settings
from gateway.core.errors import register_exception_handlers
from gateway.core.logging import configure_logging
from gateway.core.request_id import install_request_id_middleware


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and configure an isolated gateway application instance."""
    app_settings = settings or Settings()
    configure_logging(app_settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.ready = True
        try:
            yield
        finally:
            app.state.ready = False

    application = FastAPI(title=app_settings.app_name, version="0.1.0", lifespan=lifespan)
    application.state.settings = app_settings
    application.state.ready = False

    install_request_id_middleware(
        application,
        header_name=app_settings.request_id_header,
    )
    register_exception_handlers(application)
    application.include_router(health_router)
    return application


app = create_app()
