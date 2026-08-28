"""FastAPI application construction."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from gateway import __version__
from gateway.admission.controller import AdmissionController
from gateway.api.chat import router as chat_router
from gateway.api.health import router as health_router
from gateway.auth.tenants import TenantRegistry
from gateway.backends.base import InferenceBackend
from gateway.backends.vllm import VLLMBackend
from gateway.config import Settings
from gateway.core.errors import register_exception_handlers
from gateway.core.logging import configure_logging
from gateway.core.request_id import install_request_id_middleware
from gateway.routing.pool import BackendPool


def create_app(
    settings: Settings | None = None,
    *,
    backend: InferenceBackend | None = None,
    tenant_registry: TenantRegistry | None = None,
    admission_controller: AdmissionController | None = None,
) -> FastAPI:
    """Build and configure an isolated gateway application instance."""
    app_settings = settings or Settings()
    configure_logging(app_settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        active_registry = (
            tenant_registry
            if tenant_registry is not None
            else TenantRegistry(app_settings.tenants_json)
        )
        active_admission = (
            admission_controller
            if admission_controller is not None
            else AdmissionController(
                active_registry.tenants,
                global_max_inflight=app_settings.global_max_inflight,
                global_max_queue=app_settings.global_max_queue,
                queue_timeout_seconds=app_settings.admission_queue_timeout_seconds,
            )
        )
        if backend is None:
            leaf_backends = {
                backend_id: VLLMBackend(
                    backend_id,
                    backend_config,
                    connect_timeout_seconds=app_settings.vllm_connect_timeout_seconds,
                    request_timeout_seconds=app_settings.vllm_request_timeout_seconds,
                    health_timeout_seconds=app_settings.backend_health_timeout_seconds,
                )
                for backend_id, backend_config in app_settings.backends_json.items()
            }
            active_backend: InferenceBackend = BackendPool(
                leaf_backends,
                health_interval_seconds=app_settings.backend_health_interval_seconds,
                health_timeout_seconds=app_settings.backend_health_timeout_seconds,
            )
        else:
            active_backend = backend

        active_pool = active_backend if isinstance(active_backend, BackendPool) else None
        app.state.tenant_registry = active_registry
        app.state.admission_controller = active_admission
        app.state.backend = active_backend
        app.state.backend_pool = active_pool
        try:
            if active_pool is not None:
                await active_pool.start()
            app.state.ready = True
            yield
        finally:
            app.state.ready = False
            try:
                await active_admission.shutdown()
            finally:
                await active_backend.close()

    application = FastAPI(title=app_settings.app_name, version=__version__, lifespan=lifespan)
    application.state.settings = app_settings
    application.state.ready = False
    application.state.backend_pool = None

    install_request_id_middleware(
        application,
        header_name=app_settings.request_id_header,
    )
    register_exception_handlers(application)
    application.include_router(health_router)
    application.include_router(chat_router)
    return application


app = create_app()
