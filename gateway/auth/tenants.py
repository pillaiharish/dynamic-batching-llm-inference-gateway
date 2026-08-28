"""Configured tenant validation and bearer-token authentication."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from secrets import compare_digest
from typing import cast

from fastapi import Request
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from gateway.core.errors import UnauthorizedError


class TenantConfig(BaseModel):
    """Secret-bearing configuration for one tenant."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    api_key: SecretStr
    max_inflight: int = Field(gt=0)
    max_queue: int = Field(ge=0)

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, value: SecretStr) -> SecretStr:
        api_key = value.get_secret_value()
        if (
            not api_key
            or not api_key.isascii()
            or any(character.isspace() for character in api_key)
        ):
            raise ValueError("tenant API key must be a non-empty ASCII bearer token")
        return value


@dataclass(frozen=True, slots=True)
class TenantContext:
    """Safe request-scoped tenant identity without credential material."""

    tenant_id: str
    max_inflight: int
    max_queue: int

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValueError("tenant ID must not be blank")
        if self.max_inflight <= 0 or self.max_queue < 0:
            raise ValueError("tenant admission limits are invalid")


class TenantRegistry:
    """Resolve configured bearer credentials to safe tenant contexts."""

    def __init__(self, tenants: Mapping[str, TenantConfig]) -> None:
        self._entries = tuple(
            (
                config.api_key.get_secret_value(),
                TenantContext(
                    tenant_id=tenant_id,
                    max_inflight=config.max_inflight,
                    max_queue=config.max_queue,
                ),
            )
            for tenant_id, config in tenants.items()
        )

    @property
    def tenants(self) -> tuple[TenantContext, ...]:
        """Return safe configured identities for admission initialization."""
        return tuple(context for _api_key, context in self._entries)

    def authenticate(self, authorization: str | None) -> TenantContext:
        """Authenticate an exact bearer header without revealing failure details."""
        token = self._extract_bearer_token(authorization)
        matched: TenantContext | None = None
        for configured_key, tenant in self._entries:
            if compare_digest(token, configured_key):
                matched = tenant
        if matched is None:
            raise UnauthorizedError()
        return matched

    @staticmethod
    def _extract_bearer_token(authorization: str | None) -> str:
        if authorization is None:
            raise UnauthorizedError()
        parts = authorization.split(" ")
        if (
            len(parts) != 2
            or parts[0].casefold() != "bearer"
            or not parts[1]
            or not parts[1].isascii()
            or any(character.isspace() for character in parts[1])
        ):
            raise UnauthorizedError()
        return parts[1]


def authenticate_tenant(request: Request) -> TenantContext:
    """FastAPI dependency that records a safe tenant identity on the request."""
    registry = cast(TenantRegistry, request.app.state.tenant_registry)
    tenant = registry.authenticate(request.headers.get("Authorization"))
    request.state.tenant = tenant
    return tenant
