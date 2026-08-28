"""Typed application configuration."""

from __future__ import annotations

from typing import Literal

from pydantic import AnyHttpUrl, Field, SecretStr, TypeAdapter, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from gateway.auth.tenants import TenantConfig

_http_url_adapter = TypeAdapter(AnyHttpUrl)


class Settings(BaseSettings):
    """Gateway settings loaded from environment variables and an optional .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = Field(default="dynamic-batching-inference-gateway", min_length=1)
    environment: str = Field(default="development", min_length=1)
    log_level: Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"] = "INFO"
    host: str = Field(default="0.0.0.0", min_length=1)
    port: int = Field(default=8080, ge=1, le=65535)
    request_id_header: str = Field(default="X-Request-ID", min_length=1)
    vllm_base_url: str = "http://localhost:8000"
    vllm_api_key: SecretStr | None = None
    vllm_connect_timeout_seconds: float = Field(default=5.0, gt=0)
    vllm_request_timeout_seconds: float = Field(default=120.0, gt=0)
    max_completion_tokens: int = Field(default=4096, gt=0)
    max_choices: int = Field(default=4, gt=0)
    tenants_json: dict[str, TenantConfig] = Field(default_factory=dict)
    global_max_inflight: int = Field(default=16, gt=0)
    global_max_queue: int = Field(default=64, ge=0)
    admission_queue_timeout_seconds: float = Field(default=5.0, gt=0)

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, value: object) -> object:
        """Allow conventional case-insensitive log-level values."""
        return value.upper() if isinstance(value, str) else value

    @field_validator("request_id_header")
    @classmethod
    def validate_request_id_header(cls, value: str) -> str:
        """Reject values that cannot be safely used as an HTTP header name."""
        allowed = frozenset("!#$%&'*+-.^_`|~")
        if not all(
            character.isascii() and (character.isalnum() or character in allowed)
            for character in value
        ):
            raise ValueError("must be a valid HTTP header name")
        return value

    @field_validator("vllm_base_url")
    @classmethod
    def validate_vllm_base_url(cls, value: str) -> str:
        """Validate an HTTP URL and normalize trailing slashes."""
        validated_url = _http_url_adapter.validate_python(value)
        return str(validated_url).rstrip("/")

    @field_validator("vllm_api_key", mode="before")
    @classmethod
    def normalize_empty_vllm_api_key(cls, value: object) -> object:
        """Treat an empty environment value as an unconfigured API key."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def validate_tenant_mapping(self) -> Settings:
        """Require safe tenant IDs and a one-to-one API-key mapping."""
        seen_api_keys: set[str] = set()
        for tenant_id, tenant in self.tenants_json.items():
            if not tenant_id.strip():
                raise ValueError("tenant IDs must not be blank")
            api_key = tenant.api_key.get_secret_value()
            if api_key in seen_api_keys:
                raise ValueError("tenant API keys must be unique")
            seen_api_keys.add(api_key)
        return self
