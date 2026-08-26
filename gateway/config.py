"""Typed application configuration."""

from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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
