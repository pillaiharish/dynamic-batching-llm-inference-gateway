"""Supported subset of the OpenAI Chat Completions request contract."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatMessage(BaseModel):
    """A simple text-only chat message."""

    model_config = ConfigDict(extra="forbid", strict=True)

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def reject_blank_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message content must not be blank")
        return value


class ChatCompletionRequest(BaseModel):
    """The explicitly supported Chat Completions request fields."""

    model_config = ConfigDict(extra="forbid", strict=True)

    model: str = Field(min_length=1)
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, gt=0)
    stop: str | list[str] | None = None
    seed: int | None = None
    n: int = Field(default=1, gt=0)
    stream: bool = False

    @field_validator("model")
    @classmethod
    def reject_blank_model(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model must not be blank")
        return value

    @field_validator("stop")
    @classmethod
    def validate_stop(cls, value: str | list[str] | None) -> str | list[str] | None:
        if value is None:
            return None
        values = [value] if isinstance(value, str) else value
        if not values or any(not item.strip() for item in values):
            raise ValueError("stop values must be non-empty strings")
        return value

    def to_upstream_payload(self) -> dict[str, object]:
        """Serialize only fields supplied by the client and required contract fields."""
        return self.model_dump(mode="json", exclude_none=True, exclude_unset=True)
