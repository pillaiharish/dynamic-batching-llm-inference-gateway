import pytest
from pydantic import ValidationError

from gateway.config import Settings

SETTING_NAMES = (
    "APP_NAME",
    "ENVIRONMENT",
    "LOG_LEVEL",
    "HOST",
    "PORT",
    "REQUEST_ID_HEADER",
    "VLLM_BASE_URL",
    "VLLM_API_KEY",
    "VLLM_CONNECT_TIMEOUT_SECONDS",
    "VLLM_REQUEST_TIMEOUT_SECONDS",
    "MAX_COMPLETION_TOKENS",
    "MAX_CHOICES",
)


@pytest.fixture(autouse=True)
def clean_settings_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in SETTING_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_settings_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.app_name == "dynamic-batching-inference-gateway"
    assert settings.environment == "development"
    assert settings.log_level == "INFO"
    assert settings.host == "0.0.0.0"
    assert settings.port == 8080
    assert settings.request_id_header == "X-Request-ID"
    assert settings.vllm_base_url == "http://localhost:8000"
    assert settings.vllm_api_key is None
    assert settings.vllm_connect_timeout_seconds == 5.0
    assert settings.vllm_request_timeout_seconds == 120.0
    assert settings.max_completion_tokens == 4096
    assert settings.max_choices == 4


def test_environment_overrides_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_NAME", "test-gateway")
    monkeypatch.setenv("LOG_LEVEL", "debug")
    monkeypatch.setenv("PORT", "9090")
    monkeypatch.setenv("VLLM_BASE_URL", "https://vllm.example.test/api/")
    monkeypatch.setenv("VLLM_API_KEY", "backend-secret")
    monkeypatch.setenv("MAX_COMPLETION_TOKENS", "2048")
    monkeypatch.setenv("MAX_CHOICES", "2")

    settings = Settings(_env_file=None)

    assert settings.app_name == "test-gateway"
    assert settings.log_level == "DEBUG"
    assert settings.port == 9090
    assert settings.vllm_base_url == "https://vllm.example.test/api"
    assert settings.vllm_api_key is not None
    assert settings.vllm_api_key.get_secret_value() == "backend-secret"
    assert settings.max_completion_tokens == 2048
    assert settings.max_choices == 2


@pytest.mark.parametrize("invalid_port", ["not-a-number", "0", "65536"])
def test_invalid_port_fails_validation(monkeypatch: pytest.MonkeyPatch, invalid_port: str) -> None:
    monkeypatch.setenv("PORT", invalid_port)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


@pytest.mark.parametrize("invalid_header", ["bad header", "X-Réquest-ID"])
def test_invalid_request_id_header_fails_validation(invalid_header: str) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, request_id_header=invalid_header)


@pytest.mark.parametrize(
    "field,value",
    [
        ("vllm_base_url", "ftp://vllm.example.test"),
        ("vllm_base_url", "not-a-url"),
        ("vllm_connect_timeout_seconds", 0),
        ("vllm_request_timeout_seconds", -1),
        ("max_completion_tokens", 0),
        ("max_choices", 0),
    ],
)
def test_invalid_vllm_settings_fail_validation(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})


def test_empty_vllm_api_key_is_unconfigured() -> None:
    settings = Settings(_env_file=None, vllm_api_key="  ")

    assert settings.vllm_api_key is None
