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


def test_environment_overrides_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_NAME", "test-gateway")
    monkeypatch.setenv("LOG_LEVEL", "debug")
    monkeypatch.setenv("PORT", "9090")

    settings = Settings(_env_file=None)

    assert settings.app_name == "test-gateway"
    assert settings.log_level == "DEBUG"
    assert settings.port == 9090


@pytest.mark.parametrize("invalid_port", ["not-a-number", "0", "65536"])
def test_invalid_port_fails_validation(monkeypatch: pytest.MonkeyPatch, invalid_port: str) -> None:
    monkeypatch.setenv("PORT", invalid_port)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


@pytest.mark.parametrize("invalid_header", ["bad header", "X-Réquest-ID"])
def test_invalid_request_id_header_fails_validation(invalid_header: str) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, request_id_header=invalid_header)
