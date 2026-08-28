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
    "BACKENDS_JSON",
    "VLLM_BASE_URL",
    "VLLM_API_KEY",
    "VLLM_CONNECT_TIMEOUT_SECONDS",
    "VLLM_REQUEST_TIMEOUT_SECONDS",
    "BACKEND_HEALTH_INTERVAL_SECONDS",
    "BACKEND_HEALTH_TIMEOUT_SECONDS",
    "MAX_COMPLETION_TOKENS",
    "MAX_CHOICES",
    "TENANTS_JSON",
    "GLOBAL_MAX_INFLIGHT",
    "GLOBAL_MAX_QUEUE",
    "ADMISSION_QUEUE_TIMEOUT_SECONDS",
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
    assert set(settings.backends_json) == {"default"}
    assert settings.backends_json["default"].base_url == "http://localhost:8000"
    assert settings.backends_json["default"].api_key is None
    assert settings.vllm_connect_timeout_seconds == 5.0
    assert settings.vllm_request_timeout_seconds == 120.0
    assert settings.backend_health_interval_seconds == 5.0
    assert settings.backend_health_timeout_seconds == 2.0
    assert settings.max_completion_tokens == 4096
    assert settings.max_choices == 4
    assert settings.tenants_json == {}
    assert settings.global_max_inflight == 16
    assert settings.global_max_queue == 64
    assert settings.admission_queue_timeout_seconds == 5.0


def test_environment_overrides_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_NAME", "test-gateway")
    monkeypatch.setenv("LOG_LEVEL", "debug")
    monkeypatch.setenv("PORT", "9090")
    monkeypatch.setenv(
        "BACKENDS_JSON",
        '{"gpu-a":{"base_url":"https://vllm-a.example.test/api/",'
        '"api_key":"backend-secret"},'
        '"gpu-b":{"base_url":"http://vllm-b.example.test:8000"}}',
    )
    monkeypatch.setenv("BACKEND_HEALTH_INTERVAL_SECONDS", "3")
    monkeypatch.setenv("BACKEND_HEALTH_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv("MAX_COMPLETION_TOKENS", "2048")
    monkeypatch.setenv("MAX_CHOICES", "2")
    monkeypatch.setenv(
        "TENANTS_JSON",
        '{"tenant-a":{"api_key":"tenant-a-key","max_inflight":2,"max_queue":3}}',
    )
    monkeypatch.setenv("GLOBAL_MAX_INFLIGHT", "8")
    monkeypatch.setenv("GLOBAL_MAX_QUEUE", "12")
    monkeypatch.setenv("ADMISSION_QUEUE_TIMEOUT_SECONDS", "1.5")

    settings = Settings(_env_file=None)

    assert settings.app_name == "test-gateway"
    assert settings.log_level == "DEBUG"
    assert settings.port == 9090
    assert set(settings.backends_json) == {"gpu-a", "gpu-b"}
    assert settings.backends_json["gpu-a"].base_url == "https://vllm-a.example.test/api"
    assert settings.backends_json["gpu-a"].api_key is not None
    assert settings.backends_json["gpu-a"].api_key.get_secret_value() == "backend-secret"
    assert settings.backends_json["gpu-b"].api_key is None
    assert settings.backend_health_interval_seconds == 3.0
    assert settings.backend_health_timeout_seconds == 1.0
    assert settings.max_completion_tokens == 2048
    assert settings.max_choices == 2
    assert set(settings.tenants_json) == {"tenant-a"}
    assert settings.tenants_json["tenant-a"].max_inflight == 2
    assert settings.tenants_json["tenant-a"].api_key.get_secret_value() == "tenant-a-key"
    assert settings.global_max_inflight == 8
    assert settings.global_max_queue == 12
    assert settings.admission_queue_timeout_seconds == 1.5


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
        ("vllm_connect_timeout_seconds", 0),
        ("vllm_request_timeout_seconds", -1),
        ("backend_health_interval_seconds", 0),
        ("backend_health_timeout_seconds", -1),
        ("max_completion_tokens", 0),
        ("max_choices", 0),
    ],
)
def test_invalid_vllm_settings_fail_validation(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})


def test_empty_backend_api_key_is_unconfigured() -> None:
    settings = Settings(
        _env_file=None,
        backends_json={"gpu-a": {"base_url": "http://vllm-a:8000", "api_key": "  "}},
    )

    assert settings.backends_json["gpu-a"].api_key is None


@pytest.mark.parametrize(
    "backends",
    [
        {},
        {" ": {"base_url": "http://vllm:8000"}},
        {"gpu-a": {"base_url": "ftp://vllm.example.test"}},
        {"gpu-a": {"base_url": "not-a-url"}},
        {"gpu-a": {"base_url": "http://vllm:8000", "unknown": "value"}},
    ],
)
def test_invalid_backend_config_fails_validation(backends: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, backends_json=backends)


def test_backend_secrets_are_redacted() -> None:
    settings = Settings(
        _env_file=None,
        backends_json={"gpu-a": {"base_url": "http://vllm-a:8000", "api_key": "backend-secret"}},
    )

    assert "backend-secret" not in repr(settings)


def test_legacy_single_backend_environment_does_not_create_parallel_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_BASE_URL", "https://legacy-vllm.example.test")
    monkeypatch.setenv("VLLM_API_KEY", "legacy-secret")

    settings = Settings(_env_file=None)

    assert set(settings.backends_json) == {"default"}
    assert settings.backends_json["default"].base_url == "http://localhost:8000"
    assert "legacy-secret" not in repr(settings)


def test_valid_multi_tenant_config() -> None:
    settings = Settings(
        _env_file=None,
        tenants_json={
            "tenant-a": {"api_key": "tenant-a-key", "max_inflight": 2, "max_queue": 4},
            "tenant-b": {"api_key": "tenant-b-key", "max_inflight": 1, "max_queue": 0},
        },
    )

    assert settings.tenants_json["tenant-a"].max_queue == 4
    assert settings.tenants_json["tenant-b"].max_inflight == 1
    assert "tenant-a-key" not in repr(settings)


@pytest.mark.parametrize(
    "tenants",
    [
        {"tenant-a": {"api_key": "", "max_inflight": 1, "max_queue": 1}},
        {"tenant-a": {"api_key": "   ", "max_inflight": 1, "max_queue": 1}},
        {"tenant-a": {"api_key": "non-ascii-é", "max_inflight": 1, "max_queue": 1}},
        {"tenant-a": {"api_key": "key", "max_inflight": 0, "max_queue": 1}},
        {"tenant-a": {"api_key": "key", "max_inflight": 1, "max_queue": -1}},
        {" ": {"api_key": "key", "max_inflight": 1, "max_queue": 1}},
    ],
)
def test_invalid_tenant_config_fails_validation(tenants: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, tenants_json=tenants)


def test_duplicate_tenant_api_keys_fail_validation() -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            tenants_json={
                "tenant-a": {"api_key": "shared-key", "max_inflight": 1, "max_queue": 1},
                "tenant-b": {"api_key": "shared-key", "max_inflight": 1, "max_queue": 1},
            },
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("global_max_inflight", 0),
        ("global_max_queue", -1),
        ("admission_queue_timeout_seconds", 0),
    ],
)
def test_invalid_global_admission_config_fails_validation(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})
