from importlib.metadata import version

from gateway import __version__
from gateway.app import create_app
from gateway.config import Settings


def test_gateway_version_is_consistent() -> None:
    app = create_app(Settings(_env_file=None))

    assert __version__ == "0.2.0"
    assert app.version == __version__
    assert version("dynamic-batching-llm-inference-gateway") == __version__
