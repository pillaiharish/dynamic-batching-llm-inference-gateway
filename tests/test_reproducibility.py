import json
import shutil
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path("benchmark/scripts").resolve()))
sys.path.insert(0, str(Path("deploy").resolve()))
sys.path.insert(0, str(Path("infra/vast").resolve()))

from exact_prompt import build_exact_prompt, templated_token_count, token_count  # noqa: E402
from validate_replacement import validate_replacement  # noqa: E402
from validate_topology import validate  # noqa: E402
from vast_lab import parse_instance_id  # noqa: E402
from vllm_replica import start, status, stop  # noqa: E402


class BatchEncoding(dict):
    @property
    def input_ids(self):
        return self["input_ids"]


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(kwargs)
        repetitions = len(messages[0]["content"]) // 2
        tokens = repetitions + 12
        return BatchEncoding(input_ids=list(range(tokens)), attention_mask=[1] * tokens)


def test_batch_encoding_counts_ids_not_fields():
    encoded = BatchEncoding(input_ids=list(range(128)), attention_mask=[1] * 128)
    assert len(encoded) == 2
    assert token_count(encoded) == 128


@pytest.mark.parametrize("target", [128, 1024])
def test_exact_prompt_generation(target):
    tokenizer = FakeTokenizer()
    template_kwargs = {"enable_thinking": False}
    prompt = build_exact_prompt(
        tokenizer,
        target,
        chat_template_kwargs=template_kwargs,
    )
    measured = templated_token_count(
        tokenizer,
        prompt,
        chat_template_kwargs=template_kwargs,
    )
    assert measured == target
    assert all(call["enable_thinking"] is False for call in tokenizer.calls)


def test_all_topology_files_are_valid():
    for path in sorted(Path("deploy/topologies").glob("*.json")):
        validate(json.loads(path.read_text()))


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"topology": "T2"},
        {
            "topology": "T2",
            "replicas": ["A", "B"],
            "traffic": ["A"],
            "gateway_backends": ["B"],
            "aggregation": False,
        },
    ],
)
def test_topology_validation_fails_closed(config):
    with pytest.raises(ValueError):
        validate(config)


def test_lifecycle_uses_and_stops_process_group(tmp_path):
    start(tmp_path, "A", 18001, [sys.executable, "-c", "import time; time.sleep(30)"])
    assert status(tmp_path, "A")["alive"] is True
    stop(tmp_path, "A", 5)
    for _ in range(20):
        if not (tmp_path / "A.json").exists():
            break
        time.sleep(0.05)
    assert not (tmp_path / "A.json").exists()


def test_lifecycle_refuses_reused_pid_without_signalling(tmp_path, monkeypatch):
    start(tmp_path, "A", 18001, [sys.executable, "-c", "import time; time.sleep(30)"])
    path = tmp_path / "A.json"
    record = json.loads(path.read_text())
    path.write_text(json.dumps({**record, "identity": "wrong"}))
    killpg_called = False

    def unexpected_killpg(*_args):
        nonlocal killpg_called
        killpg_called = True

    monkeypatch.setattr("os.killpg", unexpected_killpg)
    with pytest.raises(RuntimeError, match="stale/reused PID"):
        stop(tmp_path, "A", 5)
    assert killpg_called is False
    assert status(tmp_path, "A")["stale_or_reused"] is True

    monkeypatch.undo()
    path.write_text(json.dumps(record))
    stop(tmp_path, "A", 5)


def test_vast_create_response_is_fail_closed():
    assert parse_instance_id('{"success": true, "new_contract": 123}') == 123
    with pytest.raises(ValueError):
        parse_instance_id('{"success": true}')


@pytest.mark.skipif(not shutil.which("terraform"), reason="Terraform CLI is not installed")
def test_terraform_lab_changes_destroy_before_create():
    validate_replacement(shutil.which("terraform"), Path("infra/vast"))


def test_historical_artifact_manifest_is_deterministic_and_complete():
    path = Path("benchmark/evidence/manifests/historical-artifacts.json")
    artifacts = json.loads(path.read_text())
    filenames = [artifact["filename"] for artifact in artifacts]
    assert filenames == sorted(filenames)
    assert len(filenames) == len(set(filenames)) == 10
    assert all(artifact["verified"] for artifact in artifacts)
    assert all(len(artifact["sha256"]) == 64 for artifact in artifacts)
