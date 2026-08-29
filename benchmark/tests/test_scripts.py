from __future__ import annotations

import json
import sys
import threading
import unittest
import urllib.request
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK_ROOT / "scripts"))

import fake_server  # noqa: E402
import run_matrix  # noqa: E402
import summarize  # noqa: E402
import validate_result  # noqa: E402


class MatrixTests(unittest.TestCase):
    def test_rotation(self) -> None:
        targets = [{"label": "direct"}, {"label": "off"}, {"label": "on"}]
        self.assertEqual(
            [target["label"] for target in run_matrix.rotated(targets, 2)],
            ["off", "on", "direct"],
        )
        self.assertEqual(
            [target["label"] for target in run_matrix.rotated(targets, 3)],
            ["on", "direct", "off"],
        )

    def test_request_guideline(self) -> None:
        self.assertEqual(run_matrix.requests_for({"requests": "auto"}, 1), 200)
        self.assertEqual(run_matrix.requests_for({"requests": "auto"}, 64), 1280)

    def test_batch_admission_validation(self) -> None:
        with self.assertRaises(ValueError):
            run_matrix.validate_admission(
                {
                    "label": "on",
                    "batching_enabled": "true",
                    "batch_max_size": 8,
                    "tenant_max_inflight": 4,
                    "global_max_inflight": 16,
                }
            )


class SummaryTests(unittest.TestCase):
    def test_counter_delta_and_reset(self) -> None:
        before = summarize.parse_prometheus_text('counter_total{kind="x"} 10\n')
        after = summarize.parse_prometheus_text('counter_total{kind="x"} 14\n')
        self.assertEqual(summarize.counter_delta(before, after, "counter_total")["value"], 4)
        reset = summarize.parse_prometheus_text('counter_total{kind="x"} 2\n')
        self.assertTrue(summarize.counter_delta(before, reset, "counter_total")["reset"])

    def test_invalid_comparison_drift(self) -> None:
        def arm(label: str, sha: str, batching: str) -> dict[str, object]:
            return {
                "metadata": {"label": label},
                "validity": {"valid": True},
                "summary": {"attempted": 10},
                "environment": {"gateway_git_sha": sha},
                "configuration": {
                    "dataset_sha256": "a" * 64,
                    "model": "m",
                    "workload_fingerprint": "work",
                    "vllm_config_fingerprint": "vllm",
                    "batching_enabled": batching,
                },
            }

        arms = {
            "direct": arm("direct", "sha", "not_applicable"),
            "gateway_no_batch": arm("gateway_no_batch", "sha-one", "false"),
            "gateway_batch": arm("gateway_batch", "sha-two", "true"),
        }
        self.assertIn(
            "gateway OFF/ON Git SHA is missing or different",
            summarize.comparison_errors(arms),
        )


class ValidationTests(unittest.TestCase):
    def test_validator_rejects_non_streaming_ttft(self) -> None:
        result = {
            "schema_version": 1,
            "metadata": {},
            "configuration": {
                "model": "m",
                "mode": "non_streaming",
                "concurrency": 1,
                "requests": 1,
                "warmup": 0,
                "dataset_sha256": "a" * 64,
                "workload_fingerprint": "w",
                "vllm_config_fingerprint": "v",
            },
            "timing": {},
            "warmup": {},
            "summary": {"attempted": 1},
            "per_request": [{"ttft_ms": 2}],
            "metric_artifacts": {},
            "environment": {},
            "validity": {},
        }
        self.assertTrue(any("TTFT" in error for error in validate_result.validate(result)))


class FakeServerSmokeTests(unittest.TestCase):
    def test_json_stream_and_metrics(self) -> None:
        fake_server.Handler.generated_tokens = 0
        fake_server.Handler.requests = 0
        server = fake_server.ThreadingHTTPServer(("127.0.0.1", 0), fake_server.Handler)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            payload = json.dumps(
                {"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]}
            ).encode()
            request = urllib.request.Request(
                base + "/v1/chat/completions",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            response = json.loads(urllib.request.urlopen(request, timeout=2).read())
            self.assertEqual(response["usage"]["completion_tokens"], 4)
            metrics = urllib.request.urlopen(base + "/metrics", timeout=2).read().decode()
            self.assertIn("vllm:generation_tokens 4", metrics)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
