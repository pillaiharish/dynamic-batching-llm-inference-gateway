from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
import urllib.request
from dataclasses import replace
from pathlib import Path
from unittest import mock

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK_ROOT / "scripts"))

import fake_server  # noqa: E402
import run_matrix  # noqa: E402
import summarize  # noqa: E402
import validate_result  # noqa: E402


class MatrixTests(unittest.TestCase):
    def write_completed_run(
        self,
        output: Path,
        command: list[str],
        expected: run_matrix.RunCoordinate,
        expected_configuration: dict[str, object],
        plan_sha256: str,
        *,
        exit_code: int = 0,
        failed: int = 0,
        valid: bool = True,
        per_request_count: int | None = None,
    ) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        successful = expected.requests - failed
        result = {
            "schema_version": 1,
            "metadata": {
                "run_id": expected.run_id,
                "timestamp_utc": "2026-08-29T00:00:00Z",
                "label": expected.label,
                "repeat": expected.repeat,
                "execution_order": expected.execution_order,
            },
            "configuration": {
                **expected_configuration,
                "timeout_seconds": 5,
                "workload_fingerprint": "workload",
                "gateway_config_fingerprint": "gateway",
                "vllm_config_fingerprint": "vllm",
            },
            "timing": {
                "measured_start_utc": "2026-08-29T00:00:00Z",
                "measured_end_utc": "2026-08-29T00:00:01Z",
                "duration_seconds": 1,
            },
            "warmup": {"attempted": 0, "success": 0, "failed": 0},
            "summary": {
                "attempted": expected.requests,
                "successful": successful,
                "failed": failed,
                "error_rate": failed / expected.requests,
                "errors_by_category": {},
                "e2e": {},
                "ttft": {},
                "request_throughput_rps": 1,
                "output_throughput_tps": 1,
                "authoritative_token_coverage": 1,
            },
            "per_request": [
                {}
                for _ in range(
                    expected.requests if per_request_count is None else per_request_count
                )
            ],
            "metrics": {},
            "metric_artifacts": {},
            "environment": {},
            "validity": {"valid": valid, "reasons": []},
        }
        output.write_text(json.dumps(result), encoding="utf-8")
        (output.parent / "process.json").write_text(
            json.dumps(
                {
                    "exit_code": exit_code,
                    "command": command,
                    "resume_plan_sha256": plan_sha256,
                }
            ),
            encoding="utf-8",
        )

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

    def test_completed_run_requires_exact_successful_coordinate(self) -> None:
        expected = run_matrix.RunCoordinate(
            run_id="matrix-non_streaming-c1-r1-direct",
            label="direct",
            mode="non_streaming",
            concurrency=1,
            repeat=1,
            execution_order=1,
            requests=2,
        )
        command = ["go", "run", "benchmark"]
        expected_configuration = {
            "base_url": "http://127.0.0.1:18000",
            "endpoint": "/v1/chat/completions",
            "model": "test-model",
            "mode": expected.mode,
            "concurrency": expected.concurrency,
            "requests": expected.requests,
            "warmup": 0,
            "dataset_sha256": "a" * 64,
            "temperature": 0,
            "top_p": 1,
            "max_tokens": 8,
            "seed": 1,
            "n": 1,
            "stream": False,
            "stream_include_usage": False,
            "batching_enabled": "not_applicable",
            "prefix_caching": "disabled",
        }
        plan_sha256 = "b" * 64
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "client-result.json"
            self.assertFalse(
                run_matrix.completed_run(
                    output, command, expected, expected_configuration, plan_sha256
                )
            )

            self.write_completed_run(output, command, expected, expected_configuration, plan_sha256)
            self.assertTrue(
                run_matrix.completed_run(
                    output, command, expected, expected_configuration, plan_sha256
                )
            )

            (output.parent / "process.json").unlink()
            self.assertFalse(
                run_matrix.completed_run(
                    output, command, expected, expected_configuration, plan_sha256
                )
            )

            self.write_completed_run(output, command, expected, expected_configuration, plan_sha256)
            process_path = output.parent / "process.json"
            process = json.loads(process_path.read_text(encoding="utf-8"))
            process.pop("resume_plan_sha256")
            process_path.write_text(json.dumps(process), encoding="utf-8")
            self.assertFalse(
                run_matrix.completed_run(
                    output, command, expected, expected_configuration, plan_sha256
                )
            )

            for case, changes in [
                ("failed request", {"failed": 1}),
                ("invalid result", {"valid": False}),
                ("failed process", {"exit_code": 1}),
                ("partial result", {"per_request_count": 1}),
            ]:
                with self.subTest(case=case):
                    self.write_completed_run(
                        output,
                        command,
                        expected,
                        expected_configuration,
                        plan_sha256,
                        **changes,
                    )
                    self.assertFalse(
                        run_matrix.completed_run(
                            output, command, expected, expected_configuration, plan_sha256
                        )
                    )

            self.write_completed_run(output, command, expected, expected_configuration, plan_sha256)
            self.assertFalse(
                run_matrix.completed_run(
                    output, ["different"], expected, expected_configuration, plan_sha256
                )
            )
            self.assertFalse(
                run_matrix.completed_run(
                    output,
                    command,
                    replace(expected, repeat=2),
                    expected_configuration,
                    plan_sha256,
                )
            )
            self.assertFalse(
                run_matrix.completed_run(
                    output,
                    command,
                    expected,
                    {**expected_configuration, "model": "drifted"},
                    plan_sha256,
                )
            )
            self.assertFalse(
                run_matrix.completed_run(
                    output, command, expected, expected_configuration, "c" * 64
                )
            )

            output.write_text("{", encoding="utf-8")
            self.assertFalse(
                run_matrix.completed_run(
                    output, command, expected, expected_configuration, plan_sha256
                )
            )

            self.write_completed_run(output, command, expected, expected_configuration, plan_sha256)
            (output.parent / "process.json").write_text("{", encoding="utf-8")
            self.assertFalse(
                run_matrix.completed_run(
                    output, command, expected, expected_configuration, plan_sha256
                )
            )

    def test_resume_plan_fingerprint_detects_input_content_drift(self) -> None:
        expected = run_matrix.RunCoordinate(
            run_id="matrix-non_streaming-c1-r1-direct",
            label="direct",
            mode="non_streaming",
            concurrency=1,
            repeat=1,
            execution_order=1,
            requests=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset.jsonl"
            dataset.write_text('{"id":"before"}\n', encoding="utf-8")
            config = {"dataset": str(dataset)}
            target = {"label": "direct"}
            before_hashes = run_matrix.referenced_input_hashes(config, target)
            before = run_matrix.resume_plan_sha256(["benchmark"], expected, before_hashes, None)

            dataset.write_text('{"id":"after"}\n', encoding="utf-8")
            after_hashes = run_matrix.referenced_input_hashes(config, target)
            after = run_matrix.resume_plan_sha256(["benchmark"], expected, after_hashes, None)

            self.assertNotEqual(before_hashes, after_hashes)
            self.assertNotEqual(before, after)

    def test_resume_skips_completed_run_without_initial_settling_delay(self) -> None:
        config = {
            "run_id": "resume-test",
            "model": "test-model",
            "dataset": "datasets/sample.jsonl",
            "targets": [
                {
                    "label": "direct",
                    "base_url": "http://127.0.0.1:18000",
                    "batching_enabled": "not_applicable",
                }
            ],
            "modes": ["non_streaming"],
            "concurrency": [1],
            "requests": 1,
            "warmup": 0,
            "repetitions": 2,
            "settle_seconds": 5,
        }
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            target = config["targets"][0]
            output = (
                output_root
                / "resume-test"
                / "non_streaming"
                / "c1"
                / "repeat-1"
                / "direct"
                / "client-result.json"
            )
            run_id = "resume-test-non_streaming-c1-r1-direct"
            command = run_matrix.build_command(
                config, target, "non_streaming", 1, 1, 1, output, run_id
            )
            expected = run_matrix.RunCoordinate(
                run_id=run_id,
                label="direct",
                mode="non_streaming",
                concurrency=1,
                repeat=1,
                execution_order=1,
                requests=1,
            )
            input_hashes = run_matrix.referenced_input_hashes(config, target)
            expected_configuration = run_matrix.planned_result_configuration(
                config, target, "non_streaming", 1, input_hashes
            )
            plan_sha256 = run_matrix.resume_plan_sha256(command, expected, input_hashes, None)
            self.write_completed_run(output, command, expected, expected_configuration, plan_sha256)

            with (
                mock.patch.object(run_matrix, "execute") as execute,
                mock.patch.object(run_matrix.time, "sleep") as sleep,
            ):
                run_matrix.run_matrix(config, output_root, set(), False, resume=True)

            execute.assert_called_once()
            self.assertIn("repeat-2", str(execute.call_args.args[2]))
            sleep.assert_not_called()

    def test_resume_settles_only_between_executed_remaining_runs(self) -> None:
        config = {
            "run_id": "settle-test",
            "model": "test-model",
            "dataset": "datasets/sample.jsonl",
            "targets": [
                {
                    "label": "direct",
                    "base_url": "http://127.0.0.1:18000",
                    "batching_enabled": "not_applicable",
                }
            ],
            "modes": ["non_streaming"],
            "concurrency": [1],
            "requests": 1,
            "warmup": 0,
            "repetitions": 3,
            "settle_seconds": 5,
        }
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            target = config["targets"][0]
            output = (
                output_root
                / "settle-test"
                / "non_streaming"
                / "c1"
                / "repeat-1"
                / "direct"
                / "client-result.json"
            )
            run_id = "settle-test-non_streaming-c1-r1-direct"
            command = run_matrix.build_command(
                config, target, "non_streaming", 1, 1, 1, output, run_id
            )
            expected = run_matrix.RunCoordinate(
                run_id=run_id,
                label="direct",
                mode="non_streaming",
                concurrency=1,
                repeat=1,
                execution_order=1,
                requests=1,
            )
            input_hashes = run_matrix.referenced_input_hashes(config, target)
            expected_configuration = run_matrix.planned_result_configuration(
                config, target, "non_streaming", 1, input_hashes
            )
            plan_sha256 = run_matrix.resume_plan_sha256(command, expected, input_hashes, None)
            self.write_completed_run(output, command, expected, expected_configuration, plan_sha256)

            with (
                mock.patch.object(run_matrix, "execute") as execute,
                mock.patch.object(run_matrix.time, "sleep") as sleep,
            ):
                run_matrix.run_matrix(config, output_root, set(), False, resume=True)

            self.assertEqual(execute.call_count, 2)
            self.assertIn("repeat-2", str(execute.call_args_list[0].args[2]))
            self.assertIn("repeat-3", str(execute.call_args_list[1].args[2]))
            sleep.assert_called_once_with(5.0)


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
        errors = validate_result.validate(result)
        self.assertTrue(any("TTFT" in error for error in errors))
        self.assertIn("missing metrics", errors)
        self.assertIn("configuration missing gateway_config_fingerprint", errors)


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
