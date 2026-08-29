#!/usr/bin/env python3
"""Deterministic Chat Completions/SSE/metrics server for local smoke tests only."""

from __future__ import annotations

import argparse
import json
import signal
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    generated_tokens: ClassVar[int] = 0
    requests: ClassVar[int] = 0
    lock: ClassVar[threading.Lock] = threading.Lock()

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self.send_json({"status": "ok"})
            return
        if self.path != "/metrics":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        with type(self).lock:
            tokens = type(self).generated_tokens
            requests = type(self).requests
        body = (
            "# TYPE vllm:generation_tokens counter\n"
            f"vllm:generation_tokens {tokens}\n"
            "# TYPE vllm:request_success counter\n"
            f'vllm:request_success{{finished_reason="stop"}} {requests}\n'
            "# TYPE vllm:num_requests_running gauge\n"
            "vllm:num_requests_running 0\n"
            "# TYPE vllm:num_requests_waiting gauge\n"
            "vllm:num_requests_waiting 0\n"
            "# TYPE vllm:kv_cache_usage_perc gauge\n"
            "vllm:kv_cache_usage_perc 0\n"
        ).encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
        except (ValueError, json.JSONDecodeError):
            self.send_json({"error": {"code": "invalid_json"}}, HTTPStatus.BAD_REQUEST)
            return
        with type(self).lock:
            type(self).generated_tokens += 4
            type(self).requests += 1
        if payload.get("stream"):
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            events = [
                ': keepalive\n\ndata: {"choices":[{"delta":{"role":"assistant"}}]}\n\n',
                'data: {"choices":[{"delta":{"content":"deterministic"}}]}\n\n',
                'data: {"choices":[],"usage":{"completion_tokens":4}}\n\n',
                "data: [DONE]\n\n",
            ]
            for event in events:
                self.wfile.write(event.encode())
                self.wfile.flush()
                time.sleep(0.001)
            self.close_connection = True
            return
        self.send_json(
            {
                "id": "fake",
                "object": "chat.completion",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "deterministic"}}
                ],
                "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
            }
        )

    def send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--ready-file", type=Path)
    arguments = parser.parse_args()
    server = ThreadingHTTPServer((arguments.host, arguments.port), Handler)
    if arguments.ready_file:
        arguments.ready_file.write_text(str(server.server_port), encoding="utf-8")

    def stop(_signum: int, _frame: object) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        if arguments.ready_file:
            arguments.ready_file.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
