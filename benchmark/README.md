# Reproducible benchmark and evidence suite

This directory contains a Go benchmark harness for vLLM-compatible Chat Completions endpoints,
matrix orchestration, raw Prometheus/GPU sampling, result validation, summaries, and plotting. It is
separate from gateway runtime code and does not edit or tune that code.

> **vLLM already performs continuous batching internally.** vLLM continuously schedules requests
> at the engine level even when each HTTP request arrives separately. Gateway dynamic batching is
> an additional HTTP-level aggregation layer and may be redundant for some workloads.

This suite does not assume that gateway aggregation is beneficial. A workload can show that it
helps, hurts, or is neutral. A negative result is valid evidence. At concurrency one, an enabled
gateway batcher normally adds its timeout wait because no compatible peer arrives. At moderate or
high concurrency, native vLLM scheduling may already capture most available batching benefit.

Curated real-GPU evidence is available for one
[H100 Qwen3.8-27B A/B/C sweep](evidence/h100-qwen38-27b-20260829/README.md). Its conclusions are
limited to the recorded non-streaming workload and deployment. The fake-server tests still prove
protocol and measurement behavior only; they are not synthetic performance evidence.

## Scientific question and three-arm design

The principal question is:

> What incremental latency and throughput effect does this gateway introduce, and what additional
> effect does gateway-side `/v1/chat/completions/batch` aggregation have on top of vLLM's own
> continuous batching?

Every primary non-streaming comparison contains all three arms under otherwise identical
conditions:

```text
A. direct vLLM
        │
        ├── B - A = gateway overhead
        │
B. gateway → vLLM, dynamic batching OFF
        │
        ├── C - B = gateway batching effect
        │
C. gateway → vLLM, dynamic batching ON
```

Comparing only A with C is invalid because it conflates gateway overhead with batching behavior.
The same harness, ordered dataset, generation parameters, concurrency algorithm, HTTP transport,
and endpoint semantics target each arm. `direct`, `gateway_no_batch`, and `gateway_batch` are labels
and configuration records only; no label selects a different request generator.

For one A/B/C comparison block, keep the same vLLM process and configuration, model weights, GPU,
precision, and scheduler settings. Use the same gateway Git SHA for B and C. Change only
`DYNAMIC_BATCHING_ENABLED` plus explicitly recorded batch size/wait settings. Do not generate
competing traffic against another arm at the same time.

## Directory map

```text
benchmark/
├── cmd/gateway-bench/       Go CLI
├── internal/                client, runner, stats, metrics, results, workload, samplers
├── configs/                 smoke, concurrency, policy, and server config examples
├── datasets/sample.jsonl    varied deterministic corpus
├── scripts/                 matrix, fake server, host sampler, validator, summary, plots
├── schema/result.schema.json
├── tests/                   standard-library helper tests
└── evidence/                deliberately curated evidence only
```

`benchmark/results/` is ignored by Git. Every repetition has its own directory and raw result; the
runner never concatenates repetitions into a single opaque run.

## Go CLI

From `benchmark/`:

```bash
go run ./cmd/gateway-bench \
  --base-url http://127.0.0.1:8080 \
  --endpoint /v1/chat/completions \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset datasets/sample.jsonl \
  --mode non_streaming \
  --concurrency 16 \
  --requests 320 \
  --warmup 20 \
  --timeout 120s \
  --label gateway_batch \
  --batching-enabled true \
  --batch-max-size 8 \
  --batch-max-wait-seconds 0.005 \
  --tenant-max-inflight 128 \
  --global-max-inflight 128 \
  --gateway-metrics-url http://127.0.0.1:8080/metrics \
  --vllm-metrics-url http://127.0.0.1:8000/metrics \
  --output-token-counter gateway_observed_output_tokens_total \
  --sample-interval 500ms \
  --gpu-sample-interval 500ms \
  --prefix-caching disabled \
  --vllm-version 0.x.y \
  --vllm-config configs/vllm-reference.json \
  --gateway-config configs/gateway-on.json \
  --output results/manual/client-result.json
```

Set a target credential only in the environment:

```bash
export BENCH_AUTH_TOKEN='replace-at-runtime'
```

The token never enters CLI arguments, results, manifests, CSV, or process metadata. Base URLs with
embedded user information are rejected. The client does not store prompts, authorization headers,
generated text, or unsafe backend error details. It stores the corpus request ID and a bounded safe
`error.code` when present.

Required flags are `--base-url`, `--model`, `--dataset`, `--label`, and `--output`. Run
`go run ./cmd/gateway-bench --help` for all metadata, metric, and sampler flags.

### Workload and request construction

The JSONL input remains in file order and every run records the SHA-256 of the exact input bytes.
Each record has a unique ID and non-empty text messages. Requests cycle through the same ordered
corpus if the request count is larger than the dataset. Primary defaults are:

```text
temperature=0
top_p=1
max_tokens=128
seed=1
n=1
streaming: stream=true and stream_options.include_usage=true
non-streaming batching: stream=false
```

Different prompts reduce domination by one repeated prefix. They do not guarantee identical output
length; authoritative token counters make that variation visible.

### Closed-loop concurrency and timing boundaries

The custom harness is a closed-loop client, not an open-loop RPS generator:

```text
start at most C worker requests
as one finishes, that worker starts its next request
stop after N measured attempts finish
```

The worker pool makes it impossible to exceed C outstanding requests. `runner_test.go` blocks a
fake server and proves that configured concurrency four both reaches four and never exceeds four.

Measured start is the monotonic timestamp taken immediately before workers can launch the first
measured request. Measured end is immediately after the last measured request completes. Throughput
uses this wall duration. Connection establishment happens naturally through the same reusable
`http.Client` and `Transport`; the pool has explicit idle-connection, per-host, TLS, and request
timeouts sized for concurrency.

Warmup is a completely separate phase using the same client and concurrency. Its attempted,
successful, and failed counts are retained, but no warmup request result or latency enters measured
percentiles or throughput. A deterministic test makes warmups slow and measured responses fast to
prove exclusion.

### Client metrics and exact definitions

Each measured request records start offset, E2E duration, HTTP status, success, safe error category
and code, response byte count, and authoritative completion tokens when available. Categories are
`transport`, `timeout`, `http_4xx`, `http_429`, `http_5xx`, `sse_protocol`, and `other`.

Client latency percentiles use deterministic nearest rank: sort N successful observations and
select one-based rank `ceil(p*N)`. The CLI reports p50, p90, p95, and p99. Failed requests are not
mixed into successful latency distributions; they remain visible in attempted/failed counts and
error rate.

Request throughput is exactly:

```text
successful timed requests / measured wall-clock seconds
```

The numerator is not attempted requests. Error rate is `failed / attempted`.

Output-token throughput means generated output tokens per measured second. It is populated only
from an authoritative source:

- per-request `usage.completion_tokens` when every successful response has valid usage;
- a configured before/after `vllm:generation_tokens` delta for direct vLLM; or
- a configured before/after `gateway_observed_output_tokens_total` delta for gateway runs.

The Prometheus counter source supersedes client usage when `--output-token-counter` is set. For a
batched response, the harness never divides aggregate usage among members. A counter decrease marks
the run `invalid_counter_reset`; the harness never emits a negative token rate.

### Streaming and TTFT

TTFT is measured only in `streaming` mode. It ends at the first complete SSE `data:` JSON event
whose `choices[*].delta.content` is a non-empty string. It does not end at the first TCP read, SSE
comment, role-only delta, usage-only event, or `[DONE]`.

The parser buffers across network reads, handles several events in one read, accepts CRLF or LF,
ignores comments, processes usage events, requires `[DONE]`, and caps a single event at 1 MiB. A
malformed, oversized, or incomplete protocol fails safely as `sse_protocol`. Tests cover fragmented
content, role-only events, usage, comments, `[DONE]`, and protocol errors.

The harness does **not** calculate TPOT or ITL by treating an SSE delta as a token. SSE chunks,
events, bytes, characters, and tokens are different units. If TPOT or ITL are reported, they come
from the official `vllm bench serve` client and must be labeled `vllm-bench client metrics`.
Non-streaming batch comparisons do not report TTFT.

## Raw metric and system evidence

If metric URLs are supplied, the CLI scrapes immediately before and after the measured phase and
retains the raw text. Absolute counters are never called run results. The parser supports labeled
series, new lazily materialized series with an implicit zero before value, histogram buckets,
missing optional families, and reset detection.

Gateway families inspected include:

```text
gateway_requests_total
gateway_request_duration_seconds
gateway_admission_queue_wait_seconds
gateway_client_ttft_seconds
gateway_backend_ttft_seconds
gateway_observed_output_tokens_total
gateway_token_accounting_requests_total
gateway_errors_total
gateway_batch_eligibility_total
gateway_batches_total
gateway_batch_size
gateway_batch_wait_seconds
gateway_admission_inflight
gateway_admission_queued
gateway_backend_healthy
gateway_backend_inflight
gateway_batch_pending
gateway_batch_inflight
```

vLLM families inspected when available include:

```text
vllm:generation_tokens
vllm:prompt_tokens
vllm:request_success
vllm:num_preemptions
vllm:num_requests_running
vllm:num_requests_waiting
vllm:kv_cache_usage_perc
vllm:time_to_first_token_seconds
vllm:e2e_request_latency_seconds
vllm:request_queue_time_seconds
vllm:request_time_per_output_token_seconds
```

An absent optional vLLM family is recorded as missing and does not fail the run. During measurement,
`--sample-interval 500ms` writes available scheduler, KV-cache, gateway admission, batch, and
backend gauges to `samples.csv`. Missing samples remain blank rather than becoming zero.

`--gpu-sample-interval 500ms` uses `nvidia-smi`, when available, for GPU utilization, memory used,
memory total, and power. The manifest explicitly records `gpu_sampling.available=false` when the
tool cannot be used. GPU sampling is optional and never required in CI. Run
`python scripts/sample_system.py --output results/environment.json` before a real matrix to record
host, kernel, Go/Python, CUDA, GPU name/memory, and driver details.

Gateway histogram deltas yield mean batch size (`sum_delta/count_delta`), bucket-estimated p95 batch
size and wait, admission queue wait, and size-versus-timeout flush counts. Histogram p95 selects the
first bucket whose delta reaches the nearest rank; it is a bucket estimate, not reconstructed raw
observations.

## Artifacts and schema

A matrix run directory contains:

```text
manifest.json
client-result.json
summary.json
gateway-before.prom       # gateway targets
gateway-after.prom
vllm-before.prom          # when configured
vllm-after.prom
samples.csv
gpu.csv                   # only when enabled and available
stdout.log
stderr.log
process.json
```

`client-result.json` uses `schema_version: 1` and contains metadata, configuration, UTC/monotonic
timing boundaries, warmup counts, summary, per-request results, metric artifact filenames,
environment data, and validity reasons. `schema/result.schema.json` is the portable JSON Schema;
`scripts/validate_result.py` is a dependency-free CI validator.

The manifest records the run ID and UTC time, gateway/harness SHAs and version, target, endpoint,
model, mode, concurrency, request/warmup counts, dataset digest, generation parameters, batch and
admission settings, prefix-cache state, vLLM version and launch configuration, GPU/driver/host
information, language versions, timeouts, and SHA-256 configuration fingerprints. Exact vLLM
launch configuration is required for valid real evidence. A run without `--vllm-config` remains
usable for client debugging but is explicitly invalid as performance evidence.

Fingerprints use normalized JSON with sorted keys for workload, vLLM, and gateway configuration.
The summarizer refuses comparisons with counter resets, zero measured requests, corpus/model/workload
drift, different vLLM fingerprints, different gateway OFF/ON SHAs, missing target metadata, or
incorrect OFF/ON batching flags.

## Matrix execution, repetitions, and order

The primary matrix defaults to concurrency `1,4,8,16,32,64`, three repetitions, and request count
`max(200, concurrency*20)`. It rotates arms deterministically:

```text
repeat 1: direct → gateway_no_batch → gateway_batch
repeat 2: gateway_no_batch → gateway_batch → direct
repeat 3: gateway_batch → direct → gateway_no_batch
```

A configurable quiet interval, five seconds by default, occurs between arms and is outside measured
duration. Prepare exact configs and environment metadata, then dry-run before launching:

```bash
python scripts/sample_system.py --output results/environment.json
export BENCH_MODEL='Qwen/Qwen2.5-0.5B-Instruct'
export BENCH_VLLM_VERSION='replace-with-exact-version'
export BENCH_GATEWAY_GIT_SHA='replace-with-deployed-gateway-full-sha'
export BENCH_VLLM_CONFIG="$PWD/configs/vllm-reference.json"
export BENCH_ENVIRONMENT_JSON="$PWD/results/environment.json"
export BENCH_VLLM_BASE_URL='http://127.0.0.1:8000'
export BENCH_VLLM_METRICS_URL='http://127.0.0.1:8000/metrics'
export BENCH_GATEWAY_OFF_BASE_URL='http://127.0.0.1:8081'
export BENCH_GATEWAY_OFF_METRICS_URL='http://127.0.0.1:8081/metrics'
export BENCH_GATEWAY_ON_BASE_URL='http://127.0.0.1:8082'
export BENCH_GATEWAY_ON_METRICS_URL='http://127.0.0.1:8082/metrics'
export BENCH_GATEWAY_AUTH_TOKEN='runtime-only-tenant-key'
python scripts/run_matrix.py configs/concurrency-sweep.json --dry-run
python scripts/run_matrix.py configs/concurrency-sweep.json
```

If a matrix process is interrupted, rerun the same configuration and output root with `--resume`.
The runner skips only results whose saved command, run coordinate, and secret-free plan fingerprint
exactly match the reconstructed plan and whose process and request accounting show a complete,
valid, failure-free run. The fingerprint includes SHA-256 values for the dataset and referenced
configuration/environment files, so changing a file in place forces a rerun. Missing, partial,
malformed, failed, invalid, legacy, or configuration-drifted results run again. Settling delays
apply only between runs that are actually executed, with no delay before the first remaining run.

```bash
python scripts/run_matrix.py configs/concurrency-sweep.json --resume
```

The example uses independently configured OFF and ON gateway processes so matrix metadata matches
server state. They must run the same commit and target the same vLLM process; do not run traffic to
both simultaneously. The runner never edits source, restarts services, or automatically tunes
anything. Resume is local artifact validation and deterministic plan reconstruction, not a
transactional checkpoint or distributed recovery protocol.

### Admission sizing is part of the experiment

Admission happens before batching, and gateway batching is tenant-local. If
`DYNAMIC_BATCH_MAX_SIZE=8`, both the benchmark tenant's `max_inflight` and
`GLOBAL_MAX_INFLIGHT` must be at least eight, preferably with headroom for concurrent batches. A
limit below batch size prevents intended aggregation and invalidates policy interpretation.
`run_matrix.py` rejects such configured targets, and the manifest records both limits.

### Prefix-cache control and reference vLLM launch

For the primary batching experiment, explicitly disable prefix caching so later arms do not benefit
from warmed prompt prefixes. If it is enabled, record that truthfully and treat prefix behavior as a
separate experiment.

One reproducible single-GPU reference command is:

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --dtype auto \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 4096 \
  --no-enable-prefix-caching \
  --host 0.0.0.0 \
  --port 8000
```

Record whether chunked prefill is explicitly set or left at the exact vLLM version's default. Do
the same for dtype, quantization, tensor parallelism, maximum model length, memory utilization,
maximum sequences, and maximum batched tokens. The harness does not hard-code a GPU provider and
can run separately from local GPU, RunPod, Vast.ai, or another Linux GPU host.

## Experiments

### A — gateway overhead with streaming

Use `streaming` mode and compare direct vLLM with gateway batching OFF at concurrency
`1,4,8,16,32,64`. Streaming bypasses the gateway batcher. Report client TTFT p50/p95/p99, E2E
p50/p95/p99, successful request throughput, output token throughput, error rate, gateway admission
queue wait, and gateway/backend-observed TTFT. This isolates auth, validation, admission, routing,
and SSE proxy overhead.

### B — incremental dynamic batching effect

Use `non_streaming`, `n=1`, one tenant, identical parameters, and all A/B/C arms. Report E2E,
request/output throughput, error rate, queue wait, batch wait, mean/effective batch size,
batch-size distribution, and size/timeout flush counts. Do not report TTFT. Concurrency one is a
negative control: its batch wait should normally be bounded by the configured max wait, and the
latency penalty must remain visible.

### C — batch policy sensitivity

`configs/batching-sweep.json` provides the full optional grid `max_size=2,4,8`,
`max_wait=0.001,0.005,0.010` at concurrency 16 and 32. Each policy
needs an independently and truthfully configured gateway URL, or must be run separately after an
operator reconfiguration. Report tradeoffs among achieved batch size, wait, E2E, and throughput;
do not declare a universal optimum.

### D — optional tenant isolation

Run a noisy tenant A client at sustained closed-loop concurrency and a separate lower-concurrency
probe tenant B client. Use separate output directories and auth environment variables. Retain both
timelines and gateway admission samples. Ask whether B retains bounded admission opportunity rather
than waiting for all of A's backlog. This does not claim equal throughput or perfect performance
isolation.

The optional helper starts both closed-loop clients against one gateway and retains their results
separately:

```bash
export BENCH_TENANT_A_AUTH_TOKEN='runtime-only-noisy-key'
export BENCH_TENANT_B_AUTH_TOKEN='runtime-only-probe-key'
python scripts/run_tenant_isolation.py configs/tenant-isolation.json
```

The default probe uses concurrency one and far fewer requests; it is lower-intensity closed-loop
traffic, not a claimed Poisson/open-loop arrival rate. System-wide counter deltas overlap by design
in this interference experiment, so interpret per-client E2E/errors alongside the shared gauge
timeline rather than attributing shared token counters to either tenant.

### E — manual backend failure evidence

Do not automate destructive termination in the standard command:

1. Start one gateway with two healthy, otherwise identical vLLM backends.
2. Start load and capture gateway metrics/samples.
3. Stop backend B manually; record the UTC action time.
4. Observe `gateway_backend_healthy{backend_id="B"}` become zero, future routing use A, backend
   inflight gauges, gateway errors, and client request errors.
5. Restart B with the same config and observe health recovery and later routing.
6. Retain manifests, action timestamps, raw metrics, client results, and error counts.

The expected property is health-aware future-request routing and recovery, not transparent replay
of an in-flight failed generation.

## Official `vllm bench serve` cross-check

The official client is a secondary cross-check and the appropriate source for TPOT/ITL and optional
controlled request-rate, Poisson, or burst experiments. It does not replace the gateway-specific Go
batch metrics.

Direct example:

```bash
vllm bench serve \
  --backend openai-chat \
  --base-url "$BENCH_VLLM_BASE_URL" \
  --endpoint /v1/chat/completions \
  --model "$BENCH_MODEL" \
  --dataset-name random \
  --input-len 256 \
  --output-len 128 \
  --max-concurrency 16 \
  --num-prompts 200 \
  --num-warmups 20 \
  --save-result \
  --save-detailed
```

Gateway example, with the credential expanded only at execution time and never committed:

```bash
vllm bench serve \
  --backend openai-chat \
  --base-url "$BENCH_GATEWAY_OFF_BASE_URL" \
  --endpoint /v1/chat/completions \
  --header "Authorization=Bearer $BENCH_GATEWAY_AUTH_TOKEN" \
  --model "$BENCH_MODEL" \
  --dataset-name random \
  --input-len 256 \
  --output-len 128 \
  --max-concurrency 16 \
  --num-prompts 200 \
  --num-warmups 20 \
  --save-result \
  --save-detailed
```

Check `vllm bench serve --help` for the installed version's header syntax and record the exact
command with the credential redacted. Label TPOT/ITL as `vllm-bench client metrics` and preserve
their provenance separately from gateway Prometheus observations.

## Summaries and plots

Create a compact CSV, per-repeat deltas, median/range comparison, and Markdown table:

```bash
python scripts/summarize.py results/MATRIX_ID --output-dir results/MATRIX_ID/report
python scripts/plot.py results/MATRIX_ID/report/summary.csv \
  --output-dir results/MATRIX_ID/report/plots
```

The report uses explicit `GW-off - Direct` gateway-overhead deltas and `GW-on - GW-off` batching
effect deltas. Throughput also includes `(GW-on/GW-off - 1)*100`. It says delta or difference—not
improvement. Three repetitions produce per-repeat values plus median/min/max, not significance,
confidence, or regression theater. Separate SVGs show E2E p95, TTFT p95, request throughput, output
TPS, mean batch size, batch wait p95, and error rate. Blank means not measured; it is never replaced
with zero. The source CSV/JSON remains authoritative.

Permissible evidence language includes:

> At C=1, a 5 ms batch wait increased p95 E2E by X ms.

> At C=32, mean gateway batch size was Y and output throughput differed from batching-off by Z%.

Without valid evidence, do not claim that batching improves GPU utilization, reduces cost, doubles
throughput, or is generally optimal.

## Local smoke and verification

The fake server is deterministic and protocol-only:

```bash
python scripts/fake_server.py --port 18000
python scripts/run_matrix.py configs/smoke.json
python scripts/validate_result.py results/local-fake-smoke/**/client-result.json
python scripts/summarize.py results/local-fake-smoke \
  --output-dir results/local-fake-smoke/report
```

Run the complete no-GPU checks from the repository root:

```bash
ruff check .
ruff format --check .
pytest -q
python -m pip check
python -m unittest discover -s benchmark/tests
(cd benchmark && test -z "$(gofmt -l .)" && go vet ./... && go test ./...)
git diff --check
```
