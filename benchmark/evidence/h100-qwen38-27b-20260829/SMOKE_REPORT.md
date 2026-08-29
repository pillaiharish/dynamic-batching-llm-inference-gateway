# H100 Real Dynamic-Batching Smoke Report

## Environment

- GPU: NVIDIA H100 80GB HBM3; driver 610.57.04; CUDA UMD 13.3; toolkit 13.0.
- vLLM 0.26.0; Transformers 5.14.1; model `Qwen/Qwen3.8-27B`; port 18000.
- Command: `vllm serve Qwen/Qwen3.8-27B --max-model-len 32768 --gpu-memory-utilization 0.90 --host 0.0.0.0 --port 18000 --max-num-seqs 16 --kv-cache-dtype fp8 --language-model-only --tensor-parallel-size 1 --compilation-config {"cudagraph_capture_sizes": [1,2,3,4,5,6,7,8]}`.
- Engine: BF16 weights, FP8 KV cache, tensor parallelism 1, max model length 32768, max sequences 16, max batched tokens 8192, chunked prefill enabled, prefix caching disabled, no quantization.
- Go: 1.27.0 linux/amd64. Gateway: 0.8.0 at `c62661e67c94052f8b1269b73852424bec31be61`.

## Infrastructure validation

- Python: 260 passed. Go tests: passed. Go vet: passed.
- Authenticated vLLM health/models: HTTP 200; served model verified.
- Direct ordinary completion: HTTP 200. Direct two-conversation batch: HTTP 200 with indices 0 and 1 and aggregate usage.
- Gateway OFF and ON ordinary completions: HTTP 200; both `/readyz` endpoints ready.
- vLLM metrics: unauthenticated HTTP 200. Output-token counter: `vllm:generation_tokens_total`.
- Six result files generated and independently accepted by `scripts/validate_result.py`.

## C=1 results

| Arm | Attempted/success/failed | Error | E2E p50/p90/p95/p99 ms | Req/s | Tok/s | Token source |
| --- | --- | ---: | --- | ---: | ---: | --- |
| A direct | 40/40/0 | 0% | 2684.404 / 2694.998 / 2697.435 / 2708.559 | 0.4036 | 47.318 | `vllm:generation_tokens_total` |
| B gateway OFF | 40/40/0 | 0% | 2691.225 / 2704.461 / 2708.696 / 2714.536 | 0.4023 | 47.168 | `gateway_observed_output_tokens_total` |
| C gateway ON | 40/40/0 | 0% | 2699.466 / 2707.898 / 2710.032 / 3026.618 | 0.3996 | 46.857 | `gateway_observed_output_tokens_total` |

Gateway admission queue p95 was represented by the 1 ms histogram bucket for both gateway arms. Gateway ON formed 40 batches with mean size 1 and p95 size 1; all 40 were timeout flushes, size flush count was unavailable/null, and batch-wait p95 was represented by the 10 ms histogram bucket around the configured 5 ms wait. This is the expected C=1 behavior.

## C=8 results

| Arm | Attempted/success/failed | Error | E2E p50/p90/p95/p99 ms | Req/s | Tok/s | Token source |
| --- | --- | ---: | --- | ---: | ---: | --- |
| A direct | 40/40/0 | 0% | 3417.269 / 3866.752 / 3900.261 / 3906.182 | 2.2992 | 270.273 | `vllm:generation_tokens_total` |
| B gateway OFF | 40/40/0 | 0% | 3491.209 / 4000.585 / 4040.199 / 4059.467 | 2.2485 | 264.308 | `gateway_observed_output_tokens_total` |
| C gateway ON | 40/40/0 | 0% | 2970.312 / 2988.202 / 2988.409 / 2988.637 | 2.6967 | 319.564 | `gateway_observed_output_tokens_total` |

Gateway admission queue p95 was represented by the 1 ms histogram bucket. Gateway ON formed real multi-request batches: mean size 5, p95 size 8, 3 size flushes, 5 timeout flushes, and batch-wait p95 represented by the 10 ms bucket. The batch-formation gate passed.

## Descriptive deltas

| C | Measure | B-A gateway overhead | C-B batching effect |
| ---: | --- | ---: | ---: |
| 1 | E2E p50 / p90 / p95 / p99 ms | +6.821 / +9.463 / +11.260 / +5.977 | +8.241 / +3.437 / +1.336 / +312.082 |
| 1 | Request throughput req/s | -0.0013 | -0.0026 (-0.66%) |
| 1 | Output throughput tok/s | -0.151 | -0.310 (-0.66%) |
| 8 | E2E p50 / p90 / p95 / p99 ms | +73.940 / +133.833 / +139.938 / +153.285 | -520.897 / -1012.383 / -1051.791 / -1070.830 |
| 8 | Request throughput req/s | -0.0507 | +0.4483 (+19.94%) |
| 8 | Output throughput tok/s | -5.965 | +55.257 (+20.91%) |

These are one-repetition descriptive differences, not claims of improvement, speedup, or production performance.

## GPU and vLLM observations

| C | Arm | GPU util mean/max | Memory mean/max MiB | Power mean/max W | vLLM running max | waiting max | KV-cache max |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: |
| 1 | direct | 91.98% / 100% | 74033 / 74033 | 460.71 / 487.45 | 1 | 0 | 1.06% |
| 1 | gateway OFF | 93.97% / 100% | 74033 / 74033 | 461.41 / 487.72 | 1 | 0 | 1.06% |
| 1 | gateway ON | 90.11% / 100% | 74033 / 74033 | 459.02 / 487.88 | 1 | 0 | 1.06% |
| 8 | direct | 76.46% / 100% | 74033.74 / 74035 | 416.80 / 495.45 | 8 | 0 | 8.51% |
| 8 | gateway OFF | 72.44% / 100% | 74035 / 74035 | 411.11 / 497.19 | 8 | 0 | 8.51% |
| 8 | gateway ON | 94.83% / 100% | 74035 / 74035 | 464.10 / 497.26 | 8 | 0 | 8.51% |

- Prompt-token delta was 2441 in every run. Generation-token deltas were 4690 at C=1; at C=8 they were 4702 direct/OFF and 4740 ON, reflecting actual generated lengths.
- The `vllm:num_preemptions` metric was unavailable, so the smoke makes no preemption-count claim.
  Waiting requests sampled at zero in every run, and no observed counter reset occurred.
- Higher GPU utilization is supporting evidence only and is not treated as proof of better service performance.

## Validity and caveats

- All six runs report `valid=true`; counters were monotonic; no requests failed; TTFT was correctly omitted for non-streaming runs.
- This is a one-repetition, same-host smoke using a Qwen3.8 reasoning model. It validates protocol, metrics, counter accounting, and real batch formation but is not final performance evidence.
- Prefix caching was independently verified disabled, so arm ordering was not affected by prefix-cache reuse. The experiment still retains fixed-order and single-repetition limitations.
- Batched gateway responses expose aggregate rather than per-member usage, so their authoritative per-request token coverage is zero; throughput comes from the gateway aggregate Prometheus counter.

## Recommendation

`READY_FOR_FULL_SWEEP`

Review this smoke before authorizing any larger experiment. Rotate runtime credentials after
artifact transfer as standard hygiene.
