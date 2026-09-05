# 2026-08-29 Qwen3.8-27B H100 full sweep

Status: valid scoped benchmark  
Evidence confidence: `ARTIFACT_BACKED`

## Question, design, and limits

For the recorded non-streaming workload, how did gateway overhead and compatible aggregation vary
with concurrency? A, B, and C follow the [claim boundary](../architecture/decisions/benchmark-claim-boundaries.md).
The hypothesis was that real multi-request aggregation could help at moderate concurrency but add
wait at C=1. The sweep does not prove behavior for other prompts, models, streaming, hosts, budgets,
or production traffic.

The environment matched the smoke: one H100, driver 610.57.04, CUDA toolkit 13.0, Qwen3.8-27B
(revision unrecorded), vLLM 0.26.0, BF16/FP8, TP=1, max length 32768, max sequences 16, max batched
tokens 8192, gateway/harness SHA `c62661e67c94052f8b1269b73852424bec31be61`.

The exact matrix was concurrency 1, 4, 8, 16, 32, 64 × A/B/C × three rotated repetitions: 54 runs.
Each used 10 warmups and 120 measured closed-loop requests, non-streaming `n=1`, temperature 0,
top_p 1, max_tokens 128, and seed 1. All 6480 measured requests succeeded and counters were monotonic.

## Results, interruptions, and interpretation

Median paired C−B at C=4 was p95 −520.48 ms, +0.088 req/s, +11.13 output tok/s; at C=8 it was
−976.07 ms, +0.447 req/s, +53.93 tok/s. C=1 was slightly unfavorable, and C=16/32/64 had wide or
sign-changing ranges and inconsistent directions. vLLM reached 16 running sequences at C=16;
waiting began at C=32. The runner was interrupted and resumed; incomplete/credential-error artifacts
were rerun and excluded by validation. This changed wall-clock order and reinforces caution higher up.

The decision was to keep aggregation off by default and consider it only for this revalidated moderate
concurrency regime. C−B is aggregation effect, not generic GPU batching. Re-run after model/runtime/
hardware/config/workload changes. Archive SHA-256:
`f2600ab5cacb18570efd7bc77a6a9e9ead486a589a373b998c07d0ba659a0d9b`.
