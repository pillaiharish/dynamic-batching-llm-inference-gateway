# H100 Qwen3.8-27B full A/B/C sweep

## Scoped conclusion

Keep `DYNAMIC_BATCHING_ENABLED=false` as the general default. For this exact non-streaming workload,
consider gateway batching (`max_size=8`, `max_wait=5 ms`) only for a deployment that reproduces
sustained concurrency 4–8 and independently revalidates the latency/throughput trade-off. At C=8 in
this experiment, batching versus gateway-off reduced median p95 by 976 ms (23.5%) and increased
median request throughput by 20.1% and output throughput by 20.7%. The observed direction did not
remain consistent at C=16, 32, or 64.

## Validation and environment gates

- 54/54 result files passed `scripts/validate_result.py`; all comparisons passed A/B/C consistency
  checks; every measured run had 120 successes, zero failures, `valid=true`, and monotonic counters.
- Three repetitions exist for every combination of concurrency 1, 4, 8, 16, 32, and 64 and arms
  A direct, B gateway without batching, and C gateway with batching.
- Dataset SHA-256: `689233a4dc9e192c4a1716ce749f5855e7a89238e17cf3c2813626bc4fa7defd`.
- GPU: NVIDIA H100 80GB HBM3; driver 610.57.04; CUDA toolkit 13.0; Go 1.27.0.
- Model: `Qwen/Qwen3.8-27B`; vLLM 0.26.0; BF16 weights; FP8 KV cache; TP=1;
  max model length 32768; max sequences 16; max batched tokens 8192; chunked prefill enabled;
  prefix caching disabled.
- Gateway: 0.8.0 at Git SHA `c62661e67c94052f8b1269b73852424bec31be61`.
- Gateway and Go benchmark source matched that commit. The Python matrix orchestrator gained an
  uncommitted `--resume` recovery patch after the original session was interrupted; the exact patch
  is retained with this evidence.
- Workload: non-streaming Chat Completions, closed-loop, 120 measured requests plus 10 warmups,
  `temperature=0`, `top_p=1`, `max_tokens=128`, `seed=1`.

## Primary results

Values are median `[min, max]` across three repetitions. Latencies are milliseconds; throughput is
requests/s and generated output tokens/s. A=direct vLLM, B=gateway off, C=gateway batching on.

| C | Arm | p50 ms | p95 ms | p99 ms | Req/s | Tok/s | Errors |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | A | 2687.18 [2687.16,2688.73] | 2701.26 [2698.59,2701.81] | 2705.12 [2703.91,2814.15] | 0.404 [0.404,0.404] | 47.25 [47.25,47.25] | 0/360 |
| 1 | B | 2692.08 [2690.45,2692.09] | 2704.86 [2703.25,2707.09] | 2712.60 [2708.42,2712.96] | 0.403 [0.403,0.403] | 47.17 [47.17,47.22] | 0/360 |
| 1 | C | 2698.97 [2698.43,2700.72] | 2711.44 [2707.99,2722.41] | 2718.82 [2713.34,2727.35] | 0.402 [0.401,0.402] | 47.05 [47.00,47.08] | 0/360 |
| 4 | A | 3242.37 [3198.45,3259.52] | 3428.22 [3416.20,3441.97] | 3445.00 [3444.92,3452.40] | 1.326 [1.325,1.337] | 154.30 [153.60,156.26] | 0/360 |
| 4 | B | 3200.70 [3178.33,3247.31] | 3433.90 [3366.12,3455.01] | 3458.93 [3399.36,3470.26] | 1.332 [1.320,1.355] | 155.58 [153.24,158.20] | 0/360 |
| 4 | C | 2835.26 [2773.76,2851.54] | 2918.97 [2876.92,2934.53] | 2969.43 [2894.27,2977.25] | 1.413 [1.407,1.420] | 166.53 [164.45,166.70] | 0/360 |
| 8 | A | 3816.54 [3782.60,3825.02] | 4114.51 [4110.95,4181.32] | 4182.02 [4167.23,4215.88] | 2.231 [2.230,2.241] | 261.22 [260.33,263.17] | 0/360 |
| 8 | B | 3790.79 [3752.77,3870.25] | 4151.03 [4066.28,4230.49] | 4214.99 [4201.97,4247.96] | 2.226 [2.199,2.251] | 260.58 [257.59,263.05] | 0/360 |
| 8 | C | 2985.67 [2934.98,3025.54] | 3137.88 [3090.21,3293.18] | 3138.28 [3090.59,3373.29] | 2.684 [2.596,2.697] | 316.98 [303.63,317.15] | 0/360 |
| 16 | A | 6222.74 [6128.95,6228.24] | 6863.25 [6515.50,6926.44] | 6898.23 [6599.29,7017.45] | 2.681 [2.647,2.817] | 314.35 [310.14,330.24] | 0/360 |
| 16 | B | 5325.30 [3374.36,5668.47] | 5475.28 [3520.48,6558.90] | 5513.22 [3571.48,6592.39] | 3.166 [3.038,4.842] | 370.32 [356.37,565.77] | 0/360 |
| 16 | C | 5525.03 [4642.27,5626.92] | 5576.91 [4957.03,5995.48] | 5577.30 [4957.42,5998.92] | 2.888 [2.830,3.723] | 339.73 [331.97,437.77] | 0/360 |
| 32 | A | 11608.93 [9362.19,11721.45] | 12838.72 [10895.04,12916.41] | 13081.93 [11002.08,13270.66] | 2.712 [2.702,3.384] | 316.71 [314.91,397.69] | 0/360 |
| 32 | B | 11806.63 [6292.47,11929.23] | 12795.68 [7032.26,13579.37] | 13012.43 [7032.51,13882.45] | 2.672 [2.661,4.879] | 314.61 [309.99,571.55] | 0/360 |
| 32 | C | 9879.43 [8665.10,11046.54] | 12126.85 [10882.53,12593.25] | 12127.23 [10882.82,12593.60] | 3.080 [2.769,3.457] | 360.42 [320.68,404.47] | 0/360 |
| 64 | A | 22334.01 [11242.23,22858.82] | 25441.92 [13263.35,26839.79] | 25661.76 [13336.59,27207.01] | 2.647 [2.560,4.992] | 310.00 [302.31,581.11] | 0/360 |
| 64 | B | 18280.39 [14803.42,21153.56] | 21371.79 [20219.95,24708.33] | 21987.04 [20811.58,25182.36] | 3.098 [2.749,3.794] | 363.34 [321.60,445.11] | 0/360 |
| 64 | C | 18954.96 [12952.13,21282.53] | 22869.06 [19919.77,24472.76] | 23779.04 [19923.28,24862.22] | 3.027 [2.788,3.907] | 358.27 [325.62,459.10] | 0/360 |

## A/B/C effects

Median run-paired differences are shown below. Negative latency is better; positive throughput is
better. Full three-repetition ranges are in `comparison.md` and `comparison.json`.

| C | B−A p95 ms | B−A Req/s | B−A Tok/s | C−B p95 ms | C−B Req/s | C−B Tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | +4.66 | -0.001 | -0.08 | +8.19 | -0.001 | -0.12 |
| 4 | +13.03 | +0.007 | +1.27 | -520.48 | +0.088 | +11.13 |
| 8 | +40.08 | -0.004 | -0.64 | -976.07 | +0.447 | +53.93 |
| 16 | -1387.97 | +0.484 | +55.98 | +520.21 | -0.335 | -38.35 |
| 32 | +662.96 | -0.041 | -4.92 | -668.82 | +0.108 | +10.69 |
| 64 | -4070.13 | +0.451 | +53.34 | +2649.11 | -0.310 | -37.72 |

Gateway-only overhead (B−A) is very small and stable through C=8. At C>=16 the paired deltas have
wide, sign-changing ranges, so they should not be interpreted as a causal gateway speedup.

## Batch formation, admission, scheduler, KV cache, and GPU

| C | Median mean batch | p95 batch | Batch wait p95 | Size flushes (3 runs) | Timeout flushes (3 runs) | vLLM running/waiting max | KV max |
| ---: | ---: | ---: | ---: | --- | --- | --- | ---: |
| 1 | 1.00 | 1 | 10 ms | 0/0/0 | 120/120/120 | 1/0 | 1.1% |
| 4 | 4.00 | 4 | 10 ms | 0/0/0 | 30/36/30 | 4/0 | 4.3% |
| 8 | 4.00 | 8 | 10 ms | 7/2/1 | 16/36/29 | 8/0 | 8.5% |
| 16 | 6.67 | 8 | 10 ms | 12/11/13 | 6/8/4 | 16/0 | 17.0% |
| 32 | 5.00 | 8 | 10 ms | 8/9/7 | 14/15/17 | 16/16 | 17.0% |
| 64 | 5.22 | 8 | 25 ms | 2/4/8 | 21/21/15 | 16/48 | 17.0% |

- Gateway admission queue p95 was always in the 1 ms bucket. No admission errors occurred.
- At C=1 every batch timed out at size one, explaining the small batching penalty. Real multi-request
  batching starts at C=4; size-triggered flushes become common at C=8–16.
- vLLM reaches its configured 16 running sequences at C=16. Waiting begins at C=32 and reaches 48 at
  C=64. This is consistent with scheduler saturation and coincides with the latency knee above C=16.
- KV-cache utilization peaks around 17%, so the retained samples do not indicate KV exhaustion. The
  `vllm:num_preemptions` metric was unavailable in vLLM 0.26.0, so this run cannot claim a zero
  preemption count. No reset or failure evidence suggests preemption-induced invalidity.
- GPU memory stayed at approximately 74,035 MiB. Median mean GPU utilization for A/B/C was
  81.7/81.1/92.3% at C=4, 74.9/71.9/91.9% at C=8, and roughly 49–59% at C>=16. Peak utilization
  reached 100% in every arm. The C arm had higher sustained utilization in the recorded C=4–8
  samples; this is supporting evidence for this experiment, not a general performance claim.

## Plateau and latency knee

Direct output throughput rises from 47 tok/s at C=1 to 314 tok/s at C=16, then plateaus near
310–317 tok/s at C=32–64 by the three-run median. Gateway batching reaches 317 tok/s already at C=8,
then provides no stable additional gain. The practical latency knee is C=16: p95 is about 4.1 s at
C=8, 5.6–6.9 s at C=16, 12.1–12.8 s at C=32, and 21.4–25.4 s at C=64. C=32 and C=64 repetition
ranges are wide, reinforcing that the system is beyond its stable low-latency region.

## Caveats and interruption record

- The sweep is one host, one model, one prompt corpus, one output cap, and non-streaming only.
- The original runner was interrupted after 20 artifacts when its parent Codex session was closed.
  Nineteen runs had complete success records. One older incompletely recorded C=1 batched result was
  rerun. A recovery attempt without inherited credentials produced six immediate HTTP 4xx artifacts;
  it was stopped, credentials were recovered from the local services, and all six were rerun. An
  independent local audit confirmed that the retained set contains 54 unique coordinates, 54 zero
  process exit codes, and 54 runs with 120/120 successful requests and `valid=true`.
- The pause and rerun changed wall-clock ordering for the affected repetition. High-concurrency
  variability means C>=16 deltas are descriptive, not proof of improvement.
- The authoritative curated comparison artifacts are `summary.csv` and `comparison.json`; the raw
  per-run directories remain in the checksummed external archive. SVG plots are visual aids only.

## Security confirmation

Credentials were supplied only through process environments. Commands, result JSON, manifests,
CSV, reports, and packaged artifacts contain no Authorization headers, prompts, generated text, or
API-key values. Rotate the runtime vLLM and gateway keys after artifact transfer as standard hygiene.

## Targeted confirmation

For a later higher-sample confirmation, rerun C=4, C=8, and C=16 with 300–500 measured requests and
three repetitions. C=4 and C=8 confirm the observed moderate-concurrency region; C=16 is the boundary
control where the batching effect changed direction. Keep any conclusion limited to the recorded
model, workload, gateway configuration, and vLLM deployment.

## Artifact provenance

The raw distributable is `qwen38-h100-full-sweep-v1.tar.gz`, retained outside Git. Its authoritative
SHA-256 is `f2600ab5cacb18570efd7bc77a6a9e9ead486a589a373b998c07d0ba659a0d9b` and is also recorded in
`ARTIFACT_CHECKSUMS.sha256`. The compact CSV/JSON files in this directory are authoritative for the
curated comparison; the raw archive remains authoritative for request-level evidence.

TARGETED_CONFIRMATION_READY
