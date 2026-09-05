# 2026-09-05 Qwen3-8B memory calibration

Status: V1 shared-replica candidate calibrated; not performance evidence  
Evidence confidence: `ARTIFACT_BACKED`

## Question and frozen environment

Which vLLM policy yields two equal, independently healthy Qwen3-8B replicas on one H100, regardless
of startup order? Model `Qwen/Qwen3-8B` revision `b968826d9c46dd6066d109eabc6255188de91218`, repo
SHA `400d79fae3269a6d3722a44bc9d3e824f78617bc`, vLLM 0.26.0, BF16 weights, FP8 KV, TP=1, max
length 4096, max sequences 128, max batched tokens 8192. The H100 UUID and concise 2026-09-05
environment fingerprint are preserved in curated evidence. The Python package probe was incomplete,
though vLLM version/config are independently logged.

## Calibration evolution

1. Automatic KV with `gpu_memory_utilization=0.44`: both fit, but first/second capacities differed:
   18.03/18.54 GiB and 262,608/269,936 tokens.
2. Fixed `kv_cache_memory_bytes=18,700,000,000` alone: A started at 17.42 GiB, 253,632 tokens,
   61.92x at 4096; B failed because the default 0.92 admission guard still demanded about 72.85 GiB
   free with only about 44.3 GiB available.
3. Fixed KV plus utilization 0.44: both reported 17.42 GiB, 253,632 tokens, 61.92x, and 35,172 MiB
   idle EngineCore memory; total idle allocation was 70,372 MiB.
4. B-first then A reproduced those exact capacities and allocations; both passed smoke requests.

## Decision and limits

The combined policy became the V1 candidate and advanced to T2 stress. It demonstrates configuration,
capacity, admission, and startup-order invariance only in this tested environment—not latency,
throughput, or portability. Archive SHA-256:
`ab2cac1819b48778b963dc1df1aa8b6f10a26f5dd09bccde4272a3b69e68572e`.
