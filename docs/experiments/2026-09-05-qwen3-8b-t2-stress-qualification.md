# 2026-09-05 Qwen3-8B T2 stress qualification

Status: attempt 1 invalid; corrected attempt passed capacity/stability qualification  
Evidence confidence: `ARTIFACT_BACKED`

## Question and topology

Could active replica A sustain the frozen worst-case functional workload while B remained resident
and idle on the same H100? Both used the pinned Qwen3-8B revision and combined 0.44/18,700,000,000
memory policy. A received direct traffic; B was deliberately not registered with the gateway.

## Attempt 1: invalid generator

All 64 requests returned HTTP 400, local prompt tokens were recorded as 2, no GPU/KV activity
occurred, and both engines stayed healthy. `apply_chat_template(..., tokenize=True)` returned a
`BatchEncoding`; `len(encoded)` counted its `input_ids` and `attention_mask` fields. The binary search
built a huge prompt that vLLM rejected before inference. This is workload-generator failure evidence,
not negative H100 capacity evidence.

## Attempt 2: corrected qualification

Counting `encoded.input_ids` found 1012 text repetitions produced exactly 1024 templated tokens;
1013 produced 1025. One request cross-checked local 1024 against server `usage.prompt_tokens=1024`.
The final closed-loop C=64 run requested exactly 1024 input and forced 1024 output tokens with
`ignore_eos=true`: 64/64 succeeded, all 64 completions were full length, total output was 65,536,
server prompt min/max was 1024/1024, exit code 0, both health checks were 200, preemptions were zero,
problem scans were empty, and B inference counters did not change.

Post-run A used about 36,138 MiB while idle B remained 35,172 MiB. This is additional active runtime
allocation, not changed KV allocation. The 11.73-second wall time and latency fields are debug-only and
must not support performance claims. The decision accepted the V1 policy for the next T1/T2/T3
campaign. Archive SHA-256:
`d1a1562feeba961f517fe7514d0a3689d1f3eefc826b2279678cf80693a2c496`.
