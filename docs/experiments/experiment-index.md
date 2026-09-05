# Experiment index

| Date | Experiment | Status | Model | Topology | Primary question | Outcome | Confidence | Evidence |
|---|---|---|---|---|---|---|---|---|
| 2026-08-29 | [H100 real smoke](2026-08-29-h100-real-smoke.md) | Valid smoke | Qwen3.8-27B | A/B/C, one vLLM | Does the protocol work on real H100? | Ready for sweep; no performance claim | ARTIFACT_BACKED | `benchmark/evidence/h100-qwen38-27b-20260829/` |
| 2026-08-29 | [Full sweep](2026-08-29-qwen38-full-sweep.md) | Valid, scoped | Qwen3.8-27B | A/B/C, one vLLM | Where did aggregation help this workload? | Favorable C−B at C=4–8; inconsistent higher | ARTIFACT_BACKED | same plus external archive |
| 2026-08-30 | [Targeted confirmation](2026-08-30-qwen38-targeted-confirmation.md) | Invalid | Qwen3.8-27B | A/B/C, one vLLM | Does the earlier direction repeat? | 25 C=8/B 503s invalidate matrix | ARTIFACT_BACKED | curated failure summary + external archive |
| 2026-08-30 | [Clean C=8 confirmation](2026-08-30-c8-clean-confirmation.md) | Valid, directional | Qwen3.8-27B | A/B/C, one vLLM | Does C=8 direction repeat cleanly? | 2700/2700; favorable C−B in 3 repeats | ARTIFACT_BACKED | curated summary + external archive |
| 2026-09-05 | [Qwen3-8B memory calibration](2026-09-05-qwen3-8b-memory-calibration.md) | V1 candidate calibrated | Qwen3-8B pinned | Two resident replicas | Which policy gives equal admissible KV? | Fixed KV + 0.44 is order-invariant here | ARTIFACT_BACKED | curated summary + external archive |
| 2026-09-05 | [T2 stress qualification](2026-09-05-qwen3-8b-t2-stress-qualification.md) | Attempt 1 invalid; attempt 2 passed | Qwen3-8B pinned | T2 | Can active A sustain the frozen worst case with B resident? | 64/64 after generator correction | ARTIFACT_BACKED | curated attempt summaries + external archive |
