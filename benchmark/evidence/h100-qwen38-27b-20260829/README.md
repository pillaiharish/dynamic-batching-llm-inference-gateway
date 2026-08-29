# H100 Qwen3.8-27B evidence

This directory contains the deliberately curated evidence from the 2026-08-29 real-GPU smoke and
three-repetition A/B/C concurrency sweep. The gateway and Go benchmark harness were pinned to
`c62661e67c94052f8b1269b73852424bec31be61` (version 0.8.0).

The full sweep contains 54 unique runs covering concurrency 1, 4, 8, 16, 32, and 64. Every retained
run completed 120/120 measured requests with zero failures and `valid=true`. The primary finding is
narrow: for this non-streaming Qwen3.8-27B workload, gateway batching with maximum size 8 and a 5 ms
wait helped at sustained concurrency 4–8, but the effect was not consistent at concurrency 16–64.
Dynamic batching therefore remains disabled by default.

## Contents

- `SMOKE_REPORT.md` records the protocol, counter, and real batch-formation smoke.
- `FULL_SWEEP_REPORT.md` records the environment, methodology, results, caveats, and next step.
- `summary.csv`, `comparison.json`, and `comparison.md` are the source data for the curated tables.
- `plots/` contains SVG views derived from `summary.csv`; the CSV/JSON remain authoritative.
- `configs/`, `environment.json`, and `system.txt` record the non-secret experiment configuration.
- `run-matrix-resume.patch` preserves the exact orchestration-only recovery patch used on the host.
- `ARTIFACT_CHECKSUMS.sha256` identifies the raw archives retained outside Git.

The request-level JSON, Prometheus snapshots, GPU samples, and logs remain in the checksummed raw
archive and are intentionally excluded from Git. These results apply only to the recorded host,
model, workload, gateway settings, and vLLM configuration; they are not a general performance claim.
