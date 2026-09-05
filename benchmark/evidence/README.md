# Curated benchmark evidence

No performance conclusion is valid without a real run artifact identifying hardware, model, vLLM
version and launch configuration, workload and dataset digest, gateway Git SHA, benchmark harness
SHA, exact gateway configuration, and benchmark configuration.

This directory is for small, deliberately reviewed evidence artifacts such as a manifest, compact
summary CSV/JSON, source-backed plots, and a short neutral conclusion. Raw run directories belong in
the Git-ignored `benchmark/results/` tree. Never commit API keys, Authorization headers, cloud
credentials, generated model content, or enormous raw traces.

Available curated evidence:

- [H100 Qwen3.8-27B smoke and full A/B/C sweep](h100-qwen38-27b-20260829/README.md) — one H100,
  vLLM 0.26.0, non-streaming, concurrency 1–64, and three full-sweep repetitions.
- `2026-08-30-targeted-confirmation/failure-summary.json` — invalid 25-response health incident.
- `2026-08-30-c8-clean-confirmation/summary.json` — three clean directional C=8 repetitions.
- `2026-09-05-qwen3-8b-memory-calibration/` — compact V1 policy and environment facts.
- `2026-09-05-qwen3-8b-t2-stress/attempts.json` — invalid generator and corrected qualification.
- `manifests/historical-artifacts.json` — deterministic inventory, checksums, and disposition of every
  inspected archive.

The repository's fake-server smoke results remain protocol validation and must never be promoted to
performance evidence. Real-GPU findings must stay scoped to their recorded environment and workload.
