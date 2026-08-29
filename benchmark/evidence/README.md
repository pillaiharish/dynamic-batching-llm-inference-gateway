# Curated benchmark evidence

No performance conclusion is valid without a real run artifact identifying hardware, model, vLLM
version and launch configuration, workload and dataset digest, gateway Git SHA, benchmark harness
SHA, exact gateway configuration, and benchmark configuration.

This directory is for small, deliberately reviewed evidence artifacts such as a manifest, compact
summary CSV/JSON, source-backed plots, and a short neutral conclusion. Raw run directories belong in
the Git-ignored `benchmark/results/` tree. Never commit API keys, Authorization headers, cloud
credentials, generated model content, or enormous raw traces.

Real-GPU A/B/C performance evidence has not yet been collected. The repository's fake-server smoke
results are protocol validation and must never be promoted to performance evidence.
