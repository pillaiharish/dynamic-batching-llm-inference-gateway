# Fixed KV did not remove default memory admission

Evidence confidence: `ARTIFACT_BACKED`

With only `kv_cache_memory_bytes=18,700,000,000`, the first Qwen3-8B engine started with 17.42 GiB KV,
253,632 tokens, and 61.92x reported maximum concurrency at 4096. The second failed before startup:
vLLM retained default `gpu_memory_utilization=0.92` and required roughly 72.85 GiB free when only about
44.3 GiB remained.

Explicit KV bytes control cache allocation, not the separate startup admission guard. The corrected
policy specified both fixed KV and `gpu_memory_utilization=0.44`, then verified both startup orders.
