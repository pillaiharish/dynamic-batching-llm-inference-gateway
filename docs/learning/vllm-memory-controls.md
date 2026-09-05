# vLLM memory controls

`gpu_memory_utilization` participates in startup admission and automatic KV sizing.
`kv_cache_memory_bytes` fixes the cache allocation but does not disable the utilization admission
guard. For two Qwen3-8B replicas on the tested H100, neither control alone supplied the required
symmetry and admission behavior; their calibrated combination did. Treat reported KV capacity,
idle HBM, and runtime allocation as different observations. The active replica's post-stress HBM
increase was runtime allocation, not evidence that fixed KV size changed.
