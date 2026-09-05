# 2026-08-30 clean C=8 confirmation

Status: valid directional replication; not broad statistical proof  
Evidence confidence: `ARTIFACT_BACKED`

## Question and procedure

After the invalid targeted matrix, would the C=8 C−B direction reproduce with complete requests and
stable health? The same Qwen3.8-27B/vLLM 0.26.0 configuration and gateway SHA `4473e78...` ran a
fresh C=8-only A/B/C matrix: three rotated repetitions, 20 warmups and 300 measured requests per arm,
non-streaming `n=1`, temperature 0, top_p 1, max_tokens 128, seed 1.

## Results and interpretation

All 2700/2700 measured requests succeeded, counters stayed monotonic, and no backend health transition
or `no_healthy_backend` event occurred. Every repetition favored C−B for p95, request throughput, and
output throughput. The median repetition reported p95 −532.499 ms, +0.177 req/s, and +17.157 output
tok/s; the other C−B p95 values were −558.900 and −521.163 ms.

This supports directional replication at C=8 for the exact single-host workload. Three repetitions do
not establish universal improvement, statistical significance, other concurrency behavior, or a GPU
batching claim. Re-run when the frozen environment or workload changes. Archive SHA-256:
`646193ac996a2eaff8274a679f970cb10ce12c83aa3ac9485aea0ce30e0a02e8`.
