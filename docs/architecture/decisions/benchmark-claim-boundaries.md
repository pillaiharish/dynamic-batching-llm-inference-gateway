# Benchmark claim boundaries

The A/B/C arms are:

- A: client directly to vLLM.
- B: client through the gateway to vLLM, compatible aggregation disabled.
- C: client through gateway aggregation to vLLM's batch endpoint, compatible aggregation enabled.

Therefore B−A estimates gateway/router overhead, while C−B is the end-to-end effect of
gateway-side compatible aggregation. C−B must not be called simply “GPU batching”: vLLM already
performs continuous batching. The difference includes compatibility grouping, gateway wait and
formation, backend selection, HTTP batch submission, vLLM scheduling/GPU execution, and response
handling.

An artifact's `valid=true` means only that it passed that schema/validator. A performance claim also
requires experiment-level success criteria, complete requests, stable health, correct workload,
comparable arms, and stated limitations. Debug timing from capacity stress is not benchmark data.
The 2026-08-30 clean C=8 run is directional replication across three repetitions, not universal
statistical proof.
