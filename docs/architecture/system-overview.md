# System overview

The gateway accepts OpenAI-compatible Chat Completions, applies authentication and admission,
routes to healthy vLLM backends, and optionally aggregates compatible non-streaming requests into
vLLM's batch HTTP endpoint. vLLM—not the gateway—owns continuous batching, KV-cache management,
scheduling, and GPU execution.

The reproducibility boundary has three layers:

1. Terraform creates the billable compute, disk, network/SSH inputs, optional persistent volume,
   and bootstrap metadata: the laboratory.
2. The deploy lifecycle establishes and verifies a T0/T1/T2/T3 runtime topology.
3. The benchmark campaign performs isolated measurements and captures evidence.

Chakra Vault remains outside this repository and owns model acquisition, hash/provenance checks,
manifesting, and safe restoration. Terraform does not iterate benchmarks, and benchmark code does
not provision infrastructure.
